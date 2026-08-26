# KV Transfer Compatibilities in Prefill/Decode Disaggregation

**Interview question:** If a Prefill worker computes KV cache and a Decode worker consumes it, what exactly must be compatible? Which differences can be handled by an exact transformation, and which differences require recomputation or an approximate semantic mapper?

The core mental model is:

> **KV compatibility is not one boolean. We must distinguish semantic/value compatibility from physical/layout compatibility.**

A source and destination can contain the **same mathematical KV state** but store or shard it differently. That can often be repaired exactly. If they are running different learned models, however, the KV *values themselves* may live in different learned representation spaces. That is a much harder problem.

---

# 1. Start from the KV Cache Mathematically

For Transformer layer `l`, simplified:

\[
K_l = H_l W_l^K
\]

\[
V_l = H_l W_l^V
\]

and in many architectures positional encoding such as RoPE is applied to keys:

\[
K_l^{rope} = R_\theta(position)K_l
\]

During Decode, the target model produces a new query:

\[
Q_{new}
\]

and consumes the cache:

\[
Attention(Q_{new},K,V)
\]

The most important compatibility statement is:

> **A Decode model expects the KV state that its own mathematical model would have produced for the same token context.**

Everything else follows from this.

---

# 2. Two Fundamental Compatibility Dimensions

## 2.1 Semantic / Value Compatibility

Question:

> Are these the mathematical K/V values the destination Decode model expects?

This depends on:

```text
model architecture
model weights / exact revision
adapter / LoRA state
attention architecture
KV head count / head dimension
position encoding / RoPE semantics
token sequence
tokenizer / prompt-template semantics
model-specific execution-state representation
```

If semantic compatibility fails, rearranging bytes cannot fix it.

## 2.2 Physical / Layout Compatibility

Question:

> Assuming the K/V values are mathematically correct, are they arranged in the representation the Decode engine expects?

This depends on:

```text
TP sharding
PP / layer ownership
KV tensor layout (HND / NHD / engine-specific)
paged block size
physical block mapping
KV dtype / quantization encoding
attention backend layout
alignment / packing
engine-specific cache format
```

If semantic compatibility holds but the physical representation differs, we can often perform an **exact transform**:

```text
split
gather
transpose
reblock
reshard
redistribute layers
```

without rerunning the Transformer Prefill.

---

# 3. Four Useful Compatibility Levels

| Level | Example | Action |
|---|---|---|
| **Exact Direct** | same model + same KV representation | copy bytes |
| **Exact Transform** | same model, P TP2 -> D TP4 | deterministic reshape / redistribution |
| **Semantic Mapping** | one family model -> another family model | learned / mathematical approximation |
| **Incompatible** | no trusted transform or map | target re-Prefill |

```text
DIRECT:
    P KV --copy--> D KV

EXACT TRANSFORM:
    P logical KV
         |
         | deterministic conversion
         v
    D logical KV

SEMANTIC MAP:
    Model A KV
         |
         | learned/fitted mapper
         v
    approximate Model B KV

INCOMPATIBLE:
    target model runs Prefill
```

---

# 4. Same Model + Different Tensor Parallelism

Suppose a model has eight KV heads.

Prefill TP2:

```text
P rank 0:
    heads 0 1 2 3

P rank 1:
    heads 4 5 6 7
```

Decode TP4:

```text
D rank 0:
    heads 0 1
D rank 1:
    heads 2 3
D rank 2:
    heads 4 5
D rank 3:
    heads 6 7
```

The logical K/V values are identical. We only redistribute head shards:

```text
             PREFILL / TP2

          P0             P1
       [0..3]          [4..7]
       /    \          /    \
      v      v        v      v

     D0      D1       D2      D3
   [0,1]   [2,3]    [4,5]   [6,7]

              DECODE / TP4
```

No Transformer forward pass is required.

### Principle

> **Different TP does not inherently mean incompatible KV. It means a trusted head/rank transformation must exist.**

Current vLLM NIXL supports heterogeneous TP under documented model/layout restrictions, and TensorRT-LLM supports different context/generation parallel strategies through KV layout conversion.

---

# 5. Different Pipeline Parallelism

Example:

```text
Prefill PP2:
    P0 -> layers 0-19
    P1 -> layers 20-39

Decode PP1:
    D0 -> layers 0-39
```

The mathematical layer KV remains the same. The exchange layer must remap layer ownership:

```text
P0 layer 0  -> D layer 0 destination
...
P0 layer 19 -> D layer 19 destination

P1 layer 20 -> D layer 20 destination
...
P1 layer 39 -> D layer 39 destination
```

This is an exact distribution problem, not model recomputation.

---

# 6. Tensor Layout: HND vs NHD

Two runtimes may store identical KV with different dimension order.

```text
HND:
    [head][token][head_dim]

NHD:
    [token][head][head_dim]
```

Exact conversion:

```text
HND
 |
 | transpose / permutation
 v
NHD
```

Current vLLM NIXL defaults to HND for non-MLA models, supports NHD, and documents an experimental local HND<->NHD permutation path under restrictions.

---

# 7. KV Block / Page Size

Prefill:

```text
16-token blocks:
[0-15][16-31][32-47][48-63]
```

Decode:

```text
32-token blocks:
[0----------------31]
[32---------------63]
```

Logical token-indexed KV is unchanged.

Exact conversion may require:

```text
reblocking
packing
multiple source blocks -> one destination block
block-table remapping
```

Current vLLM supports some heterogeneous block-size combinations subject to architecture/layout restrictions.

---

# 8. KV Dtype and Quantization

Suppose:

```text
P KV = BF16
D KV = FP8
```

This is no longer only an address permutation.

We need:

```text
BF16 KV
    |
    | quantization
    | destination-compatible scale policy
    v
FP8 KV
```

The destination must agree on:

```text
cache dtype
FP8 variant
scale format
scale granularity
scale storage
packing
```

Current vLLM takes a conservative approach: P and D must use the same `cache_dtype`; unsupported scale-transfer cases fail the compatibility handshake.

### Principle

> **Quantization metadata is part of KV representation compatibility, not an incidental implementation detail.**

---

# 9. Attention Backend Compatibility

Example:

```text
P:
    FlashAttention-style context backend

D:
    FlashInfer / decode-specialized backend
```

They may run the same model and produce mathematically identical K/V while expecting different physical cache representations:

```text
dimension order
page metadata
alignment
fused cache layout
quantized layout
```

Possible outcomes:

```text
same representation
    -> direct copy

different but exactly convertible
    -> transform

different semantic cache representation
    -> incompatible without a special conversion
```

This is why backend/cache-layout identity participates in production compatibility checking.

---

# 10. Same Shape Does NOT Mean Same Semantics

Two models can have:

```text
same layer count
same KV head count
same head dimension
same tensor shape
```

and still produce incompatible KV.

For model A:

\[
K_A = H_A W_A^K
\]

For B:

\[
K_B = H_B W_B^K
\]

B's Decode query expects:

\[
Q_B K_B^T
\]

not:

\[
Q_B K_A^T
\]

Therefore:

```text
shape compatibility
    !=
semantic compatibility
```

This distinction is the foundation for understanding cross-model KV research.

---

# 11. Model Revision and Adapter Identity

Even "same base model" workers can differ:

```text
P:
    revision 17
    LoRA support-bot-v3

D:
    revision 17
    no LoRA
```

If the adapter changes hidden states or attention projections, the resulting KV can differ.

A semantic compatibility namespace should therefore include the exact model revision and any adapter state that affects KV.

---

# 12. Tokenizer / Exact Token Context

KV represents an exact token prefix, not raw text.

If two models tokenize the same text differently:

```text
Model A:
    [un, believable]

Model B:
    [unbel, iev, able]
```

then cache position 137 does not represent the same token position.

Same-model P/D avoids this naturally.

Cross-model transfer requires compatible tokenization or explicit sequence alignment.

---

# 13. Chat Template / Special Tokens

Even with the same tokenizer vocabulary, different prompt serialization can generate different token sequences.

Therefore direct reuse requires:

```text
same exact token prefix
```

not merely the same logical conversation text.

The tokenizer/template revision therefore belongs in a robust logical KV namespace.

---

# 14. RoPE / Position Encoding

Cached keys may be position-rotated:

\[
K^{rope}=R_\theta(position)K
\]

Differences in:

```text
RoPE theta
RoPE scaling
position indexing
context-extension policy
sliding-window semantics
```

can make keys incompatible.

This is semantic compatibility, not just physical layout.

---

# 15. Attention Architecture

"KV cache" is not one universal state representation.

Examples:

```text
MHA
MQA
GQA
MLA
sliding-window attention
hybrid attention + Mamba/SSM
```

For GQA:

```text
Model A: 8 KV heads
Model B: 4 KV heads
```

No automatic one-to-one mapping exists.

MLA may store compressed latent state rather than conventional per-head K/V.

Hybrid models may require recurrent state in addition to standard KV.

Production runtimes therefore have architecture-specific compatibility restrictions.

### Principle

> **KV cache is an abstraction; actual execution state is architecture specific.**

---

# 16. Runtime Validity / Epoch Fencing

Static representation can be compatible while runtime metadata is stale.

If:

```text
R1 epoch 14
```

restarts as:

```text
R1 epoch 15
```

old GPU addresses/manifests are invalid.

Safe transfer also requires:

```text
source epoch
destination epoch
allocation ID
context digest
block-manifest version
```

This is not semantic compatibility, but it is part of transfer correctness.

---

# 17. A Better Compatibility Contract

A conservative V1 can use:

```text
source.kv_compatibility_id
    ==
destination.kv_compatibility_id
```

But heterogeneous P/D serving suggests a richer conceptual model:

```text
KVCompatibilityProfile {
    semantic_space_id

    model_id
    model_revision
    adapter_id

    tokenizer_template_id
    position_encoding_id
    attention_architecture_id

    logical_kv_shape

    physical_layout_id
    kv_dtype
    quantization_scheme_id
    block_size

    tp_mapping
    pp_mapping
    attention_backend_layout_id

    supported_exact_transforms[]
}
```

Classification:

```text
same semantic space
+ same physical layout
    -> DIRECT

same semantic space
+ trusted exact transform
    -> EXACT_TRANSFORM

different semantic space
+ qualified semantic mapper
    -> SEMANTIC_MAPPING

otherwise
    -> RECOMPUTE
```

---

# 18. NVIDIA Dynamo

Current Dynamo deployment guidance is conservative: Prefill and Decode should agree on the model and important KV properties such as dtype, block size and KV layout.

Dynamo's deeper disaggregated-serving design also handles different layouts caused by different parallelism. For different TP layouts, it describes applying a high-performance KV transpose/layout kernel in the NIXL transfer path.

So Dynamo is primarily solving:

```text
same logical model state
+
possibly different physical P/D representation
```

not arbitrary cross-model semantic conversion.

---

# 19. vLLM NIXL Compatibility

vLLM currently exposes one of the clearest explicit contracts.

Its NIXL connector performs a compatibility handshake before transfer.

Supported differences can include, under restrictions:

```text
heterogeneous TP
some heterogeneous block sizes
different physical number of KV blocks
some HND/NHD conversion cases
different speculative draft depths
```

The contract validates model/cache properties including architecture, logical KV dimensions, cache dtype and layout/backend compatibility.

### Design Lesson

> **Handshake first; perform only explicitly supported exact transformations; reject unsafe combinations rather than guessing.**

---

# 20. TensorRT-LLM KV Cache Exchange

TensorRT-LLM separates:

```text
KV Cache Manager
KV Cache Exchange
communication backend
```

KV Cache Exchange owns:

```text
transmission
reception
cache-space release
layout conversion
```

and supports backends including:

```text
NIXL
UCX
MPI
```

It explicitly supports context and generation instances using different parallel strategies and performs the required block/rank/layer mapping.

This is a strong example of **Exact Transform** being a first-class serving subsystem.

---

# 21. NIXL's Role

NIXL sees:

```text
source descriptors
        ->
destination descriptors
```

It does not know:

```text
layer number
TP head ownership
RoPE semantics
model revision
whether Model A KV is valid for Model B
```

Therefore:

> **NIXL solves transport; the serving runtime solves Transformer-state compatibility.**

---

# 22. Mooncake / LMCache-Class Systems

Distributed KV stores and transfer systems generally focus on:

```text
store
lookup
move
offload
restore
tier across GPU / CPU / storage
```

They still require a logical cache namespace/contract ensuring that the consumer interprets the state correctly.

They do not automatically convert one model's learned KV space into another.

---

# 23. Cross-Model KV Cache Transfer Paper

**Paper:** *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse* — Heo et al., August 2026.

This attacks a different problem.

Ordinary heterogeneous P/D:

```text
Qwen3-32B TP2
       |
       | exact re-sharding
       v
Qwen3-32B TP4
```

The paper asks whether we can do:

```text
Qwen3-14B KV
       |
       | learned linear mapper
       v
approximate Qwen3-32B KV
```

so the larger model can skip its own Prefill.

That is **semantic mapping**.

---

# 24. Matched-KV Model Pairs

The paper studies model-family pairs with structurally aligned KV interfaces such as matching KV-head count and per-head dimensions, plus compatible tokenization assumptions.

This makes tensor mapping feasible, but does not make the representations identical.

For each target layer/head, the authors identify predictive source layers and fit closed-form ridge mappings.

Conceptually:

```text
source layer a KV
source layer b KV
source layer c KV
        |
        v
concatenate
        |
        v
ridge mapper
        |
        v
target layer/head KV approximation
```

K and V are handled separately.

---

# 25. Why the Paper Removes RoPE

The authors strip positional rotation from keys before learning the mapping:

```text
source RoPE K
     |
     | inverse positional rotation
     v
source content-space K
     |
     | linear mapper
     v
target content-space K
```

This makes the mapping less tied to particular token positions/context lengths.

It cleanly separates:

```text
learned semantic representation
```

from:

```text
position-dependent representation
```

---

# 26. Shape Compatibility Is Still Not Sufficient

The paper finds substantial variation across structurally matched model pairs.

Some retain a large fraction of target standalone quality; others degrade sharply.

The systems lesson is:

> **Matching tokenizer, KV-head count and head dimension helps, but does not prove semantic compatibility.**

Cross-model semantic transfer must be empirically qualified.

---

# 27. Raw KV Error Is Not Enough

Decode consumes mapped K through:

\[
Q_{target}\hat K^T
\]

and mapped V through the resulting attention weights.

So tensor reconstruction error alone does not tell us how errors affect the target attention computation.

The paper reports attention-output similarity as a more meaningful indicator of downstream retention than raw KV reconstruction quality.

A production semantic mapper therefore needs qualification on:

```text
attention behavior
downstream task quality
context-length slices
multi-turn behavior
OOD workloads
```

not only tensor MSE/R².

---

# 28. Productionizing Semantic Mapping

If cross-model mapping becomes a production mechanism, compatibility could become:

```text
DIRECT_EXACT

EXACT_TRANSFORM(
    transform_id
)

SEMANTIC_MAP(
    mapper_id,
    quality_profile,
    latency_profile,
    supported_context_range
)

INCOMPATIBLE
```

The mapper becomes a versioned serving artifact:

```text
source model revision
target model revision
mapper version
calibration data/version
quality qualification
latency/cost profile
context-range constraints
fallback policy
```

This is much richer than ordinary NIXL transfer.

---

# 29. Decision Tree

```text
Same logical KV semantics?
       |
       +-- YES
       |     |
       |     v
       |  Same physical layout?
       |     |
       |     +-- YES -> DIRECT TRANSFER
       |     |
       |     +-- NO
       |          |
       |          v
       |      Exact transform supported?
       |          |
       |          +-- YES -> TRANSFORM + TRANSFER
       |          |
       |          +-- NO -> RE-PREFILL
       |
       +-- NO
             |
             v
       Qualified semantic mapper?
             |
             +-- YES -> SEMANTIC MAP
             |
             +-- NO -> TARGET RE-PREFILL
```

---

# 30. Compatibility Checklist

## Semantic

```text
same model/revision?
same adapter?
same exact token prefix?
same tokenizer/template semantics?
same position/RoPE semantics?
same attention architecture?
same logical KV-head/dim semantics?
```

## Physical

```text
same TP?
if not, supported reshard?

same PP?
if not, supported layer mapping?

same HND/NHD/engine layout?
same block size or supported reblocking?
same cache dtype / scale semantics?
same attention-backend cache representation?
```

## Runtime

```text
source epoch valid?
destination epoch valid?
source blocks still present?
source pin valid?
destination allocation valid?
context digest/version valid?
```

Fail closed to a known-safe path if any required invariant is uncertain.

---

# 31. Short Interview Answer

> **I separate KV compatibility into semantic compatibility and physical compatibility. Semantic compatibility asks whether the Prefill worker produced the mathematical K/V values the Decode model expects for the exact token context; that depends on model revision, adapters, tokenization, RoPE and attention architecture. Physical compatibility asks how those correct values are represented—TP/PP sharding, HND/NHD layout, page size, cache dtype and backend layout. If both match, transfer is direct. If semantics match but layout differs, systems such as vLLM and TensorRT-LLM can do exact head/rank/layout transformations. If the model itself differs, matching tensor shapes are not enough because the learned KV spaces differ. The cross-model KV paper introduces a third category: an empirically qualified semantic mapper. NIXL is below all of this—it moves buffers but does not determine whether those buffers represent valid target-model KV.**

---

# 32. Memory Note

```text
KV COMPATIBILITY

SEMANTIC
    model/revision
    adapter
    exact token prefix
    RoPE/position semantics
    attention architecture
    logical KV shape

PHYSICAL
    TP
    PP
    HND/NHD
    block size
    dtype/quantization
    backend layout

same semantic + same physical
    -> DIRECT

same semantic + different physical
    -> EXACT TRANSFORM

different semantic + qualified mapper
    -> SEMANTIC MAP

otherwise
    -> RE-PREFILL

Systems:
    Dynamo     -> conservative matching + layout transform
    vLLM       -> compatibility handshake + supported heterogeneity
    TensorRT   -> KV Cache Exchange + layout conversion
    NIXL       -> transport only
```

---

# References

- NVIDIA Dynamo — Disaggregated Serving.
- vLLM — NixlConnector Compatibility Matrix, August 2026.
- TensorRT-LLM — Disaggregated Serving / KV Cache Exchange / Cache Layout Transformation.
- Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*, arXiv:2608.03893, August 2026.
