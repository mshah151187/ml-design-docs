# Structural Parallelism Analysis for Context Parallelism (CP)

> **Purpose:** Determine whether a proposed Context Parallel degree is structurally legal before evaluating performance. This phase checks whether a sequence can be partitioned across ranks, whether distributed attention preserves the model's exact attention semantics, whether KV ownership is well-defined, whether position/mask behavior remains correct, and whether the runtime/attention backend supports the required context-sharded execution.

> **Terminology note:** Different runtimes and papers may use related terms such as context parallelism, sequence/context sharding, ring attention, or backend-specific distributed-attention modes. The structural reasoning here is intentionally generic: the sequence/token dimension is partitioned while attention still needs logically correct access to the full required context.

---

# 1. Natural-Language Interpretation of CP

Context Parallelism means:

> **Split one long sequence across GPUs so different ranks own different token ranges of the same request, while collectively executing attention as if the full sequence were present.**

Example:

```text
Sequence length = 128K tokens
CP              = 4
```

A simple partition is:

```text
GPU0:
    tokens 0      ... 31,999

GPU1:
    tokens 32,000 ... 63,999

GPU2:
    tokens 64,000 ... 95,999

GPU3:
    tokens 96,000 ... 127,999
```

Natural-language interpretation:

> **Instead of every GPU holding the entire book, each GPU holds a different chapter range — but when a token needs to attend to earlier chapters, the GPUs must cooperate so the result is identical to full-context attention.**

The central structural question is:

> **Can the model's attention computation be decomposed across token ranges while preserving exact positional, masking, KV, and attention semantics?**

---

# 2. CP vs TP vs EP vs PP

Keep the object being partitioned explicit:

```text
TP:
    split tensors/operators inside a layer

EP:
    split experts across GPUs

PP:
    split the layer stack across GPUs

CP:
    split the token/context dimension of the same sequence
```

Natural-language interpretation:

```text
TP:
    divide one calculation

EP:
    distribute specialists

PP:
    divide the assembly line

CP:
    divide the document / conversation itself
```

This is why CP has very different structural checks.

---

# 3. Why CP Exists

Long-context inference creates two major pressures:

```text
1. Attention work grows with context length

2. KV cache memory grows with context length
```

For one very long sequence, a single rank may face:

```text
too much KV memory

too much attention work

too much per-request latency
```

CP attempts to distribute that pressure across ranks by partitioning sequence positions.

Natural-language interpretation:

> **When one request itself is too large, CP lets several GPUs collaborate on different portions of that same request's history.**

This is different from DP, where different GPUs handle different requests.

---

# 4. Main Structural Checks

```text
CP Structural Analysis

1. Context Partitionability
   - sequence/token dimension partitionable?
   - chunk boundaries legal?
   - uneven final chunk supported?

2. Attention Semantics
   - distributed attention backend available?
   - each query can access all required K/V?
   - causal mask preserved?
   - softmax normalization remains globally correct?

3. Position / Mask Semantics
   - RoPE / positional encoding supports global positions?
   - position IDs remain correct after sharding?
   - sliding-window / local-attention rules preserved?

4. KV Ownership
   - which rank owns which KV positions?
   - KV metadata/page ownership valid?
   - remote KV access/exchange supported?
   - decode-time ownership semantics valid?

5. Model / Attention Compatibility
   - MHA / GQA / MQA supported?
   - special attention patterns supported?
   - quantized KV representation supported under CP?

6. Composition Requirements
   - CP + TP?
   - CP + PP?
   - CP + EP?
   - process-group composition supported?

7. Workload Relevance
   - context long enough to justify generating CP candidates?
   - long-context traffic material enough?

8. Structural Result
   - VALID
   - VALID_WITH_UNEVEN_CHUNKS
   - VALID_FOR_LONG_CONTEXT
   - REQUIRES_SUPPORTED_ATTENTION_BACKEND
   - UNSUPPORTED
```

---

# 5. Context Partitionability

Suppose:

```text
sequence_length = 128K
CP = 4
```

Then:

```text
tokens_per_rank = 32K
```

which gives a clean partition.

But if:

```text
sequence_length = 130K
CP = 4
```

then:

```text
130K / 4
```

does not divide evenly.

Unlike an attention head in TP, token ranges can often be uneven:

```text
GPU0: 32.5K
GPU1: 32.5K
GPU2: 32.5K
GPU3: 32.5K
```

or, depending on chunk granularity:

```text
GPU0: 32K
GPU1: 32K
GPU2: 32K
GPU3: 34K
```

The exact legal partition depends on the backend's chunk/block requirements.

So:

```text
sequence_length % CP != 0
```

does not necessarily mean invalid.

It may imply:

```text
Required Capability:
    UNEVEN_CONTEXT_PARTITION
```

or:

```text
padding to supported block/chunk size
```

Natural-language interpretation:

> **Can we divide the document into legal token ranges that the distributed-attention implementation knows how to process?**

---

# 6. Why Attention Semantics Are the Hard Part

Suppose a token on GPU3 is at position:

```text
100,000
```

For causal self-attention, it may need to attend to:

```text
tokens 0 ... 100,000
```

But GPU3 locally owns only:

```text
96K ... 128K
```

So the token's required K/V history is distributed across:

```text
GPU0
GPU1
GPU2
GPU3
```

Natural-language interpretation:

> **A GPU owns only one chapter of the conversation, but a query token may need information from all previous chapters.**

That is why CP requires a distributed-attention algorithm, not just simple tensor slicing.

---

# 7. Global Attention Must Still Be Exact

For one query vector `q`, ordinary attention computes:

```text
scores = q K^T

weights = softmax(scores)

output = weights V
```

Under CP, `K` and `V` are partitioned:

```text
K = [K0, K1, K2, K3]
V = [V0, V1, V2, V3]
```

Each rank can compute local scores:

```text
scores_i = q K_i^T
```

But a local softmax is not enough.

Why?

Because:

```text
softmax over K0
```

plus:

```text
softmax over K1
```

is not equal to:

```text
softmax over [K0, K1]
```

The normalization denominator must be global across all visible keys.

Natural-language interpretation:

> **Each chapter can compute how relevant its own words are, but all chapters must agree on one global probability distribution.**

So the distributed attention backend must preserve:

```text
global max
global softmax denominator
global weighted value sum
```

or an equivalent mathematically correct formulation.

This is one of the core structural requirements of CP.

---

# 8. Causal Mask Semantics

For autoregressive attention, token `t` may only attend to:

```text
positions <= t
```

Suppose:

```text
GPU0 owns 0-31K
GPU1 owns 32K-63K
GPU2 owns 64K-95K
GPU3 owns 96K-127K
```

A query at position:

```text
70K
```

on GPU2 may attend to:

```text
GPU0:
    all local positions

GPU1:
    all local positions

GPU2:
    only positions <= 70K

GPU3:
    nothing
```

So causal visibility depends on:

```text
global token position
```

not merely local position inside a rank.

Natural-language interpretation:

> **Splitting the document must not make a token accidentally see the future.**

Structural CP Analysis must therefore ensure that:

```text
global causal mask
```

is preserved across rank boundaries.

---

# 9. Position IDs and RoPE

Most decoder LLMs use positional information such as RoPE.

Suppose GPU2 owns:

```text
tokens 64K ... 95K
```

The first local token on GPU2 is not global position `0`.

It is:

```text
position = 64K
```

So local execution must preserve global position IDs.

Conceptually:

```text
GPU0:
    local index 0 -> global position 0

GPU1:
    local index 0 -> global position 32K

GPU2:
    local index 0 -> global position 64K

GPU3:
    local index 0 -> global position 96K
```

Natural-language interpretation:

> **Each GPU owns a different chapter, but page numbering must remain global.**

If RoPE, ALiBi, relative position biases, or custom position schemes are used, the distributed backend must preserve their semantics.

---

# 10. Sliding-Window / Local Attention

Some models do not use full attention over all previous tokens.

They may use:

```text
sliding window attention

local attention

global + local patterns
```

Suppose the window is:

```text
8K tokens
```

A query near a context partition boundary may need:

```text
some K/V from the local rank
+
some K/V from the previous rank
```

So context boundaries must not truncate the logical window.

Natural-language interpretation:

> **A chapter boundary cannot change which previous pages the model is allowed to read.**

This becomes a model-specific structural check.

---

# 11. KV Ownership

Under CP, a natural ownership rule is:

```text
each rank owns KV for its token positions
```

Example:

```text
GPU0:
    K/V for positions 0-31K

GPU1:
    K/V for positions 32K-63K

GPU2:
    K/V for positions 64K-95K

GPU3:
    K/V for positions 96K-127K
```

Natural-language interpretation:

> **Each GPU stores the memory associated with the chapter it owns.**

This can reduce per-rank KV residency dramatically.

If total KV is:

```text
80 GB
```

then ideal CP4 sequence partitioning might reduce sequence-position KV payload toward:

```text
~20 GB/rank
```

before other sharding/replication effects.

---

# 12. CP and KV Cache Are Naturally Related

Without CP, one rank may need KV for the entire long sequence.

With CP:

```text
KV(sequence positions)
```

is distributed.

This is one of the most important reasons CP becomes attractive for very long contexts.

Natural-language interpretation:

> **Instead of every GPU storing the full conversation history, each GPU stores part of the history and the attention algorithm coordinates access.**

---

# 13. KV Metadata and Paging

Serving runtimes often manage KV using blocks/pages.

So the actual ownership may be:

```text
Rank 0:
    KV pages for context chunk A

Rank 1:
    KV pages for context chunk B
```

rather than one contiguous tensor.

Structural analysis must ensure that:

```text
block tables
page ownership
position mapping
attention metadata
```

can represent distributed context ownership.

Search Foundation does not need to reproduce the exact block-table implementation.

It only needs to derive:

```text
Required Capability:
    DISTRIBUTED_KV_OWNERSHIP
```

and let the runtime capability validator / smoke test prove it.

---

# 14. Decode-Time Semantics Are Different from Prefill

During prefill, there are many context tokens and CP is easy to motivate:

```text
128K prompt
    ->
split across four ranks
```

During autoregressive decode, each request produces one new token per step.

Now the new token must attend to the already distributed KV history.

Natural-language interpretation:

> **The new token is one question, but its memory is spread across several GPUs.**

The runtime needs a decode-time distributed-attention path that can query all relevant KV partitions.

This is why a runtime may support some form of context sharding for prefill but have different or more restrictive decode support.

Structural Analysis should distinguish:

```text
prefill_context_parallel_support

decode_context_parallel_support
```

when the runtime exposes them separately.

---

# 15. MHA / GQA / MQA Interaction

CP partitions by sequence position, so unlike TP it does not fundamentally require:

```text
num_heads % CP == 0
```

because heads are not the primary partition axis.

However, the attention backend still has to support the model's attention form:

```text
MHA
GQA
MQA
```

under distributed context execution.

For example, GQA may have:

```text
64 query heads
8 KV heads
```

and CP may shard:

```text
sequence positions
```

while TP simultaneously shards:

```text
heads
```

This leads to composition:

```text
TP:
    head dimension / projection partition

CP:
    sequence dimension partition
```

Natural-language interpretation:

> **TP divides how many attention perspectives each GPU computes; CP divides how much history each GPU owns.**

They are orthogonal in principle, but runtime support for the combination must be checked.

---

# 16. Quantized KV Under CP

Suppose:

```text
KV dtype = FP8
```

with some scale metadata.

Under CP, the KV payload and relevant metadata must remain valid per context partition.

Conceptually:

```text
GPU0:
    FP8 KV chunk 0
    associated scale metadata

GPU1:
    FP8 KV chunk 1
    associated scale metadata
```

If scale granularity is:

```text
per-block
```

and block ownership aligns with CP ownership, this can be natural.

If metadata is global or requires cross-partition interpretation, runtime support matters.

Natural-language interpretation:

> **When we split the history, each rank must still know how to interpret its compressed piece of that history.**

---

# 17. CP + TP Composition

Suppose:

```text
TP = 2
CP = 4
```

Conceptually:

```text
Context Group 0:
    token range 0-31K
    processed with TP2

Context Group 1:
    token range 32K-63K
    processed with TP2

Context Group 2:
    token range 64K-95K
    processed with TP2

Context Group 3:
    token range 96K-127K
    processed with TP2
```

Natural interpretation:

> **CP divides the book into chapters; TP lets multiple GPUs collaborate on the computation inside each chapter.**

The runtime needs two communication domains:

```text
TP collectives:
    inside tensor-parallel groups

CP communication:
    across context partitions for attention
```

So Structural Analysis derives:

```text
Required Capability:
    CP_TP_COMPOSITION
```

---

# 18. CP + PP Composition

Suppose:

```text
PP = 2
CP = 4
```

Each pipeline stage owns a subset of layers.

Within each stage, context is distributed across four context ranks.

Conceptually:

```text
PP Stage 0:
    Layers 0-23
    Context shards C0 C1 C2 C3

PP Stage 1:
    Layers 24-47
    Context shards C0 C1 C2 C3
```

The hidden states must move from Stage 0 to Stage 1 while preserving context-shard ownership.

Natural-language interpretation:

> **The assembly line is split into stages, and within every stage the long document is also split into chapters.**

This requires runtime support for:

```text
PP + CP process-group composition
```

and well-defined stage-to-stage transfer semantics for sharded hidden states.

---

# 19. CP + EP Composition

For an MoE model:

```text
CP:
    divides token positions

EP:
    distributes experts
```

A context shard may contain tokens that route to experts on many EP ranks.

So execution can involve both:

```text
distributed attention across context ranks
```

and:

```text
token dispatch across expert ranks
```

Natural-language interpretation:

> **First the long conversation is distributed by position; then tokens may still travel to whichever specialists their routers choose.**

This is structurally possible in principle, but runtime composition support becomes important.

---

# 20. Workload Relevance

Unlike TP or PP, CP may be structurally valid but pointless for short-context traffic.

Suppose:

```text
median input length = 2K
p99 input length    = 8K
```

Generating:

```text
CP = 8
```

would likely create tiny context shards.

So Structural Analysis should use traffic and serving requirements to annotate:

```text
CP_RELEVANCE
```

rather than treating every legal CP degree as equally useful.

Natural-language interpretation:

> **Do we actually have a document long enough to justify dividing it among several readers?**

This is not yet a performance ranking.

It is a coarse search-space relevance filter.

---

# 21. A Useful Relevance Rule

Suppose:

```text
max_context_requirement = 128K

traffic:
    60% short
    30% medium
    10% long
```

CP may still matter because the platform must support:

```text
long-context tail
```

even if the majority of traffic is short.

The analysis can emit:

```text
CP applicability:
    LONG_CONTEXT_ONLY
```

instead of:

```text
CP globally enabled for all requests
```

This may later influence serving-profile design.

---

# 22. Concrete Example

Assume:

```text
Model:
    48-layer decoder-only Transformer
    hidden_size = 8192
    query_heads = 64
    kv_heads = 8
    RoPE positional encoding
    causal self-attention

Serving Requirement:
    max_context = 128K

Traffic:
    60% short
    30% medium
    10% long

Quantization:
    W4A16
    FP8 KV

Runtime:
    candidate runtime R

Hardware:
    H100

Candidate:
    CP = 4
```

Now evaluate the structural questions.

---

# 23. Context Partition

```text
128K / 4 = 32K tokens/rank
```

So:

```text
context_partition:
    VALID
```

Placement:

```text
Rank 0:
    0-31K

Rank 1:
    32K-63K

Rank 2:
    64K-95K

Rank 3:
    96K-127K
```

---

# 24. Position Semantics

Each rank must use global positions.

```text
Rank 0 local 0 -> global 0

Rank 1 local 0 -> global 32K

Rank 2 local 0 -> global 64K

Rank 3 local 0 -> global 96K
```

If the attention backend preserves global RoPE position IDs:

```text
position_semantics:
    VALID
```

Otherwise:

```text
UNSUPPORTED
```

---

# 25. Causal Attention Semantics

Take a query at global position:

```text
70K
```

on Rank 2.

It may see:

```text
Rank 0:
    all keys

Rank 1:
    all keys

Rank 2:
    keys up to 70K

Rank 3:
    none
```

The distributed backend must preserve that global visibility exactly.

If supported:

```text
causal_mask:
    VALID
```

---

# 26. Global Softmax Semantics

Each rank can compute local attention statistics, but the final attention result must equal attention over the full visible context.

Required capability:

```text
DISTRIBUTED_ATTENTION_WITH_GLOBAL_NORMALIZATION
```

If the runtime/backend supports an equivalent mathematically exact distributed attention algorithm:

```text
attention_semantics:
    VALID
```

Otherwise:

```text
UNSUPPORTED
```

---

# 27. KV Ownership

With FP8 KV:

```text
Rank 0:
    FP8 K/V for positions 0-31K

Rank 1:
    FP8 K/V for positions 32K-63K

Rank 2:
    FP8 K/V for positions 64K-95K

Rank 3:
    FP8 K/V for positions 96K-127K
```

The associated scale metadata must remain valid for each rank's KV blocks.

If block/page ownership is compatible:

```text
kv_partition:
    VALID
```

---

# 28. Decode Support

After prefill, the next token appears at:

```text
position 128K
```

That token needs attention over the distributed KV history.

The runtime therefore needs:

```text
DECODE_WITH_DISTRIBUTED_KV
```

If only prefill CP is supported, then the candidate may not satisfy an online autoregressive serving requirement.

So:

```text
prefill_support:
    SUPPORTED

decode_support:
    REQUIRED
```

must be checked separately.

---

# 29. Workload Relevance

Because only 10% of traffic is very long:

```text
CP relevance:
    LONG_CONTEXT_FOCUSED
```

This does not make CP invalid.

It simply tells later search:

```text
do not assume CP should be used for every request shape
```

---

# 30. Example Structural Result

```text
CPStructuralAnalysis {

    cp_degree:
        4

    context_partition {
        max_context:
            128K

        tokens_per_rank:
            32K

        partition:
            EVEN

        status:
            VALID
    }

    attention_semantics {
        causal_attention:
            REQUIRED

        global_softmax:
            REQUIRED

        distributed_attention_backend:
            REQUIRED

        status:
            VALID_IF_RUNTIME_SUPPORTED
    }

    position_semantics {
        scheme:
            RoPE

        global_position_ids:
            REQUIRED

        status:
            VALID
    }

    kv_partition {
        dtype:
            FP8

        ownership:
            BY_CONTEXT_RANGE

        distributed_kv:
            REQUIRED

        decode_remote_kv_access:
            REQUIRED

        status:
            VALID_IF_RUNTIME_SUPPORTED
    }

    workload_relevance {
        max_context:
            128K

        long_context_fraction:
            10%

        applicability:
            LONG_CONTEXT_FOCUSED
    }

    runtime_requirements {
        context_parallelism:
            REQUIRED

        distributed_attention:
            REQUIRED

        global_softmax_semantics:
            REQUIRED

        distributed_kv_ownership:
            REQUIRED

        decode_context_parallel:
            REQUIRED
    }

    result {
        status:
            VALID_FOR_LONG_CONTEXT_IF_RUNTIME_SUPPORTED
    }
}
```

---

# 31. Recommended Output Contract

```text
CPStructuralAnalysis {

    cp_degree

    context_partition {
        max_context
        partition_granularity
        tokens_per_rank
        even_or_uneven
        padding_required
    }

    attention_semantics {
        attention_type
        causal_mask
        distributed_attention_required
        global_normalization_required
        special_attention_constraints
    }

    position_semantics {
        positional_encoding
        global_position_ids_required
        cross_partition_correctness
    }

    kv_partition {
        kv_dtype
        ownership_model
        block_page_alignment
        quantization_metadata_compatible
        prefill_support_required
        decode_support_required
    }

    composition_requirements {
        cp_tp
        cp_pp
        cp_ep
    }

    workload_relevance {
        max_context_requirement
        long_context_fraction
        applicability
    }

    coarse_runtime_validation {
        runtime
        version

        context_parallelism:
            SUPPORTED | UNSUPPORTED | UNKNOWN

        distributed_attention:
            SUPPORTED | UNSUPPORTED | UNKNOWN

        decode_context_parallel:
            SUPPORTED | UNSUPPORTED | UNKNOWN

        required_composition:
            SUPPORTED | UNSUPPORTED | UNKNOWN
    }

    result {
        status:
            VALID
            | VALID_WITH_UNEVEN_CHUNKS
            | VALID_FOR_LONG_CONTEXT
            | REQUIRES_SUPPORTED_ATTENTION_BACKEND
            | UNSUPPORTED

        constraints:
            [...]

        rejection_reasons:
            [...]
    }
}
```

---

# 32. Analysis-Time vs Execution-Time Validation

As with TP, EP, and PP:

```text
CP Structural Analysis
        |
        v
Coarse Runtime Capability Validation
        |
        v
Analyzed Candidate
        |
        v
Compatibility Smoke Test
        |
        +--> CP process groups initialize?
        +--> distributed attention backend resolves?
        +--> global positions/masks work?
        +--> distributed KV allocates?
        +--> short prefill succeeds?
        +--> decode over distributed KV succeeds?
        |
        v
Verified Executable Candidate
```

So:

```text
Search Foundation:
    predicts feasibility

Smoke Test:
    proves executability
```

---

# 33. What This Phase Does Not Decide

Structural CP Analysis does not yet answer:

```text
Which CP degree gives best TTFT?

How expensive is ring/all-gather communication?

When does CP outperform TP?

What is the best CP degree for 32K vs 128K?

How much does CP help KV capacity?

Should CP be enabled only above a context threshold?

Which CP topology minimizes cost/token?
```

Those belong to later communication analysis, benchmarking, and objective-driven search.

---

# 34. Interview Summary

> **Context Parallelism splits the token/context dimension of the same long sequence across GPUs. Structural CP analysis therefore checks whether context ranges can be legally partitioned, whether distributed attention preserves global softmax and causal semantics, whether positional encodings remain globally correct, whether KV ownership and quantization metadata remain valid across context shards, whether decode can attend over distributed KV, and whether the runtime supports the required CP composition with TP/PP/EP. Unlike TP, PP, or EP, the hard problem is not primarily weight placement — it is preserving exact attention semantics while sequence state is physically distributed.**

---

# 35. Locked Mental Model

Think of CP as:

```text
ONE VERY LONG CONVERSATION
        |
        v
SPLIT ITS HISTORY ACROSS GPUS
        |
        v
EACH GPU OWNS DIFFERENT TOKEN RANGES
        |
        v
ATTENTION STILL NEEDS A LOGICALLY GLOBAL VIEW
        |
        v
GPUS COOPERATE TO PRODUCE THE SAME ANSWER
AS FULL-CONTEXT ATTENTION
```

Then every structural question becomes:

```text
1. Can the context be partitioned?
2. Can attention still see exactly what it should see?
3. Are global positions and masks preserved?
4. Is KV ownership well-defined?
5. Can decode use distributed history?
6. Does the runtime/backend support the required composition?
```
