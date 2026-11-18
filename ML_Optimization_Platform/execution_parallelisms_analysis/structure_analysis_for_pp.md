# Structural Parallelism Analysis for Pipeline Parallelism (PP)

> **Purpose:** Determine whether a proposed Pipeline Parallel degree is structurally legal before evaluating performance. This phase checks whether the model's sequential layer stack can be divided into valid contiguous stages, whether stage-local weights/KV/runtime memory fit, whether special modules are assigned correctly, and whether the runtime supports the required PP composition.

---

# 1. Natural-Language Interpretation of PP

Pipeline Parallelism means:

> **Split the model's sequence of layers into contiguous stages, place different stages on different GPUs, and move the hidden state from one stage to the next.**

Example:

```text
num_layers = 48
PP         = 4

Stage 0 / GPU0:
    Layers 0-11

Stage 1 / GPU1:
    Layers 12-23

Stage 2 / GPU2:
    Layers 24-35

Stage 3 / GPU3:
    Layers 36-47
```

Execution becomes:

```text
Input
  |
  v
Stage 0
Layers 0-11
  |
  | hidden state
  v
Stage 1
Layers 12-23
  |
  v
Stage 2
Layers 24-35
  |
  v
Stage 3
Layers 36-47
  |
  v
Logits
```

The core structural question is:

> **Can the model's sequential layer stack be partitioned into valid contiguous stages such that every stage preserves model semantics, fits in memory, and can execute in the selected runtime?**

---

# 2. PP vs TP vs EP

Keep the distinctions explicit:

```text
TP:
    split work inside one layer

EP:
    distribute experts inside an MoE layer

PP:
    split the model along the layer/depth axis
```

Natural-language interpretation:

```text
TP:
    several GPUs collaborate inside one operation

EP:
    different specialists live on different GPUs

PP:
    different GPUs own different sections of the assembly line
```

---

# 3. Main Structural Checks

```text
PP Structural Analysis

1. Sequential Partitionability
   - enough partitionable blocks?
   - legal contiguous stage boundaries?
   - architecture has cross-stage dependencies?

2. Boundary Ownership
   - embedding placement valid?
   - final norm / LM head placement valid?
   - special/shared modules assigned correctly?

3. Stage Memory Feasibility
   - weights fit per stage?
   - stage-local KV fits?
   - runtime overhead fits?
   - MoE-heavy stages require rebalance?

4. Composition Requirements
   - PP + TP needed?
   - PP + EP needed?
   - stage-local topology valid?

5. Coarse Runtime Compatibility
   - runtime supports PP for this model?
   - stage slicing supported?
   - required composition supported?

6. Structural Result
   - VALID
   - VALID_WITH_UNEVEN_STAGES
   - REQUIRES_STAGE_REBALANCING
   - UNSUPPORTED
```

---

# 4. Enough Layers for the PP Degree

Suppose:

```text
num_layers = 48
PP = 4
```

Then:

```text
48 / 4 = 12 layers/stage
```

This is straightforward.

Now suppose:

```text
num_layers = 4
PP = 8
```

There are not enough complete Transformer blocks to give every stage meaningful work unless individual blocks are split internally, which is no longer ordinary layer-wise PP.

A coarse structural check is therefore:

```text
num_partitionable_blocks >= PP
```

Natural-language interpretation:

> **Do we have enough sequential chunks of work to give every pipeline stage something meaningful to own?**

A very high PP degree may still be structurally legal, but performance is a later question.

---

# 5. Legal Stage Boundaries

A decoder model often looks like:

```text
Embedding
Layer 0
Layer 1
...
Layer 47
Final Norm
LM Head
```

The cleanest PP boundaries are usually **between complete Transformer blocks**.

Example:

```text
Stage 0:
    Embedding
    Layers 0-11

Stage 1:
    Layers 12-23

Stage 2:
    Layers 24-35

Stage 3:
    Layers 36-47
    Final Norm
    LM Head
```

The boundary contract is simple:

```text
hidden_state in
    ->
stage
    ->
hidden_state out
```

Natural-language interpretation:

> **A good pipeline boundary is like handing a finished intermediate product to the next factory station.**

---

# 6. Why We Avoid Cutting Inside a Transformer Block

Suppose we split:

```text
Attention on Stage 0
MLP on Stage 1
```

Now the runtime may need to transfer internal state such as:

```text
residual inputs
attention outputs
possibly other operator-local tensors
```

That complicates stage semantics.

By contrast:

```text
complete Layer 11
    |
    | hidden state
    v
complete Layer 12
```

keeps residual logic and operator internals local to one stage.

Natural-language interpretation:

> **A clean PP boundary should not split a computation that still has unfinished internal dependencies.**

---

# 7. Contiguous Stage Ownership

Pipeline stages should normally own contiguous layer intervals.

Good:

```text
Stage 0:
    Layers 0-11

Stage 1:
    Layers 12-23
```

Bad conceptual partition:

```text
Stage 0:
    Layers 0,2,4,6

Stage 1:
    Layers 1,3,5,7
```

The second layout would cause the hidden state to bounce between devices after every layer.

Natural-language interpretation:

> **Pipeline stages should own consecutive parts of the assembly line.**

So structural analysis should normally require:

```text
stage_i = contiguous layer range
```

---

# 8. Model Implementation Must Support Stage Slicing

Even if the architecture is sequential on paper, the runtime/model implementation must be able to instantiate only the stage-local subset.

For example:

```text
Stage 1:
    Layers 12-23
```

should not also materialize:

```text
Embedding
Layers 0-11
Final Norm
LM Head
```

Conceptually the implementation needs something like:

```text
PipelineStage(
    layer_start = 12,
    layer_end   = 24
)
```

or an equivalent mechanism.

Natural-language interpretation:

> **Can the model implementation itself be broken into independently runnable chunks?**

This is a coarse runtime capability check; the smoke test later proves it.

---

# 9. Embedding Ownership

The embedding layer naturally belongs to the first stage.

```text
token IDs
   |
   v
Embedding on Stage 0
   |
   v
hidden states
   |
   v
later stages
```

Later stages do not need raw token IDs for normal decoder execution.

Natural-language interpretation:

> **Only the first factory station converts raw token IDs into the model's internal representation.**

So the topology artifact should record:

```text
embedding_owner = first stage
```

unless a runtime intentionally replicates it.

---

# 10. Final Norm and LM Head Ownership

The final stage usually owns:

```text
Final Norm
LM Head
```

because only the final hidden representation is converted to vocabulary logits.

```text
Stage N
    |
    v
Final Norm
    |
    v
LM Head
    |
    v
Logits
```

Natural-language interpretation:

> **Only the final factory station converts the finished hidden state into vocabulary scores.**

This matters because the LM head can be large and may make the final stage heavier than other stages.

---

# 11. Residual Connections and Stage Boundaries

Within a Transformer block, residuals are internal dependencies.

If stage boundaries occur between complete blocks, residual state stays local:

```text
Stage 0:
    complete Layer 11

boundary

Stage 1:
    complete Layer 12
```

Only the completed hidden state crosses the boundary.

This is one reason complete-block boundaries are structurally attractive.

---

# 12. What Crosses a PP Boundary?

For ordinary decoder PP, the main tensor crossing a stage boundary is:

```text
hidden_states
```

with shape roughly:

```text
[num_tokens, hidden_size]
```

During prefill:

```text
num_tokens
```

may be large.

During decode:

```text
num_tokens
```

is closer to active sequence count.

Conceptually:

```text
Stage i
    |
    | hidden state
    v
Stage i+1
```

Unlike TP, which often communicates every layer, PP communication primarily occurs at stage boundaries.

Performance implications are analyzed later.

---

# 13. KV Cache Ownership Under PP

Each pipeline stage owns a subset of attention layers.

Therefore it naturally owns the KV cache for those layers.

For:

```text
48 layers
PP = 4
```

with 12 layers/stage:

```text
Stage 0:
    KV for Layers 0-11

Stage 1:
    KV for Layers 12-23

Stage 2:
    KV for Layers 24-35

Stage 3:
    KV for Layers 36-47
```

Natural-language interpretation:

> **Each pipeline stage stores the history required by the attention layers it owns.**

So PP partitions both:

```text
weight memory
```

and:

```text
KV memory
```

along the layer dimension.

---

# 14. Why Per-Stage Weight Memory Must Be Checked

Suppose:

```text
model weights = 240 GB
usable HBM/GPU = 70 GB
PP = 4
```

A naive average suggests:

```text
240 / 4 = 60 GB/stage
```

But actual stages may differ.

For example:

```text
Stage 0:
    58 GB

Stage 1:
    61 GB

Stage 2:
    63 GB

Stage 3:
    74 GB
```

Then Stage 3 does not fit.

Natural-language interpretation:

> **The model may fit across four GPUs in aggregate while one pipeline stage still exceeds its local GPU memory.**

So:

```text
aggregate fit != per-stage fit
```

---

# 15. Stage Memory Includes More Than Weights

A stage's memory is closer to:

```text
M_stage =
    stage_weights
  + stage_local_KV
  + activation_memory
  + runtime_overhead
  + topology-dependent buffers
```

If one stage owns more attention layers, it also owns more KV.

So balancing only weight bytes is insufficient.

Natural-language interpretation:

> **Each station owns not only its machinery, but also the working state required by that machinery.**

---

# 16. Equal Layer Count Does Not Mean Equal Stage Cost

Suppose:

```text
48 layers
PP = 4
```

A simple split is:

```text
12 / 12 / 12 / 12
```

But stages may still differ because of:

```text
LM head
embeddings
MoE layers
different layer types
shared modules
different memory/runtime requirements
```

Therefore Structural Analysis should not require equal layer counts.

It should ask:

> **Can we find a legal contiguous partition where every stage fits?**

---

# 17. MoE Makes Stage Balance Harder

Suppose the layer sequence mixes dense and MoE blocks.

An MoE layer can have much larger resident weight memory.

Example:

```text
Stage 0:
    mostly dense layers

Stage 1:
    many MoE layers
```

Even with the same number of layers, Stage 1 can be much heavier.

Natural-language interpretation:

> **Some stations on the assembly line contain much heavier machinery.**

So the structural result may be:

```text
VALID_WITH_UNEVEN_STAGES
```

or:

```text
REQUIRES_STAGE_REBALANCING
```

instead of rejecting PP entirely.

---

# 18. Uneven Layer Counts Are Often Correct

Suppose the final stage owns a large LM head.

Instead of:

```text
12 / 12 / 12 / 12
```

we might choose:

```text
13 / 13 / 13 / 9
```

so the final stage has fewer Transformer layers to compensate for the LM head.

Likewise, MoE-heavy ranges may need fewer layers.

Natural-language interpretation:

> **Balance actual work and memory, not just layer count.**

At structural-analysis time, we only need to prove that a feasible contiguous partition exists.

The performance-optimal partition comes later.

---

# 19. Architecture-Level Stage Split Compatibility

Decoder-only Transformers are naturally sequential:

```text
Embedding
 -> Layer0
 -> Layer1
 -> ...
 -> LayerN
 -> LM Head
```

More complex architectures may have:

```text
cross-layer dependencies
encoder-decoder interactions
multi-branch paths
multimodal towers
shared modules
```

Example:

```text
Vision Encoder
      |
      v
Projector
      |
      v
Language Model
```

Natural-language interpretation:

> **Is the model really a clean assembly line, or does it contain side roads and shared dependencies that complicate stage boundaries?**

The Canonical Model must expose enough structure to answer this.

---

# 20. PP + TP Composition

Suppose:

```text
PP = 4
TP = 2
```

Then each pipeline stage is a TP group:

```text
Stage 0:
    GPU0 + GPU1

Stage 1:
    GPU2 + GPU3

Stage 2:
    GPU4 + GPU5

Stage 3:
    GPU6 + GPU7
```

Execution:

```text
Stage0 TP group
      |
      | hidden state
      v
Stage1 TP group
      |
      v
Stage2 TP group
      |
      v
Stage3 TP group
```

Inside a stage:

```text
TP collectives
```

Between stages:

```text
PP send/receive
```

Natural-language interpretation:

> **Each factory station may itself be staffed by several GPUs collaborating on the local work.**

The runtime must support PP and TP process-group composition.

---

# 21. PP + EP Composition

If a pipeline stage contains MoE layers, the stage may also use EP.

Conceptually:

```text
PP Stage
    |
    +--> dense attention
    +--> routed expert execution
```

So a candidate may require:

```text
PP + EP
```

or:

```text
PP + TP + EP
```

Structural Analysis derives the required capability:

```text
PP_EP_COMPOSITION
```

or:

```text
PP_TP_EP_COMPOSITION
```

The coarse runtime validator checks support, and the smoke test proves executability.

---

# 22. Concrete Example

Assume:

```text
Model:
    48 layers
    hidden_size = 8192
    64 query heads
    8 KV heads
    some MoE layers

Quantization:
    Q3

Runtime:
    vLLM

Hardware:
    H100
```

Memory Analysis gives:

```text
aggregate conservative memory:
    250 GB

usable HBM/GPU:
    70 GB

minimum GPU lower bound:
    4
```

Try:

```text
PP = 4
TP = 1
```

Initial partition:

```text
Stage 0:
    Embedding
    Layers 0-11

Stage 1:
    Layers 12-23

Stage 2:
    Layers 24-35

Stage 3:
    Layers 36-47
    Final Norm
    LM Head
```

---

# 23. Validate Layer Partition

```text
48 / 4 = 12
```

So:

```text
layer_partition:
    VALID
```

---

# 24. Validate Stage Boundaries

Boundaries are:

```text
11 -> 12
23 -> 24
35 -> 36
```

All are between complete Transformer blocks.

So:

```text
stage_boundaries:
    VALID
```

---

# 25. Validate Special Module Ownership

```text
Embedding:
    Stage 0

Final Norm:
    Stage 3

LM Head:
    Stage 3
```

Semantically valid.

---

# 26. Validate Per-Stage Memory

Suppose:

```text
Stage 0:
    58 GB

Stage 1:
    61 GB

Stage 2:
    63 GB

Stage 3:
    74 GB
```

with:

```text
usable HBM = 70 GB
```

Then the equal-layer split fails:

```text
equal_layer_partition:
    INVALID_MEMORY
```

But PP=4 itself may still be viable.

Rebalance:

```text
Stage 0:
    Layers 0-12

Stage 1:
    Layers 13-25

Stage 2:
    Layers 26-38

Stage 3:
    Layers 39-47 + LM Head
```

Suppose:

```text
Stage 0: 63 GB
Stage 1: 64 GB
Stage 2: 65 GB
Stage 3: 67 GB
```

Now:

```text
PP4:
    VALID_WITH_UNEVEN_STAGES
```

Important distinction:

```text
one stage assignment failed
```

does not imply:

```text
PP degree failed
```

The analyzer should determine whether **some feasible legal partition exists**.

---

# 27. MoE-Heavy Example

Suppose:

```text
Layers 0-23:
    mostly dense

Layers 24-47:
    many MoE layers
```

Equal 12-layer partitions might produce:

```text
Stage0: 45 GB
Stage1: 48 GB
Stage2: 72 GB
Stage3: 85 GB
```

A better legal partition may be:

```text
Stage0:
    16 dense layers

Stage1:
    14 mixed layers

Stage2:
    10 MoE-heavy layers

Stage3:
    8 MoE-heavy layers + LM Head
```

The structural question is therefore:

> **Can the sequential layer stack be partitioned into contiguous stages such that every stage fits and preserves semantics?**

This is more precise than merely checking:

```text
num_layers % PP == 0
```

---

# 28. Recommended Output Contract

```text
PPStructuralAnalysis {

    pp_degree:
        4

    sequential_partition {

        partitionable_blocks:
            48

        contiguous_stages:
            true

        legal_boundaries:
            true

        architecture_cross_stage_dependencies:
            NONE

        equal_layer_partition:
            INVALID_MEMORY

        feasible_partition_found:
            true

        stage_ranges:
            [
                0..12,
                13..25,
                26..38,
                39..47
            ]
    }

    boundary_ownership {

        embedding_stage:
            0

        final_norm_stage:
            3

        lm_head_stage:
            3

        split_inside_transformer_block:
            false

        special_modules:
            [...]
    }

    stage_memory {

        stage_0:
            ...

        stage_1:
            ...

        stage_2:
            ...

        stage_3:
            ...

        all_stages_fit:
            true

        kv_partition_by_layer:
            VALID
    }

    composition_requirements {

        pp_tp:
            NOT_REQUIRED

        pp_ep:
            REQUIRED

        pp_tp_ep:
            NOT_REQUIRED
    }

    coarse_runtime_validation {

        runtime:
            ...

        version:
            ...

        stage_slicing:
            SUPPORTED

        required_composition:
            SUPPORTED

        status:
            SUPPORTED
            | UNSUPPORTED
            | UNKNOWN
    }

    result {

        status:
            VALID
            | VALID_WITH_UNEVEN_STAGES
            | REQUIRES_STAGE_REBALANCING
            | UNSUPPORTED

        constraints:
            [...]

        rejection_reasons:
            [...]
    }
}
```

---

# 29. Analysis-Time vs Execution-Time Validation

As with TP and EP:

```text
PP Structural Analysis
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
        +--> stage processes initialize?
        +--> stage ownership correct?
        +--> send/receive path works?
        +--> model loads?
        +--> a few requests traverse all stages?
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

# 30. What This Phase Does Not Decide

Structural PP Analysis does not yet answer:

```text
Which PP degree has best TTFT?

How large is the pipeline bubble?

How should microbatches be scheduled?

How expensive is stage-to-stage communication?

Does PP4 outperform TP4?

Which partition minimizes cost/token?
```

Those belong to later communication/performance/search analysis.

---

# 31. Interview Summary

> **Pipeline Parallelism divides the model along the layer/depth axis. Structural PP analysis checks whether the model can be cut at legal contiguous stage boundaries, whether embeddings and output modules have valid owners, whether every stage's weights, KV cache, and runtime memory fit locally, whether MoE-heavy or LM-head-heavy stages require uneven layer counts, and whether the runtime supports the resulting stage composition with TP/EP if needed. The goal is not to optimize the pipeline yet, but to prove that a legal stage decomposition exists.**

---

# 32. Locked Mental Model

Think of PP as an assembly line:

```text
Stage 0:
    early processing

Stage 1:
    middle processing

Stage 2:
    later processing

Stage 3:
    finish + logits
```

Then the structural questions become:

```text
1. Can I cut the assembly line at clean boundaries?
2. Can each station fit its machinery and state?
3. Can the intermediate product be handed to the next station?
4. Can heavy stations be made smaller if needed?
5. Does the runtime support those stations as separate stages?
```
