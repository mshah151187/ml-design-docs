# Memory Analysis Phase

> **Purpose:** Convert a runtime-compatible `QuantizationCandidateFamily` plus model, serving, traffic, runtime, and hardware information into a topology-independent memory profile. The output establishes static-model memory bounds, KV and activation pressure, runtime-memory needs, and a lower bound on the number of GPUs required before execution parallelism is selected.

## 1. Position in the Search Foundation Flow

```text
Quantization Analysis
        |
        | QuantizationCandidateFamily
        | representation metadata
        | bounded choices such as group_size=[32,64,128]
        | runtime + hardware annotations
        v
Memory Analysis
        |
        | global/logical memory truth
        | memory-pressure curves
        | minimum aggregate GPU-count lower bound
        v
Execution Parallelism Analysis
        |
        | TP / PP / EP / CP
        v
Per-Rank Memory Validation
        |
        v
Feasible Execution Space
```

Memory Analysis is deliberately **topology-independent**.

It answers:

```text
How much memory does this representation require?
How much memory does KV consume as context/concurrency grows?
How much transient activation/workspace memory should we expect?
How much runtime-owned memory must be reserved?
How many GPUs are required at minimum from an aggregate-HBM perspective?
```

It does **not** yet decide:

```text
TP = ?
PP = ?
EP = ?
CP = ?
```

## 2. Inputs

Memory Analysis consumes four major sources of information.

### 2.1 Quantization Candidate Family

Memory Analysis receives a **runtime-compatible representation family**, not merely a dtype such as `INT4`.

Example:

```text
Q3:

Attention Linear:
    weight_precision     = INT4
    activation_precision = BF16

    weight_representation {
        granularity       = PER_GROUP
        axis              = K
        group_size_candidates = [32, 64, 128]

        scale_dtype       = FP16
        scale_granularity = PER_GROUP

        symmetric         = true
        zero_point        = none
    }

MoE Expert Linear:
    weight_precision     = FP8
    activation_precision = FP8

    weight_scale {
        dtype       = FP32
        granularity = PER_TENSOR
    }

    activation_scaling {
        mode        = DYNAMIC
        granularity = PER_TOKEN
    }

KV Cache:
    precision = FP8
    scale_metadata = ...

Norm / Softmax:
    precision = BF16

Runtime:
    vLLM

Hardware Candidates:
    H100
    H200
```

The handoff also separates representation overhead into three classes:

```text
A. Deterministic metadata
   scales
   zero points
   known auxiliary tensors

B. Search-dependent metadata
   group-size candidates
   block-size candidates
   scale-granularity candidates

C. Physical-layout uncertainty
   packing inefficiency
   alignment
   padding
   runtime-native layout differences
```

Memory Analysis should calculate A explicitly, evaluate B as a range, and approximate C with a small bounded reserve.

This keeps memory estimation representation-aware without requiring exact per-kernel layout inspection.
### 2.2 Canonical Model

Provides:

```text
num_layers
hidden_size
intermediate_size
num_attention_heads
num_kv_heads
head_dim
attention parameter count
expert parameter count
embedding parameter count
lm_head parameter count
number_of_experts
experts_per_token
```

Quantization says how many bytes each tensor uses; the canonical model says how many values exist.

### 2.3 Serving Requirements

Examples:

```text
max_context_length = 128K
required max concurrent sequences, if explicit
required max batch/token capacity, if explicit
```

The maximum context is particularly important because KV grows with active tokens.

### 2.4 Expected Traffic Distribution

Example:

```text
60% short
30% medium
10% long
streaming = true
moderate prefix reuse
```

Useful distributions include:

```text
input length
output length
active sequence length
concurrent sequence count
scheduled tokens / iteration
```

Traffic lets us distinguish:

```text
hard maximum memory requirement
```

from:

```text
expected production memory pressure
```

## 3. Top-Level Memory Model

```text
M_required =

    M_model_representation

  + M_KV_payload

  + M_logical_activations

  + M_runtime_fixed

  + M_runtime_workload_dependent

  + M_operator_workspace

  + M_KV_metadata

  + M_topology_dependent_runtime   # unresolved until topology exists

  + M_safety_headroom
```

The first phase resolves topology-independent quantities and preserves topology-dependent terms as functions/bounds.

## 4. Step 1 — Static Model Memory

Static Model Analysis should normally operate at **representation-aware accuracy**, not exact kernel-layout accuracy.

The useful decomposition is:

```text
M_static_model =
    M_quantized_payload
  + M_deterministic_quantization_metadata
  + M_known_auxiliary_tensors
  + M_physical_layout_reserve
```

For a mixed-precision model:

```text
M_quantized_payload =
    M_attention_weights
  + M_expert_weights
  + M_embedding_weights
  + M_lm_head
  + M_norm_and_other
```

### 4.1 Quantized Payload

Examples:

```text
INT4:
    ~0.5 byte/value

FP8 / INT8:
    ~1 byte/value

BF16 / FP16:
    ~2 bytes/value
```

This is the logical payload before quantization metadata.

### 4.2 Deterministic Quantization Metadata

If the representation contract gives:

```text
group_size
scale_dtype
scale_granularity
zero-point semantics
known auxiliary tensors
```

Memory Analysis should calculate these explicitly.

For group-wise INT4 with FP16 scales:

```text
num_groups(G)
    ≈
num_quantized_weights / G

scale_bytes(G)
    =
num_groups(G) × 2
```

If asymmetric quantization uses one zero point per group:

```text
zero_point_bytes(G)
    =
num_groups(G)
    × bytes_per_zero_point
```

Known auxiliary tensors are added directly.

### 4.3 Search-Dependent Representation Metadata

If Quantization Analysis has not selected one group size yet:

```text
group_size_candidates = [32, 64, 128]
```

Memory Analysis should preserve the choice and compute:

```text
M_static_model(G=32)
M_static_model(G=64)
M_static_model(G=128)
```

This produces a memory range rather than forcing a quantization decision.

### 4.4 Physical-Layout Reserve

At Search Foundation time we deliberately avoid inspecting every exact kernel layout.

Implementation-specific details such as:

```text
packing inefficiency
alignment
tile padding
runtime-native reorder/layout
```

are represented by a modest bounded reserve:

```text
M_physical_layout_reserve
    =
representation_aware_bytes
× layout_reserve_percent
```

The reserve should be a policy/range derived from platform experience rather than treated as quantization metadata itself.

### 4.5 Three Accuracy Levels

```text
LEVEL 1 — Logical Payload

parameter_count × representation bytes


LEVEL 2 — Representation-Aware Estimate

logical payload
+ scales
+ zero points
+ known auxiliary tensors


LEVEL 3 — Exact Runtime-Resident Layout

Level 2
+ exact kernel packing
+ exact padding/alignment
+ runtime-native layout/repacking
```

Search Foundation normally uses:

```text
LEVEL 2
+
small physical-layout reserve
```

Exact Level 3 accounting belongs to later runtime/topology validation.

The result of Static Model Analysis should therefore include both:

```text
optimistic_lower_bound
```

and:

```text
conservative_estimate
```

rather than pretending there is one exact pre-topology number.
## 5. Step 2 — KV Memory

For an autoregressive decoder:

```text
KV_bytes_per_token
    =
2
× num_layers
× num_kv_heads
× head_dim
× bytes_per_KV_element
```

The factor `2` is for K and V.

Then:

```text
KV_bytes(sequence)
    =
KV_bytes_per_token
× active_sequence_length
```

This gives:

```text
KV for short sequence
KV for medium sequence
KV for long sequence
KV for max-context sequence
```

## 6. Step 3 — KV vs Concurrency

General form:

```text
KV_total
    =
Σ KV(active_sequence_i)
```

Approximation:

```text
KV_total
    ≈
concurrency
× KV_bytes_per_token
× average_active_sequence_length
```

Memory Analysis should produce a capacity curve:

```text
Concurrency    Expected KV Memory
-----------    ------------------
1              ...
8              ...
16             ...
32             ...
64             ...
128            ...
```

and a separate worst-case view such as:

```text
N concurrent 128K sequences
```

This exposes the relationship between:

```text
context
concurrency
KV precision
HBM pressure
```

## 7. Step 4 — Activation Memory

Activations are transient.

They depend on:

```text
batch size
scheduled token count
prefill vs decode
hidden size
intermediate size
operator precision
MoE routing pattern
```

A useful model is:

```text
M_activation =
f(
    num_scheduled_tokens,
    active_sequences,
    prefill_or_decode,
    model_shapes,
    activation_precision
)
```

For Q3:

```text
Attention activations = BF16
MoE activations       = FP8
```

At this stage we derive:

```text
logical activation-size model
+
known topology-independent workspace estimates
```

not final per-rank peak memory.

## 8. Step 5 — Runtime Memory

Serving runtimes consume memory beyond weights, KV payload, and logical activations.

Important categories:

```text
persistent runtime state
allocator pools / fragmentation reserve
CUDA graph memory
operator workspaces
KV allocator metadata
scheduler / token buffers
logits / sampling buffers
runtime-specific engine state
```

Some terms are fixed, some workload-dependent, some topology-dependent.

A useful model is:

```text
RuntimeMemoryModel {
    fixed_overhead

    scheduler_overhead(
        max_num_seqs,
        max_num_batched_tokens
    )

    graph_overhead(
        graph_config
    )

    operator_workspace(
        workload_shape
    )

    kv_metadata(
        kv_blocks,
        num_sequences
    )

    communication_workspace(
        topology
    )
}
```

The final communication term remains unresolved until parallelism is selected.

See `runtime_memory_overheads.md` for the detailed breakdown.

## 9. Step 6 — Hardware HBM Model

For each hardware candidate:

```text
total_hbm
usable_hbm
```

Conceptually:

```text
usable_hbm
    =
total_hbm
-
reserved_system/runtime_allowance
-
safety_margin
```

Nominal HBM should not be treated as fully usable serving memory.

## 10. Step 7 — Minimum Aggregate GPU Count

Once topology-independent memory is estimated:

```text
minimum_gpu_count_lower_bound
    =
ceil(
    aggregate_required_memory
    /
    usable_hbm_per_gpu
)
```

This means:

> Any viable execution topology must use at least this many GPUs worth of usable HBM.

It does **not** mean:

```text
TP = minimum_gpu_count
```

Example:

```text
aggregate_required_memory = 280 GB
usable_hbm_per_gpu = 70 GB

minimum_gpu_count_lower_bound
    = ceil(280 / 70)
    = 4
```

Conclusion:

```text
at least 4 GPUs are required
```

## 11. Minimum GPU Count Is Not Parallelism

If Memory Analysis says:

```text
minimum GPU count >= 4
```

Execution Parallelism may still consider:

```text
TP4
TP2 x PP2
TP2 x EP2
TP2 x PP1 x EP2
```

All may use four GPUs while distributing memory/computation differently.

So:

```text
Memory Analysis
    -> aggregate lower bound
```

while:

```text
Parallelism Analysis
    -> distribution strategy
```

## 12. Why Per-Rank Validation Is Still Required

Aggregate fit does not guarantee topology fit.

Example:

```text
4 GPUs × 70 GB usable
    = 280 GB aggregate

logical memory requirement
    = 270 GB
```

Under a concrete topology, some tensors may replicate, KV may shard differently, communication buffers appear, and workspaces differ by rank.

A rank could reach:

```text
76 GB
```

and fail.

Therefore:

```text
Global Memory Analysis
    ->
Parallelism Selection
    ->
Per-Rank Memory Validation
```

is required.

## 13. MoE-Specific Memory Considerations

For a resident MoE model:

```text
all expert weights occupy memory
```

even though only selected experts execute per token.

So:

```text
number_of_experts
    -> resident weight memory

experts_per_token
    -> compute / routing / communication behavior
```

Memory Analysis should expose:

```text
resident_expert_weight_bytes
```

so later execution analysis can evaluate expert parallelism.

## 14. Traffic-Aware Memory Views

Memory Analysis should produce two views.

### 14.1 Hard Requirement View

Based on:

```text
maximum context
explicit concurrency requirements
worst allowed batch/token settings
```

This answers whether the deployment can satisfy required capacity.

### 14.2 Expected Production View

Based on:

```text
traffic distribution
expected context mix
expected active-sequence distribution
```

This describes typical production memory pressure.

## 15. Worked Example — 100B MoE Configuration Q3

Assume:

```text
Model:
    100B total parameters
    decoder-only MoE Transformer
    GQA
    128K max context

Quantization Candidate Family Q3:
    Attention weights     = INT4
    Attention activations = BF16

    INT4 representation:
        group_size_candidates = [32, 64, 128]
        scale_dtype           = FP16
        symmetric             = true
        zero_point            = none

    MoE expert weights     = FP8
    MoE expert activations = FP8
    KV                     = FP8
    Norm / Softmax         = BF16

Runtime:
    vLLM

Hardware Candidate:
    H100
```

For illustration, assume:

```text
attention/projection weights:
    25B params

expert weights:
    70B params

other weights:
    5B params
```

These numbers are illustrative.

## 16. Example — Static Model Memory Range

Approximate logical payload:

```text
Attention:
    25B × 0.5 byte
    = 12.5 GB

Experts:
    70B × 1 byte
    = 70 GB

Other:
    5B × 2 bytes
    = 10 GB
```

So before metadata:

```text
logical model payload
    ≈ 92.5 GB
```

Now calculate the INT4 attention scale metadata for each candidate group size.

For FP16 scales:

```text
scale_bytes(G)
    ≈
25B / G × 2 bytes
```

Illustratively:

```text
G = 128:
    scale metadata ≈ 0.39 GB

G = 64:
    scale metadata ≈ 0.78 GB

G = 32:
    scale metadata ≈ 1.56 GB
```

Therefore the attention representation alone becomes approximately:

```text
G = 128:
    12.50 + 0.39
    ≈ 12.89 GB

G = 64:
    12.50 + 0.78
    ≈ 13.28 GB

G = 32:
    12.50 + 1.56
    ≈ 14.06 GB
```

Because this example is symmetric:

```text
zero-point bytes = 0
```

Then add:

```text
FP8 expert scaling metadata
KV-related persistent scale metadata, if any
known auxiliary quantization tensors
```

Finally apply the bounded physical-layout reserve for:

```text
packing inefficiency
alignment
padding
runtime-native layout uncertainty
```

The result should be represented as a range:

```text
static_model_memory {

    by_group_size {
        128 -> ...
        64  -> ...
        32  -> ...
    }

    optimistic_lower_bound:
        ...

    conservative_estimate:
        ...

    physical_layout_reserve:
        ...
}
```

This is more useful than collapsing the candidate family into one invented number.
## 17. Example — KV Memory

Assume:

```text
num_layers    = 80
num_kv_heads  = 8
head_dim      = 128
KV precision  = FP8
```

Then:

```text
KV_bytes_per_token
    =
2 × 80 × 8 × 128 × 1
    =
163,840 bytes/token
```

Approximately:

```text
160 KiB/token
```

At 128K tokens:

```text
~20 GiB KV / sequence
```

before additional scale/allocator metadata.

Illustrative context sizes:

```text
8K:
    ~1.25 GiB

32K:
    ~5 GiB

64K:
    ~10 GiB

128K:
    ~20 GiB
```

## 18. Example — Traffic-Aware KV

Suppose:

```text
60% short  -> 8K active length
30% medium -> 32K active length
10% long   -> 96K active length
```

Memory Analysis derives:

```text
expected KV / active sequence
```

and then a concurrency curve for:

```text
8 sequences
16 sequences
32 sequences
...
```

The purpose is to expose HBM pressure, not to choose final concurrency yet.

## 19. Example — Activation / Workspace Model

For Q3:

```text
attention activation precision = BF16
expert activation precision    = FP8
```

Memory Analysis records:

```text
M_activation(
    scheduled_tokens,
    active_sequences,
    phase
)
```

with separate prefill/decode profiles, plus known runtime workspace estimates.

## 20. Example — Runtime Memory

The runtime model may include:

```text
fixed runtime state
CUDA graph reserve
allocator pool reserve
scheduler/token buffers
KV block metadata
operator workspace
```

and leaves:

```text
communication workspace(topology)
```

unresolved until parallelism is known.

## 21. Example — Minimum Aggregate GPU Count

Suppose the memory model produces two aggregate views for Q3:

```text
OPTIMISTIC / LOWER-BOUND VIEW

static model:
    96 GB

target KV:
    100 GB

activation/workspace:
    18 GB

runtime-known overhead:
    12 GB

minimum safety allowance:
    10 GB

aggregate lower-bound estimate:
    236 GB
```

and:

```text
CONSERVATIVE VIEW

static model:
    103 GB

target KV:
    100 GB

activation/workspace:
    20 GB

runtime-known overhead:
    14 GB

safety headroom:
    13 GB

aggregate conservative estimate:
    250 GB
```

Assume:

```text
usable HBM per GPU:
    70 GB
```

Then:

```text
minimum_gpu_count_lower_bound
    =
ceil(236 / 70)
    =
4
```

and:

```text
conservative_gpu_count_estimate
    =
ceil(250 / 70)
    =
4
```

If the two estimates crossed a GPU boundary, both should be preserved.

For example:

```text
minimum_gpu_count_lower_bound:
    4

conservative_gpu_count_estimate:
    5
```

would mean:

> Fewer than four GPUs is impossible from aggregate memory alone; five is the safer pre-topology planning estimate.

Neither number determines TP/PP/EP/CP.
## 22. Final Output — MemoryProfile

For each:

```text
QuantizationCandidateFamily
+
Runtime
+
Hardware
```

emit something like:

```text
MemoryProfile {

    candidate_family_id:
        Q3

    runtime:
        vLLM

    hardware:
        H100


    static_model_memory {

        logical_payload_bytes:
            ...

        deterministic_metadata {
            scale_bytes:
                ...

            zero_point_bytes:
                ...

            auxiliary_tensor_bytes:
                ...
        }

        candidate_variants {

            group_size_128 {
                representation_aware_bytes:
                    ...

                estimated_resident_bytes_with_layout_reserve:
                    ...
            }

            group_size_64 {
                representation_aware_bytes:
                    ...

                estimated_resident_bytes_with_layout_reserve:
                    ...
            }

            group_size_32 {
                representation_aware_bytes:
                    ...

                estimated_resident_bytes_with_layout_reserve:
                    ...
            }
        }

        physical_layout_reserve {
            policy:
                ...

            percent_or_range:
                ...
        }

        optimistic_lower_bound:
            ...

        conservative_estimate:
            ...
    }


    kv_memory {

        logical_bytes_per_token:
            ...

        quantization_metadata_model:
            ...

        bytes_per_short_sequence:
            ...

        bytes_per_medium_sequence:
            ...

        bytes_per_long_sequence:
            ...

        bytes_per_max_context_sequence:
            ...

        concurrency_curve: [
            {sequences: 8,  bytes: ...},
            {sequences: 16, bytes: ...},
            {sequences: 32, bytes: ...}
        ]
    }


    activation_memory {

        prefill_memory_model:
            f(tokens, batch)

        decode_memory_model:
            f(active_sequences)

        logical_activation_precision_map:
            ...
    }


    runtime_memory {

        fixed_overhead:
            ...

        workload_dependent_model:
            f(
                max_num_seqs,
                max_num_batched_tokens,
                graph_config
            )

        topology_dependent_model:
            unresolved
    }


    hardware_memory {

        total_hbm:
            ...

        usable_hbm:
            ...
    }


    aggregate_memory {

        optimistic_lower_bound:
            ...

        conservative_estimate:
            ...
    }


    constraints {

        minimum_gpu_count_lower_bound:
            ...

        conservative_gpu_count_estimate:
            ...

        model_weight_pressure:
            HIGH

        kv_pressure:
            HIGH

        long_context_pressure:
            VERY_HIGH

        expert_weight_pressure:
            HIGH
    }
}
```

The important outputs are now:

```text
static memory by representation candidate
static-model lower bound
static-model conservative estimate
aggregate-memory lower bound
aggregate-memory conservative estimate
minimum GPU-count lower bound
conservative pre-topology GPU-count estimate
```

Memory Analysis still does not select the final group size; it preserves the memory consequences of candidate representation choices for later search/experimentation.
## 23. Handoff to Execution Parallelism Analysis

Execution Parallelism receives:

```text
QuantizationCandidateFamily
+
MemoryProfile
+
CanonicalModel Structure
+
Runtime
+
Hardware
```

and asks:

```text
Which TP degrees are structurally valid?
Is PP useful or necessary?
Should MoE experts use EP?
Does 128K context justify CP?
Which topologies satisfy the memory lower bound?
Which topology gives good compute/communication balance?
```

Then the chosen topology goes through per-rank memory validation.

## 24. Locked Principle

> **Memory Analysis computes the global/logical memory truth and a minimum aggregate GPU-count lower bound. Execution Parallelism decides how that memory and computation are distributed. Per-rank validation then proves the selected topology actually fits.**
