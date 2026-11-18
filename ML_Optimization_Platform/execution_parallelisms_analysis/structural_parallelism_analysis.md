# Structural Parallelism Analysis

> **Purpose:** Convert model structure, quantization constraints, memory pressure, serving requirements, traffic characteristics, and coarse runtime capabilities into a **structurally valid parallelism search space** across Tensor Parallelism (TP), Expert Parallelism (EP), Pipeline Parallelism (PP), and Context Parallelism (CP).

The goal is **not** to choose the final topology. The goal is to determine which TP/EP/PP/CP degrees are structurally plausible, what constraints are attached to them, which combinations are meaningful, and which candidates are impossible before spending GPU time.

---

# 1. Position in the Search Foundation

```text
CanonicalModel
ServingRequirements
ExpectedTrafficDistribution
Available Hardware
Available Runtimes
        |
        v
Quantization Analysis
        |
        v
Memory Analysis
        |
        v
Structural Parallelism Analysis
        |
        v
Structural Parallelism Analysis Artifact
        |
        v
Candidate Topology Generation
        |
        v
Per-Rank Memory Validation
        |
        v
Communication Analysis
        |
        v
Analyzed Serving Candidates
```

Ownership boundary:

```text
Search Foundation:
    predict structural feasibility

Compatibility Smoke Test:
    prove concrete executability

Benchmarking:
    measure performance

Qualification:
    prove production suitability
```

---

# 2. What Is Being Partitioned?

```text
TP:
    tensors/operators inside a layer

EP:
    experts inside an MoE layer

PP:
    the layer stack

CP:
    the token/context dimension
```

Typical pressures:

```text
TP:
    large operators / model width

EP:
    expert residency and sparse expert execution

PP:
    model depth and aggregate model memory

CP:
    long-context attention and KV pressure
```

The analyzer therefore should not blindly enumerate the full Cartesian product. It first decides which parallelism dimensions are relevant, derives legal degree domains for each one, then derives constraints for composing them.

---

# 3. Inputs

```text
StructuralParallelismAnalysisInput {

    canonical_model_ref

    serving_requirements_ref

    traffic_distribution_ref

    quantization_analysis_ref

    memory_analysis_ref

    runtime_environment

    hardware_environment
}
```

## Canonical Model

Provides:

```text
num_layers
hidden_size
intermediate_size
query_heads
kv_heads
head_dim

MoE:
    routed_experts
    shared_experts
    experts_per_token
    MoE layer locations

attention type:
    MHA / GQA / MQA

position encoding

special modules:
    embeddings
    LM head
    multimodal towers
    cross-attention
```

## Quantization Analysis

Provides:

```text
weight dtype
activation dtype
KV dtype

group sizes
scale granularity
zero-point requirements
packing/layout constraints
runtime compatibility
hardware compatibility
```

## Memory Analysis

Provides:

```text
static model memory
expert memory
KV memory
activation memory
runtime reserve
usable HBM/rank
minimum aggregate GPU lower bound
```

## Serving Requirements

Provides:

```text
max context
latency constraints
throughput constraints
quality constraints
hardware restrictions
runtime restrictions
```

## Expected Traffic Distribution

Provides:

```text
short-context fraction
medium-context fraction
long-context fraction
concurrency distribution
prefix reuse
streaming behavior
```

## Runtime and Hardware Context

Provides coarse capabilities such as:

```text
runtime:
    vLLM / SGLang / other

runtime version:
    pinned version

hardware:
    H100 / H200 / other

known capabilities:
    TP
    PP
    EP
    CP
    KV-head replication
    topology composition
```

This layer is deliberately coarse. Structural Analysis should not reproduce exact runtime kernel-selection logic.

---

# 4. High-Level Decision Process

```text
STEP 1
Read model structure and memory pressure
        |
        v
STEP 2
Determine parallelism relevance
        |
        v
STEP 3
Analyze TP structural legality
        |
        v
STEP 4
Analyze EP structural legality
        |
        v
STEP 5
Analyze PP structural legality
        |
        v
STEP 6
Analyze CP structural legality
        |
        v
STEP 7
Derive cross-parallelism composition requirements
        |
        v
STEP 8
Perform coarse runtime capability validation
        |
        v
STEP 9
Emit valid degree domains + constraints + rejections
```

---

# 5. Running Example

```text
Model:
    48-layer decoder-only Transformer

hidden_size:
    8192

intermediate_size:
    28672

query_heads:
    64

kv_heads:
    8

head_dim:
    128

MoE:
    enabled
    routed_experts = 64
    shared_experts = 1
    experts_per_token = 2

context:
    max 128K

quantization:
    W4A16
    group_size = 128
    KV = FP8

hardware:
    H100

runtime:
    pinned production runtime

traffic:
    60% short
    30% medium
    10% long
```

Memory Analysis says:

```text
aggregate model + runtime memory:
    requires multiple GPUs

expert memory pressure:
    HIGH

long-context KV pressure:
    HIGH
```

So all four dimensions are worth analyzing.

---

# 6. Step 1 — Determine Parallelism Relevance

The analyzer first asks:

```text
Is TP relevant?
Is EP relevant?
Is PP relevant?
Is CP relevant?
```

For the running example:

```text
TP:
    relevant because large operators may need sharding

EP:
    relevant because the model has 64 routed experts

PP:
    relevant because aggregate model memory is large

CP:
    relevant because max context is 128K and long-context traffic exists
```

Output:

```text
ParallelismRelevance {

    TP:
        REQUIRED_TO_ANALYZE

    EP:
        REQUIRED_TO_ANALYZE

    PP:
        REQUIRED_TO_ANALYZE

    CP:
        REQUIRED_TO_ANALYZE
}
```

A dense small-context model might mark EP and CP as `NOT_APPLICABLE`.

---

# 7. Step 2 — Tensor Parallelism Analysis

TP asks:

> **Can the operators inside one layer be split across `TP` ranks while preserving attention structure, local tensor shapes, quantization representation, and runtime legality?**

Initial candidate degrees:

```text
TP candidates:
    [1, 2, 4, 8, 16, 32]
```

## 7.1 Attention Partitionability

For:

```text
query_heads = 64
kv_heads    = 8
```

### TP = 4

```text
Q heads/rank:
    64 / 4 = 16

KV heads/rank:
    8 / 4 = 2
```

Result:

```text
TP4:
    attention = VALID
```

### TP = 16

```text
Q heads/rank:
    64 / 16 = 4

KV heads:
    8 < 16
```

Direct KV sharding is not possible, so the analyzer derives:

```text
Required Capability:
    KV_HEAD_REPLICATION
```

Result:

```text
TP16:
    VALID_WITH_CONSTRAINT
```

## 7.2 Dense Tensor Partitionability

For TP4:

```text
hidden local:
    8192 / 4 = 2048

intermediate local:
    28672 / 4 = 7168
```

Both are structurally clean.

## 7.3 Quantization Partitionability

For:

```text
W4A16
group_size = 128
```

if the local sharded dimension is:

```text
local_K = 2048
```

then:

```text
2048 % 128 == 0
```

Quantization groups remain intact.

The analyzer checks:

```text
group boundaries
scale metadata ownership
zero-point ownership
packed representation plausibility
```

It does not reproduce runtime kernel internals.

## 7.4 TP Output

```text
tensor_parallelism {

    candidate_degrees:
        [1, 2, 4, 8, 16]

    TP1:
        VALID

    TP2:
        VALID

    TP4:
        VALID

    TP8:
        VALID

    TP16:
        VALID_WITH_CONSTRAINT

        requires:
            KV_HEAD_REPLICATION

    rejected:
        TP32:
            LOCAL_TENSOR_OR_QUANTIZATION_CONSTRAINT
}
```

---

# 8. Step 3 — Expert Parallelism Analysis

EP asks:

> **Can the routed expert set be distributed across ranks while preserving routing semantics and allowing each expert to execute on its owner?**

Initial candidates:

```text
EP:
    [1, 2, 4, 8, 16]
```

## 8.1 Expert Set Partitionability

For:

```text
routed_experts = 64
```

we get:

```text
EP2:
    32 experts/rank

EP4:
    16 experts/rank

EP8:
    8 experts/rank

EP16:
    4 experts/rank
```

All divide evenly.

If instead:

```text
routed_experts = 10
EP = 4
```

the analyzer derives:

```text
Required Capability:
    UNEVEN_EXPERT_PLACEMENT
```

rather than immediately rejecting EP4.

## 8.2 Shared Experts

For:

```text
shared_experts = 1
```

the shared expert is analyzed separately from routed experts.

Possible required handling:

```text
REPLICATION_OR_SEPARATE_PARALLELIZATION
```

## 8.3 Router Placement

Default coarse assumption:

```text
router:
    REPLICATED
```

because router weights are typically tiny relative to expert FFNs.

## 8.4 Expert Local Fit

If one full expert does not fit on a rank:

```text
expert_size:
    20 GB

available expert budget/rank:
    8 GB
```

then pure EP is not enough.

The analyzer derives:

```text
REQUIRES_EXPERT_TP
```

and may derive:

```text
minimum expert-local TP >= 2
```

## 8.5 EP Output

```text
expert_parallelism {

    applicable:
        true

    candidate_degrees:
        [1, 2, 4, 8]

    EP1:
        VALID

    EP2:
        VALID

    EP4:
        VALID

    EP8:
        VALID

    shared_expert_policy:
        REPLICATION_OR_SEPARATE_PARALLELIZATION

    router_placement:
        REPLICATED
}
```

---

# 9. Step 4 — Pipeline Parallelism Analysis

PP asks:

> **Can the model's layer sequence be divided into contiguous stages such that every stage preserves semantics and fits its local memory budget?**

Initial candidates:

```text
PP:
    [1, 2, 4, 8]
```

## 9.1 Sequential Partitionability

For:

```text
48 layers
PP4
```

a simple split is:

```text
12 / 12 / 12 / 12
```

So there are enough partitionable blocks.

## 9.2 Boundary Ownership

Natural ownership:

```text
Stage 0:
    Embedding
    early layers

Middle stages:
    Transformer layers

Final stage:
    late layers
    Final Norm
    LM Head
```

Boundaries should normally be between complete Transformer blocks.

## 9.3 Per-Stage Memory

Suppose equal partitioning gives:

```text
Stage 0:
    58 GB

Stage 1:
    61 GB

Stage 2:
    63 GB

Stage 3:
    74 GB

usable HBM:
    70 GB
```

Equal partition fails.

But another legal split may work:

```text
Stage 0:
    layers 0-12

Stage 1:
    layers 13-25

Stage 2:
    layers 26-38

Stage 3:
    layers 39-47 + LM head
```

If that fits:

```text
PP4:
    VALID_WITH_UNEVEN_STAGES
```

Important:

```text
one stage assignment failing
!=
PP degree failing
```

The analyzer looks for **some feasible legal contiguous partition**.

## 9.4 PP Output

```text
pipeline_parallelism {

    candidate_degrees:
        [1, 2, 4]

    PP1:
        VALID

    PP2:
        VALID

    PP4:
        VALID_WITH_UNEVEN_STAGES

        feasible_partition:
            [
                0..12,
                13..25,
                26..38,
                39..47
            ]

    PP8:
        rejected_or_not_generated
}
```

---

# 10. Step 5 — Context Parallelism Analysis

CP asks:

> **Can the sequence/context itself be partitioned across ranks while preserving exact attention, position, mask, and KV semantics?**

Initial candidates:

```text
CP:
    [1, 2, 4, 8]
```

The running example has:

```text
max_context = 128K
```

and non-trivial long-context traffic.

## 10.1 Context Partitionability

For:

```text
CP4
128K context
```

we get:

```text
32K tokens/rank
```

Clean.

## 10.2 Attention Semantics

A query on one context rank may need K/V from earlier ranks.

Therefore the analyzer derives:

```text
DISTRIBUTED_ATTENTION
GLOBAL_SOFTMAX_SEMANTICS
GLOBAL_CAUSAL_MASK
```

## 10.3 Position Semantics

With RoPE:

```text
Rank 0 local token 0 -> global position 0
Rank 1 local token 0 -> global position 32K
Rank 2 local token 0 -> global position 64K
Rank 3 local token 0 -> global position 96K
```

Global position IDs must remain correct.

## 10.4 Distributed KV Ownership

With FP8 KV:

```text
Rank 0:
    KV positions 0-31K

Rank 1:
    KV positions 32K-63K

Rank 2:
    KV positions 64K-95K

Rank 3:
    KV positions 96K-127K
```

The analyzer checks whether:

```text
KV page ownership
KV quantization metadata
decode-time distributed KV access
```

are structurally supportable.

## 10.5 Workload Relevance

Traffic:

```text
60% short
30% medium
10% long
```

So CP may be marked:

```text
LONG_CONTEXT_FOCUSED
```

rather than assumed for every request.

## 10.6 CP Output

```text
context_parallelism {

    applicable:
        true

    applicability:
        LONG_CONTEXT_FOCUSED

    CP1:
        VALID

    CP2:
        VALID_IF_RUNTIME_SUPPORTED

    CP4:
        VALID_IF_RUNTIME_SUPPORTED

        requires:
            DISTRIBUTED_ATTENTION
            GLOBAL_CAUSAL_MASK
            DISTRIBUTED_KV
            DECODE_WITH_DISTRIBUTED_KV
}
```

---

# 11. Step 6 — Cross-Parallelism Composition

After analyzing each dimension independently, the analyzer asks:

```text
Can these dimensions compose?
```

Examples:

```text
TP + PP
TP + EP
PP + EP
TP + CP
PP + CP
EP + CP
```

Support for each individual dimension does not imply support for every combination.

## TP + PP

```text
PP = 4
TP = 2
```

Each pipeline stage becomes a TP2 group.

Required capability:

```text
TP_PP_COMPOSITION
```

## TP + EP

EP selects the expert-owning group; TP can split expert-local execution inside that group.

Required capability:

```text
TP_EP_COMPOSITION
```

## PP + EP

Pipeline stages containing MoE layers may need EP locally.

Required capability:

```text
PP_EP_COMPOSITION
```

## TP + CP

TP partitions model dimensions; CP partitions sequence positions.

Required capability:

```text
TP_CP_COMPOSITION
```

## PP + CP

Each pipeline stage may itself run context-sharded execution.

Required capability:

```text
PP_CP_COMPOSITION
```

## EP + CP

CP partitions tokens by context position; EP routes those tokens to distributed experts.

Required capability:

```text
EP_CP_COMPOSITION
```

Composition output:

```text
composition_rules {

    TP_PP:
        STRUCTURALLY_ALLOWED
        requires TP_PP_COMPOSITION

    TP_EP:
        STRUCTURALLY_ALLOWED
        requires TP_EP_COMPOSITION

    PP_EP:
        STRUCTURALLY_ALLOWED
        requires PP_EP_COMPOSITION

    TP_CP:
        STRUCTURALLY_ALLOWED
        requires TP_CP_COMPOSITION

    PP_CP:
        STRUCTURALLY_ALLOWED
        requires PP_CP_COMPOSITION

    EP_CP:
        STRUCTURALLY_ALLOWED
        requires EP_CP_COMPOSITION
}
```

---

# 12. Step 7 — Coarse Runtime Capability Validation

Structural Analysis compares required capabilities with coarse runtime metadata.

Example:

```text
RuntimeCapabilities {

    tensor_parallelism:
        SUPPORTED

    pipeline_parallelism:
        SUPPORTED

    expert_parallelism:
        SUPPORTED

    context_parallelism:
        UNKNOWN_OR_VERSION_DEPENDENT

    kv_head_replication:
        SUPPORTED

    TP_PP_COMPOSITION:
        SUPPORTED

    TP_EP_COMPOSITION:
        SUPPORTED_OR_UNKNOWN

    PP_EP_COMPOSITION:
        SUPPORTED_OR_UNKNOWN

    TP_CP_COMPOSITION:
        UNKNOWN
}
```

This is **not** exact runtime proof.

It exists to prune obviously unsupported topology families before GPU experiments.

---

# 13. What Structural Analysis Should Not Do

Do not maintain a giant handwritten mapping like:

```text
runtime version X
    kernel A
        TP2
        TP4
        TP8

kernel B
    ...
```

That would become stale.

Instead:

```text
Structural Analysis:
    stable model-level reasoning

Coarse Runtime Validator:
    known runtime capability reasoning

Smoke Test:
    actual pinned runtime proves concrete compatibility
```

---

# 14. Step 8 — Derive Final Degree Domains

Example:

```text
feasible_parallelism_domains {

    TP:
        [1, 2, 4, 8, 16]

    EP:
        [1, 2, 4, 8]

    PP:
        [1, 2, 4]

    CP:
        [1, 2, 4]
}
```

But each degree can carry attached constraints:

```text
TP16:
    requires KV_HEAD_REPLICATION

PP4:
    requires UNEVEN_STAGE_ASSIGNMENT

EP > 1:
    requires DISTRIBUTED_EXPERT_DISPATCH

CP > 1:
    requires DISTRIBUTED_ATTENTION
    requires DISTRIBUTED_KV
```

So the output is a constrained search space, not just integer lists.

---

# 15. Step 9 — Preserve Rejections

Rejected candidates should be durable evidence.

```text
rejections {

    TP32 {
        reason:
            LOCAL_TENSOR_PARTITION_INVALID
    }

    EP16 {
        reason:
            EXPERT_LOCAL_MEMORY_INFEASIBLE
    }

    PP8 {
        reason:
            NO_FEASIBLE_STAGE_PARTITION
    }

    CP8 {
        reason:
            RUNTIME_CAPABILITY_UNSUPPORTED
    }
}
```

This lets future analysis and search avoid rediscovering known-bad families.

---

# 16. Full Structural Analysis Artifact

```text
StructuralParallelismAnalysisArtifact {

    analysis_id

    lineage {
        canonical_model_ref
        quantization_analysis_ref
        memory_analysis_ref
        serving_requirements_ref
        traffic_distribution_ref
    }

    runtime_environment {
        runtime
        version
        hardware
        environment_contract_ref
    }

    model_structure {
        num_layers
        hidden_size
        intermediate_size
        query_heads
        kv_heads
        head_dim

        moe {
            enabled
            routed_experts
            shared_experts
            experts_per_token
        }

        max_context
    }

    tensor_parallelism {
        candidates
        constraints
        rejections
    }

    expert_parallelism {
        candidates
        constraints
        rejections
    }

    pipeline_parallelism {
        candidates
        stage_partitions
        constraints
        rejections
    }

    context_parallelism {
        candidates
        applicability
        constraints
        rejections
    }

    composition_rules {
        ...
    }

    runtime_capabilities {
        ...
    }

    feasible_parallelism_domains {
        TP
        EP
        PP
        CP
    }

    derived_constraints {
        ...
    }

    unresolved {
        exact_per_rank_memory
        communication_cost
        collective_latency
        pipeline_bubble
        expert_load_imbalance
        CP_attention_communication_cost
        best_topology
    }
}
```

---

# 17. Compact Handoff to Candidate Generation

The verbose artifact is durable evidence.

Candidate Generation can consume a compact projection:

```text
StructuralCandidateSpace {

    TP:
        [1, 2, 4, 8, 16]

    EP:
        [1, 2, 4, 8]

    PP:
        [1, 2, 4]

    CP:
        [1, 2, 4]


    constraints {

        TP16:
            requires KV_HEAD_REPLICATION

        PP4:
            requires UNEVEN_STAGE_PARTITION

        EP > 1:
            requires DISTRIBUTED_EXPERT_DISPATCH

        CP > 1:
            requires DISTRIBUTED_ATTENTION
            requires DISTRIBUTED_KV
    }


    composition {

        TP_PP:
            ALLOWED_IF_RUNTIME_SUPPORTED

        TP_EP:
            ALLOWED_IF_RUNTIME_SUPPORTED

        PP_EP:
            ALLOWED_IF_RUNTIME_SUPPORTED

        TP_CP:
            ALLOWED_IF_RUNTIME_SUPPORTED

        PP_CP:
            ALLOWED_IF_RUNTIME_SUPPORTED

        EP_CP:
            ALLOWED_IF_RUNTIME_SUPPORTED
    }
}
```

---

# 18. Example Candidate Tuples Generated Later

Candidate Generation may now build:

```text
T1:
    TP4
    PP1
    EP1
    CP1

T2:
    TP2
    PP2
    EP1
    CP1

T3:
    TP2
    PP1
    EP2
    CP1

T4:
    TP4
    PP1
    EP4
    CP1

T5:
    TP4
    PP1
    EP1
    CP2
```

Those tuples still require:

```text
per-rank memory validation
communication analysis
runtime smoke testing
```

Structural Analysis does not declare them production-ready.

---

# 19. Lifecycle State

After Structural Analysis:

```text
STRUCTURALLY_ANALYZED
```

means:

> **The topology family passed model-level structural reasoning and coarse runtime checks.**

It does not mean:

```text
EXECUTABLE
```

Lifecycle:

```text
STRUCTURALLY_ANALYZED
        |
        v
Candidate Generation
        |
        v
Per-Rank Memory Validation
        |
        v
Communication Analysis
        |
        v
Analyzed Serving Candidate
        |
        v
Compatibility Smoke Test
        |
        v
VERIFIED_EXECUTABLE
        |
        v
Performance / Quality Experiments
        |
        v
QUALIFIED
```

---

# 20. Relationship to Compatibility Smoke Tests

Structural Analysis may conclude:

```text
TP4 + EP4 + PP1 + CP1
    should work
```

The smoke test runs the real environment:

```text
actual model artifact
+
actual quantized checkpoint
+
actual runtime build
+
actual CUDA stack
+
actual hardware
+
actual topology
```

and checks:

```text
environment initializes
process groups initialize
model loads
quantized backend resolves
KV allocates
a few requests execute
```

So:

```text
Structural Analysis:
    predicts legality

Smoke Test:
    proves executability
```

---

# 21. Why This Design Is Maintainable

The platform owns stable reasoning:

```text
head partitionability
expert ownership
stage partitionability
context semantics
memory implications
required runtime capabilities
```

The runtime owns fast-changing implementation details:

```text
exact attention backend
exact fused kernel
exact packing path
exact process-group implementation
exact distributed-attention kernel
```

This avoids duplicating vLLM/SGLang internals inside Search Foundation.

---

# 22. Final Decision Flow

```text
CanonicalModel
MemoryAnalysis
QuantizationAnalysis
ServingRequirements
TrafficDistribution
Runtime/Hardware
        |
        v
Determine parallelism relevance
        |
        +-------------------+
        |                   |
        v                   v
       TP                  EP
operator legality       expert legality
        |                   |
        +---------+---------+
                  |
                  v
                 PP
          stage legality
                  |
                  v
                 CP
       context/attention legality
                  |
                  v
      Cross-Parallelism Composition
                  |
                  v
      Coarse Runtime Capability Check
                  |
                  v
StructuralParallelismAnalysisArtifact
                  |
                  v
      StructuralCandidateSpace
                  |
                  v
       Candidate Topology Generation
```

---

# 23. Interview Summary

> **Structural Parallelism Analysis converts model structure, quantization constraints, memory pressure, workload requirements, and coarse runtime capabilities into a constrained TP/EP/PP/CP search space. TP checks whether operators and attention heads can be partitioned. EP checks whether experts can be placed and routed correctly. PP checks whether the model's layer stack can be divided into feasible contiguous stages. CP checks whether long context can be physically sharded while preserving exact attention, position, mask, and KV semantics. The phase then derives cross-parallelism composition requirements, performs coarse runtime capability validation, records valid degree domains and rejection reasons, and hands a constrained candidate space to topology generation. It predicts structural feasibility; later smoke tests prove concrete executability.**

---

# 24. Locked Mental Model

```text
TP:
    Can I cut operators?

EP:
    Can I distribute specialists?

PP:
    Can I divide the assembly line?

CP:
    Can I divide the conversation history?
```

Then:

```text
Can those dimensions coexist?

Does the runtime broadly support what they require?

Which choices are impossible before we spend GPU time?
```

That is the complete role of Structural Parallelism Analysis.
