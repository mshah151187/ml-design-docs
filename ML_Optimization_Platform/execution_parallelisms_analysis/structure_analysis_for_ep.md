# Structural Parallelism Analysis for Expert Parallelism (EP)

> **Purpose:** Determine whether a proposed Expert Parallel degree is structurally legal before evaluating performance. This phase checks whether routed experts can be distributed across ranks, whether router decisions map to valid expert owners, whether each expert can execute locally, and whether the runtime supports the required EP composition.

---

# 1. Natural-Language Interpretation of EP

Expert Parallelism means:

> **Distribute whole expert modules across GPUs, then send each token activation to the GPU that owns the expert selected by the router.**

Example:

```text
num_experts = 8
top_k       = 2
EP          = 4

GPU0: E0 E1
GPU1: E2 E3
GPU2: E4 E5
GPU3: E6 E7
```

For:

```text
T0 -> E1, E6
T1 -> E2, E4
T2 -> E0, E3
T3 -> E5, E7
```

the runtime physically dispatches token activations to the ranks that own those experts.

The central question is:

> **Can the expert set be distributed across ranks such that routing semantics remain valid, each selected expert can execute on its owner, and the runtime can dispatch and combine token results correctly?**

---

# 2. EP vs TP

```text
TP:
    shard one operator / matrix across GPUs

EP:
    shard the set of experts across GPUs
```

Natural interpretation:

```text
TP:
    several GPUs collaborate inside one specialist

EP:
    different specialists live on different GPUs
```

EP usually preserves each expert as an intact module. Therefore it tends to have fewer matrix-divisibility issues than TP, but more placement and routing issues.

---

# 3. Main Structural Checks

```text
EP Structural Analysis

1. Expert Set Partitionability
   - routed experts evenly divisible by EP?
   - if not, uneven placement supported?
   - shared experts handled separately?

2. Routing Compatibility
   - top_k semantics supported?
   - global expert IDs map to owners?
   - multi-rank dispatch supported?
   - expert replication required or optional?

3. Expert Local Executability
   - one expert fits on its assigned rank?
   - expert tensors independently placeable?
   - quantization representation stays valid?
   - expert-local TP required?

4. Runtime Composition
   - runtime supports EP?
   - requested EP degree supported?
   - dispatch/combine implementation available?
   - EP + TP / PP composition supported?

5. Structural Result
   - VALID
   - VALID_WITH_UNEVEN_PLACEMENT
   - VALID_WITH_EXPERT_REPLICATION
   - REQUIRES_EXPERT_TP
   - UNSUPPORTED
```

---

# 4. Expert Set Partitionability

For:

```text
num_experts = 8
EP = 4
```

we get:

```text
8 / 4 = 2 experts/rank
```

so:

```text
GPU0: E0 E1
GPU1: E2 E3
GPU2: E4 E5
GPU3: E6 E7
```

Natural-language interpretation:

> **Can the specialist workforce be distributed evenly across participating GPUs?**

---

# 5. Uneven Expert Placement

For:

```text
num_experts = 10
EP = 4
```

equal placement is impossible, but whole experts can still be assigned unevenly:

```text
GPU0: E0 E1 E2
GPU1: E3 E4 E5
GPU2: E6 E7
GPU3: E8 E9
```

Therefore:

```text
num_experts % EP != 0
```

does not necessarily mean invalid.

Instead it means:

```text
Required Capability:
    UNEVEN_EXPERT_PLACEMENT
```

The coarse Runtime Capability Validator decides whether the pinned runtime supports that.

---

# 6. Shared Experts

Some models have:

```text
routed experts
+
shared experts
```

Example:

```text
8 routed experts
1 shared expert
```

The routed experts are sparse, while a shared expert may execute for every token.

So they should not be treated as nine equivalent experts.

The routed experts might be EP-sharded while the shared expert is:

```text
replicated
```

or:

```text
tensor-parallelized
```

or handled specially by the runtime.

Natural-language interpretation:

> **Routed experts are specialists visited selectively. A shared expert is a mandatory department every token may have to visit.**

---

# 7. Routing Compatibility

Suppose:

```text
top_k = 2

T0 -> E1, E6
```

with:

```text
E1 -> GPU0
E6 -> GPU3
```

The runtime must send the same token representation to two destinations:

```text
             T0
            /  \
           v    v
         GPU0  GPU3
          E1    E6
           \    /
            \  /
             v
           combine
```

The correct structural question is not:

```text
EP >= top_k?
```

It is:

> **Can the distributed dispatch/combine mechanism represent the model's top-k routing semantics?**

---

# 8. Global Expert ID to Owner Mapping

The router produces global expert IDs:

```text
router(T0):
    [1, 6]
```

The runtime needs:

```text
expert_id -> owner_rank -> local_expert_index
```

Example:

```text
E0 -> GPU0
E1 -> GPU0
E2 -> GPU1
E3 -> GPU1
E4 -> GPU2
E5 -> GPU2
E6 -> GPU3
E7 -> GPU3
```

Then:

```text
T0 -> E1, E6
```

becomes:

```text
T0 copy #1 -> rank 0, local expert 1
T0 copy #2 -> rank 3, local expert 0
```

Natural-language interpretation:

> **The router tells us which specialists are needed; the runtime needs an address book telling it where those specialists live.**

---

# 9. Dispatch and Combine

MoE has two distributed directions.

```text
DISPATCH:
    token activations -> expert owners

COMBINE:
    expert outputs -> original token order / logical owner
```

The runtime must preserve:

```text
token identity
expert identity
router weight
destination rank
original token position
```

Natural-language interpretation:

> **Sending work to specialists is only half the job; their answers must be collected and attached back to the right token.**

---

# 10. Expert Replication

A pure EP topology places each expert once.

If one expert is very hot, it may become a bottleneck.

One optional technique is:

```text
GPU1: E3 copy A
GPU3: E3 copy B
```

This requires:

```text
identical expert weights
replica-aware routing
consistent semantics
```

The structural analyzer should distinguish:

```text
NOT_REQUIRED
OPTIONAL
REQUIRED
```

Natural-language interpretation:

> **If one specialist becomes too popular, we may place another identical copy elsewhere and split the queue.**

---

# 11. Expert Weights as Placement Units

An expert is usually an independent FFN:

```text
Expert 7:
    gate_proj_7
    up_proj_7
    down_proj_7
```

That means the entire expert can usually be placed on one rank.

Natural-language interpretation:

> **Can we move one specialist's entire desk to one GPU without needing pieces of another specialist?**

If the architecture has shared parameters between experts, those need separate modeling.

---

# 12. Quantization Representation Under EP

If expert weights are quantized:

```text
FP8
```

or:

```text
INT4
group_size = 128
```

the full representation normally travels with the expert:

```text
Expert E3 {
    qweight
    scales
    zero_points
}
```

This is structurally simpler than TP because EP does not normally cut through the expert's matrices.

Natural-language interpretation:

> **When the specialist moves, the compressed weights and the metadata needed to interpret them move together.**

---

# 13. Expert-Local TP

An expert may be too large to fit on one rank.

Example:

```text
expert_size = 20 GB
available expert budget/rank = 8 GB
```

Pure EP cannot place one whole expert on a rank.

We may need TP inside the expert:

```text
Expert E0
   |
   +--> GPU0 shard
   +--> GPU1 shard
```

Natural-language interpretation:

> **EP chooses which specialist team handles the token; TP divides that specialist's internal work among multiple GPUs.**

---

# 14. EP + TP Composition

Suppose:

```text
8 GPUs
EP = 4
TP = 2
```

Conceptually:

```text
EP Group 0:
    GPU0 + GPU1
    experts E0,E1

EP Group 1:
    GPU2 + GPU3
    experts E2,E3

EP Group 2:
    GPU4 + GPU5
    experts E4,E5

EP Group 3:
    GPU6 + GPU7
    experts E6,E7
```

For a token routed to E6:

```text
router
    |
    v
choose E6
    |
    v
EP dispatch to owning group
    |
    v
TP2 executes E6
```

The runtime must support both communication domains.

---

# 15. Router Placement

For EP alone, router weights are typically small and can be replicated.

Example:

```text
W_router [hidden_size, num_experts]

hidden_size = 8192
num_experts = 64
```

This is only about 524K parameters.

A natural layout is:

```text
GPU0: router copy + E0,E1
GPU1: router copy + E2,E3
GPU2: router copy + E4,E5
GPU3: router copy + E6,E7
```

Natural-language interpretation:

> **The specialists are distributed, but every office keeps a copy of the dispatcher.**

---

# 16. Can the Router Be Sharded?

Yes, although it is usually unnecessary for EP alone.

Two conceptual forms exist.

## Hidden-Dimension Sharding

```text
x = [x0 | x1 | x2 | x3]

partial_logits_i = xi @ Wi

router_logits =
    sum(partial_logits_i)
```

This requires a collective.

Natural interpretation:

> **Each GPU contributes part of the dispatcher's opinion.**

## Expert-Dimension Sharding

Each rank computes scores for only a subset of experts:

```text
GPU0 -> E0-E15
GPU1 -> E16-E31
GPU2 -> E32-E47
GPU3 -> E48-E63
```

Global top-k then requires coordination.

Natural interpretation:

> **Each office knows only some specialists, so offices must compare recommendations before selecting the global winners.**

Because routing is small relative to expert compute, replication is usually the simplest EP baseline.

---

# 17. Runtime Composition

A mathematically valid EP layout still requires runtime support for:

```text
token grouping
per-destination counts
dispatch
expert execution
return communication
weighted combine
```

Search Foundation should not reproduce the runtime's exact dispatch kernels.

Instead it derives required capabilities:

```text
EXPERT_PARALLELISM
MULTI_DESTINATION_DISPATCH
DISPATCH_COMBINE
UNEVEN_EXPERT_PLACEMENT
EXPERT_REPLICATION
EP_TP_COMPOSITION
```

The coarse Runtime Capability Validator checks these against the pinned runtime/version.

The later compatibility smoke test proves actual executability.

---

# 18. Example A — 8 Experts, EP=4

Assume:

```text
routed_experts = 8
shared_experts = 1
top_k          = 2
expert_dtype   = FP8
EP             = 4
```

Placement:

```text
GPU0: E0 E1
GPU1: E2 E3
GPU2: E4 E5
GPU3: E6 E7
```

Result:

```text
expert divisibility:
    VALID

routing:
    VALID

quantization:
    VALID

runtime capability:
    REQUIRED: EP4 + dispatch/combine

status:
    VALID
```

Shared expert handling is analyzed separately.

---

# 19. Example B — 10 Experts, EP=4

```text
10 % 4 != 0
```

Possible placement:

```text
GPU0: 3 experts
GPU1: 3 experts
GPU2: 2 experts
GPU3: 2 experts
```

Result:

```text
expert divisibility:
    NOT EVENLY DIVISIBLE

required capability:
    UNEVEN_EXPERT_PLACEMENT

status:
    VALID_IF_RUNTIME_SUPPORTS_UNEVEN_PLACEMENT
```

---

# 20. Example C — Whole Expert Does Not Fit

Suppose:

```text
expert_size = 25 GB
available expert memory/rank = 16 GB
```

Candidate:

```text
EP = 4
TP = 1
```

Result:

```text
full_expert_fits_per_rank:
    false

status:
    REQUIRES_EXPERT_TP

minimum_expert_tp:
    >= 2
```

Memory Analysis therefore influences topology generation.

---

# 21. Relationship to Memory Analysis

Memory Analysis provides:

```text
expert weight memory
dense model memory
KV pressure
activation pressure
runtime reserve
usable HBM/rank
```

EP Structural Analysis uses those to answer:

```text
Does one full expert fit?

If not:
    what expert-local TP is required?
```

Conceptually:

```text
MemoryProfile
      |
      v
EP Structural Analysis
      |
      +--> whole expert fits
      |
      or
      |
      +--> requires TP inside expert
```

---

# 22. Recommended Output Contract

```text
EPStructuralAnalysis {

    ep_degree:
        4

    expert_partition {
        routed_experts:
            8

        experts_per_rank:
            2

        placement:
            EVEN

        shared_experts:
            1

        shared_expert_policy_requirement:
            ...
    }

    router_placement {
        mode:
            REPLICATED

        sharding_required:
            false
    }

    routing {
        top_k:
            2

        global_to_local_mapping:
            VALID

        multi_rank_dispatch_required:
            true

        combine_required:
            true

        expert_replication:
            NOT_REQUIRED
    }

    expert_local_execution {
        full_expert_fits_per_rank:
            true

        quantization_representation:
            VALID

        expert_local_tp_required:
            false

        minimum_expert_tp:
            1
    }

    runtime_requirements {
        expert_parallelism:
            REQUIRED

        dispatch_combine:
            REQUIRED

        uneven_placement:
            NOT_REQUIRED

        expert_replication:
            NOT_REQUIRED

        ep_tp_composition:
            NOT_REQUIRED
    }

    coarse_runtime_validation {
        runtime:
            ...

        version:
            ...

        status:
            SUPPORTED
            | UNSUPPORTED
            | UNKNOWN
    }

    result {
        status:
            VALID
            | VALID_WITH_UNEVEN_PLACEMENT
            | VALID_WITH_EXPERT_REPLICATION
            | REQUIRES_EXPERT_TP
            | UNSUPPORTED

        constraints:
            [...]

        rejection_reasons:
            [...]
    }
}
```

---

# 23. Analysis-Time vs Execution-Time Validation

```text
EP Structural Analysis
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
        +--> runtime initializes?
        +--> EP groups initialize?
        +--> dispatch/combine succeeds?
        +--> a few requests complete?
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

# 24. What This Phase Does Not Decide

Structural EP Analysis does not yet answer:

```text
Which EP degree gives best throughput?

How expensive is all-to-all?

Will expert imbalance dominate TPOT?

Should hot experts be replicated for performance?

Is EP2 cheaper than EP4?
```

Those belong to later communication analysis, benchmarking, and objective-driven search.

---

# 25. Interview Summary

> **Expert Parallelism distributes whole routed expert modules across GPUs. Structural EP analysis checks whether the expert set can be placed across ranks, whether global router decisions map to valid owners, whether multi-destination dispatch and result combination preserve routing semantics, whether each expert can execute locally with its quantization metadata intact, and whether the runtime supports the requested EP composition. Unlike TP, EP usually keeps an expert intact, so matrix divisibility is less central; the dominant structural issues are placement, routing, shared experts, local expert fit, and composition with TP.**

---

# 26. Locked Mental Model

```text
MANY SPECIALISTS
      |
      v
PLACE DIFFERENT SPECIALISTS ON DIFFERENT GPUS
      |
      v
ROUTER CHOOSES SPECIALISTS
      |
      v
SEND TOKEN TO THE GPU THAT OWNS THEM
      |
      v
COLLECT THEIR ANSWERS
```

Every structural question becomes:

```text
1. Can the specialists be placed?
2. Can the router address them?
3. Can tokens reach them and return?
4. Can each specialist execute locally?
5. Does the runtime support the required composition?
```
