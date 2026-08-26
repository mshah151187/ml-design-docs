# Component — Planner (KV-Tier / P-D Updated)

## 1. Why Planner Exists

Scheduling policy cannot make a good decision from a replica list alone.

It needs a request-scoped view combining:

```text
compute eligibility
P/D worker profiles
current load
expected service rates
router outstanding work
KV locality
KV tier residency
exact-transform options
network/topology cost
request SLO
```

The Planner's job is therefore:

> **Build a trustworthy, bounded set of execution-topology candidates from distributed state.**

It is still a fact/context builder.

The SLO/cost policy remains separate.

---

# 2. What Changed From Earlier V5?

Earlier V5 candidate:

```text
destination replica
+
materialization chain
```

With P/D disaggregation and an external KV tier, a request may require a small execution topology:

```text
LOCAL_PD(D7)

P3 -> D7

KV_CPU -> D7

KV_CPU -> P3 -> D7

worker D2 KV -> P3 -> D7
```

So the Planner now assembles:

```text
ExecutionTopologyCandidate[]
```

rather than only `ReplicaExecutionContext[]`.

---

# 3. Why Not Immediately Create InferenceExecutionGraphPlanner?

Because the graph is still bounded and regular.

Base vocabulary:

```text
optional context source
optional Prefill stage
one terminal Decode/execution stage
```

That remains understandable inside Planner.

Future extraction trigger:

```text
Encode -> Prefill -> Decode
semantic KV mapper stages
speculative draft/verifier graphs
multiple model stages
complex DAG retries
```

At that point:

```text
Planner
    -> InferenceExecutionGraphPlanner
```

becomes a useful separation.

For final V5, it remains a documented extraction point.

---

# 4. Input

```text
PlanningRequest {
    request_id
    model_phase

    target_model_context_artifact

    latency_slo
    cost_policy
    routing_deadline

    session_metadata?
}
```

The exact target-model artifact is required before precise KV lookup.

---

# 5. Data Dependencies

Planner materializes:

```text
ReplicaDiscovery
ServingProfileRegistry

EngineState
RouterOutstandingState
PerformanceState

KVDirectoryIndex

network/topology state
optional transfer-performance state
```

Avoid request-time fanout to every model server or KV node.

---

# 6. Step 1 — Compute Eligibility

Filter P and D workers by hard requirements:

```text
model revision
role capability: PREFILL / DECODE / HYBRID
context length
policy/compliance
adapter support
required attention semantics
engine health
```

This yields:

```text
eligible P workers
eligible D/hybrid workers
```

---

# 7. Step 2 — Query KVDirectoryIndex

Input:

```text
exact logical KV identity
canonical prefix block chain
candidate target representations
```

Receive:

```text
worker-local matches
shared GPU matches
CPU/DRAM matches
NVMe/remote matches
exact-transform options
```

A lookup timeout is optional optimization failure:

```text
assume cache hit = 0
continue with recompute candidates
```

---

# 8. Step 3 — Build Execution Modes

## LOCAL_PD

Decode/hybrid worker performs missing Prefill locally:

```text
existing D prefix
+
uncached suffix
+
Decode
```

## DISAGGREGATED_PD

```text
P -> D
```

P may itself need context materialization before processing the suffix.

## RESTORE_TO_D

```text
KVTier -> D
```

then local suffix Prefill + Decode.

## RESTORE_TO_P_THEN_D

```text
KVTier -> P
P Prefill suffix
P -> D
```

## WORKER_TRANSFER_TO_D

```text
R_source -> D
```

## RECOMPUTE_P_THEN_D

```text
fresh P Prefill -> D
```

Do not enumerate every possible permutation blindly; prune using dominance and bounded shortlists.

---

# 9. Conditional Disaggregation

Use effective uncached work, not raw ISL.

```text
effective_prefill_tokens(D)
=
total_target_tokens
-
reusable_prefix_tokens_on_D
```

Compare local and remote paths.

Local:

```text
D queue
+
local missing Prefill
+
Decode
+
predicted interference cost
```

Remote:

```text
source/context materialization to P
+
P queue
+
P missing Prefill
+
P->D KV transfer
+
D admission
+
Decode
```

Conditional disaggregation is therefore naturally expressed as competing topology candidates.

---

# 10. P-D Pair Selection

Do not independently select "best P" and "best D" and assume their pair is best.

Pair cost depends on:

```text
P queue
D queue
P/D ServingProfiles
network topology
KV bytes transferred
layout transform
D decode SLO
```

But full cross-product:

```text
|P| × |D|
```

may be too expensive.

Practical bounded approach:

```text
1. shortlist best Np Prefill workers
2. shortlist best Nd Decode workers
3. retain topology-local / KV-local candidates
4. evaluate small cross-product
```

---

# 11. ExecutionTopologyCandidate

```text
ExecutionTopologyCandidate {
    topology_id

    mode:
        LOCAL_PD
        DISAGGREGATED_PD

    prefill_worker?
    decode_worker

    source_context_plan

    prefill_materialization_chain[]
    pd_handoff_plan?

    decode_materialization_chain[]

    reusable_prefix_tokens
    uncached_prefill_tokens

    predicted_context_ready_distribution
    predicted_ttft_distribution
    predicted_tpot_distribution
    predicted_e2e_distribution

    expected_cost

    confidence
    state_versions
}
```

---

# 12. Context Materialization Chain

A stage may contain ordered fallbacks.

Example:

```text
Prefill P3 needs existing 120K prefix

1. RESTORE_KV_TIER_CPU(Node9)
2. TRANSFER_WORKER_KV(D2)
3. RECOMPUTE_FROM_TARGET_CONTEXT
```

The Planner chooses the chain order based on predicted feasibility/cost.

The ContextMaterializationCoordinator executes it later.

---

# 13. SLO/Cost Objective

For topology `G`:

```text
Feasible(G)
=
P(
   TTFT <= TTFT_SLO
   AND TPOT <= TPOT_SLO
 ) >= required_probability
```

Among feasible topologies:

```text
G* = argmin ExpectedCost(G)
```

Do not use one arbitrary weighted sum that lets cost buy its way past the SLO.

---

# 14. KV Quality Boundary

A cache hit only receives credit when it can produce valid execution state for the selected quality-approved model.

Example:

```text
Phase 2 selects Llama-70B

KVTier has 128K Llama-3B KV
```

Base V5:

```text
direct cache credit = 0
```

unless:

```text
a separately qualified 3B->70B semantic mapper exists
```

Do not downgrade model quality in Planner just to exploit cache locality.

---

# 15. Topology Pruning

Examples of dominated candidates:

```text
same P
same D
same reusable prefix

CPU restore:
    slower and more expensive than direct worker-local source

=> drop CPU candidate
```

Or:

```text
NVMe restore 200-token prefix
    predicted slower than recompute

=> drop restore
```

Keep request-path candidate count bounded.

---

# 16. Routing Latency Budget

KV/topology planning is optional sophistication.

As routing deadline approaches:

```text
NORMAL:
    precise KV directory
    pair scoring
    tier candidates
    learned latency predictor

DEGRADED:
    local worker cache only
    cheap P/D shortlists
    recompute path

EMERGENCY:
    precomputed healthy P/D or local-hybrid route
```

Never skip:

```text
model validity
policy
session version
KV semantic compatibility
```

Skip optimization, not correctness.

---

# 17. Output

```text
ExecutionTopologyPlan {
    request_id
    model_phase

    status:
        FEASIBLE
        SLO_UNSATISFIABLE

    candidates[]   # bounded Top-K

    planner_version
    predictor_version
    kv_directory_version?
    created_at
}
```

Candidates are ordered SLO-feasible options.

Normal `Kmax` can remain small (for example 3) to preserve queue-arbitrated execution.

---

# 18. Queue-Arbitration Boundary

The terminal Decode/local execution worker remains the request's primary queue/claim owner in base V5.

Why?

It preserves the existing invariant:

```text
one request
    ->
one live terminal execution lease
```

Remote Prefill is a bounded prerequisite stage after that topology wins.

This avoids putting the same logical request independently into both P and D queues and accidentally allowing two topologies to execute concurrently.

A richer graph executor can revisit this later.

---

# 19. Failure Semantics

KVDirectoryIndex unavailable:

```text
build recompute-based candidates
```

KVTier source disappears after planning:

```text
materialization fallback chain handles it
```

P unavailable at execution:

```text
release candidate / retry sibling topology according to plan
```

D cannot JIT allocate:

```text
release terminal claim
allow sibling candidate
```

No cache optimization failure should violate model-quality or session-correctness boundaries.

---

# 20. Design Principles

1. Planner assembles facts/topology candidates; scheduler policy decides.
2. Quality-approved model phases remain upstream.
3. Exact target context is known before precise KV lookup.
4. Cache locality is valuable only inside semantic compatibility.
5. P and D should be scored jointly where transfer cost matters.
6. Effective uncached Prefill is more useful than raw ISL for conditional disaggregation.
7. Worker-local, shared GPU, CPU, NVMe and recompute are competing materialization sources.
8. Tier selection is not hard-coded.
9. Full P×D cross-product is pruned aggressively.
10. Routing sophistication degrades under a hard routing budget.
11. Topology planning does not reserve destination KV while queued.
12. `InferenceExecutionGraphPlanner` is a future extraction point, not mandatory V5 plumbing.

---

# 21. Interview Answer

> **Once I add P/D disaggregation and an external KV tier, my Planner stops thinking only in terms of replicas and starts building a bounded set of execution-topology candidates. A candidate might be local Prefill+Decode on D, P3→D7, CPU-KV→D7, or CPU-KV→P3→D7. I join P/D load and ServingProfiles with exact KV-directory residency, transform support and network cost, compute effective uncached Prefill, then predict TTFT/TPOT feasibility and minimize expected cost among feasible plans. I keep the graph vocabulary deliberately bounded; if the serving architecture later grows into arbitrary Encode/Prefill/Decode or multi-model DAGs, that's when I'd extract an InferenceExecutionGraphPlanner.**
