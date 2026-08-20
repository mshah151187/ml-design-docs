# Component — V5 `SLOCostAwareReplicaScheduler`

## Responsibility

Apply scheduling policy to a `SchedulingProblem` and return up to three destination execution plans that satisfy the request SLO with sufficient probability and minimize expected cost.

V5 schedules **destination + context-materialization strategy**, not only destination replica.

## Input

Exactly one `SchedulingProblem` from `07_planner.md`.

## Analytical baseline

For replica `R`:

```text
EnginePrefillWork(R)
  = queued_prefill_tokens + active_prefill_tokens

RouterOutstandingPrefill(R)
  = ready_prefill_tokens + pre_admission_prefill_tokens

EngineDecodeWork(R)
  = estimated_remaining_decode_tokens

RouterOutstandingDecode(R)
  = ready_decode_tokens + pre_admission_decode_tokens
```

Approximate backlog pressure in the time domain:

```text
PrefillBacklogTime
  = (EnginePrefillWork + RouterOutstandingPrefill) / EffectivePrefillRate

DecodeBacklogTime
  = (EngineDecodeWork + RouterOutstandingDecode) / EffectiveDecodeRate
```

This is an analytical pressure approximation; continuous batching can create nonlinear overlap. The learned predictor replaces the additive approximation in the final V5 policy.

## Context-materialization modes

```text
LOCAL_KV_REUSE
TRANSFER_KV
RECOMPUTE_FROM_CONTEXT
```

For each destination, Scheduler constructs an ordered chain.

Examples:

```text
R1:
  0 LOCAL_KV_REUSE
  1 RECOMPUTE_FROM_CONTEXT

R3:
  0 TRANSFER_KV(source=R1)
  1 RECOMPUTE_FROM_CONTEXT
```

A transfer option is eligible only when source/destination share the required `kv_compatibility_id` and the exact target context/prefix identity is compatible.

## Learned latency-predictor features

V5 does not require the router design to specify the training algorithm, but the decision-time feature contract is explicit.

### Request/context features

```text
input_tokens
expected_output_tokens
max_output_tokens
operation
session_mode flag
base_session_version / turn depth bucket (non-identifying feature)
context-compaction flag/policy
```

### Destination execution features

```text
ServingProfile identity/features
hardware class/count/memory
engine + version
parallelism
precision/quantization
context capacity
prefix-caching capability
```

### Dynamic destination features

```text
EngineWorkState
EngineCapacityState
RouterOutstandingState
ReplicaPerformanceState
state ages/confidence
```

### KV/context features

```text
local reusable prefix tokens
local cache tier
context materialization mode
transferable prefix tokens
estimated transfer bytes
source cache tier
source/destination topology features when available
recompute token count
```

### Labels predicted/derived

The production predictor may directly estimate distributions/probabilities for:

```text
TTFT
TPOT
E2E latency
context-materialization latency
P(latency <= request SLO)
```

## Cost estimator features

Conceptually:

```text
predicted GPU execution time
ServingProfile/hardware cost prior
prefill/recompute work
transfer bytes/bandwidth cost
source participation cost
context memory occupancy cost where modeled
```

The detailed generation of cost priors is outside this component; Scheduler consumes them.

## SLO objective

For candidate execution plan `C`:

```text
SLOProbability(C) = P(Latency(X, C) <= Deadline(X))
```

Feasible set:

```text
Feasible = { C | SLOProbability(C) >= request.slo_probability_target }
```

Among feasible plans, minimize expected cost.

SLO is a constraint; cost is the optimization objective.

## Top-K semantics

Return **up to 3 distinct destination replicas**.

Do not add an SLO-infeasible replica merely to fill K=3.

For one destination replica, attach one ordered `ContextMaterializationChain` rather than output several independent copies of the same destination.

## No-feasible-replica behavior

If no plan meets the SLO:

```text
plan_status = SLO_UNSATISFIABLE
```

Return up to three least-risk degraded candidates according to a documented policy such as:

```text
highest SLO probability first
then lowest expected violation magnitude
then cost
```

Never silently label them SLO-feasible.

## Output — `ModelReplicaPlan`

```text
ModelReplicaPlan {
    request_id: string
    phase: int
    model_id: string
    model_revision: string

    plan_status: SLO_FEASIBLE | SLO_UNSATISFIABLE
    candidates: list<ReplicaExecutionPlan>   # <= 3 distinct destinations

    scheduler_policy_version: string
    latency_model_version: string
    cost_model_version: string

    planned_at: timestamp
}
```

### `ReplicaExecutionPlan`

```text
ReplicaExecutionPlan {
    execution_plan_id: string

    request_id: string
    phase: int

    destination_replica_id: string
    destination_replica_epoch: int
    destination_deployment_id: string
    destination_serving_profile_id: string
    destination_kv_compatibility_id: string

    candidate_rank: int

    predicted_ttft_p50_ms: float?
    predicted_ttft_p95_ms: float?
    predicted_ttft_p99_ms: float?
    predicted_e2e_p50_ms: float?
    predicted_e2e_p95_ms: float?
    predicted_e2e_p99_ms: float?

    predicted_slo_probability: float
    predicted_expected_cost: float

    materialization_chain: ContextMaterializationChain

    input_tokens: int
    expected_output_tokens: int

    engine_state_version: string
    router_state_version: string
    performance_state_version: string
    kv_state_version: string

    planned_at: timestamp
    plan_ttl_ms: int
}
```

### `ContextMaterializationChain`

```text
ContextMaterializationChain {
    plans: list<ContextMaterializationPlan>
}
```

### `ContextMaterializationPlan`

```text
ContextMaterializationPlan {
    plan_index: int
    mode: LOCAL_KV_REUSE | TRANSFER_KV | RECOMPUTE_FROM_CONTEXT

    kv_compatibility_id: string

    source_replica_id: string?
    source_replica_epoch: int?
    source_block_manifest_ref: string?

    expected_reusable_prefix_tokens: int
    expected_transferable_prefix_tokens: int
    estimated_transfer_bytes: int
    expected_recompute_tokens: int

    predicted_context_ready_ms: float
    predicted_materialization_cost: float
}
```

## Invariants

- Ranking policy lives here, not in Planner.
- Destination capacity is observed for scoring but not reserved while waiting in queues.
- Transfer source state can become stale; the materialization chain explicitly contains recompute fallback.
- Queue arbitration still chooses which destination actually owns execution.

## Next component

`ExecutionCoordinator` activates a `ModelReplicaPlan`; it does not rescore it.
