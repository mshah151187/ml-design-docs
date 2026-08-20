# Component — Feedback / Performance Pipeline

## Responsibility

Join decision-time features with actual context-materialization/execution outcomes to maintain online performance state and produce training examples for the learned V5 latency/SLO predictor.

The design specifies feature/event contracts but intentionally does not prescribe the model-training algorithm.

## Event 1 — `SchedulingDecisionEvent`

Emitted by Planner/ReplicaScheduler at decision time.

```text
SchedulingDecisionEvent {
    request_id: string
    phase: int
    execution_plan_id: string

    model_id: string
    model_revision: string

    destination_replica_id: string
    destination_serving_profile_id: string
    destination_kv_compatibility_id: string

    input_tokens: int
    expected_output_tokens: int

    selected_context_mode: ContextMode
    local_reusable_prefix_tokens: int
    transferable_prefix_tokens: int
    estimated_transfer_bytes: int
    expected_recompute_tokens: int

    engine_work_snapshot: EngineWorkState
    engine_capacity_snapshot: EngineCapacityState
    router_outstanding_snapshot: RouterOutstandingState
    performance_snapshot: ReplicaPerformanceState

    state_versions: map
    state_ages_ms: map

    predicted_ttft_quantiles: map
    predicted_e2e_quantiles: map
    predicted_slo_probability: float
    predicted_expected_cost: float

    latency_model_version: string
    cost_model_version: string
    scheduler_policy_version: string

    decided_at: timestamp
}
```

This snapshot prevents training on future state that was not available to the decision.

## Event 2 — `ContextMaterializationObservation`

```text
ContextMaterializationObservation {
    request_id: string
    phase: int
    execution_plan_id: string

    destination_replica_id: string
    mode_used: ContextMode
    source_replica_id: string?

    locally_reused_tokens: int
    transferred_tokens: int
    recomputed_tokens: int
    transfer_bytes: int

    materialization_latency_ms: float
    transfer_bandwidth_bytes_per_sec: float?

    fallback_plan_index_used: int
    attempt_failures: list<string>

    observed_at: timestamp
}
```

## Event 3 — `ExecutionObservation`

Assembled from router/Executioner/model-server timestamps.

```text
ExecutionObservation {
    request_id: string
    phase: int
    execution_plan_id: string

    replica_id: string
    serving_profile_id: string

    router_queue_wait_ms: float
    claim_to_materialization_start_ms: float
    context_materialization_ms: float
    backend_admission_wait_ms: float

    prefill_ms: float
    ttft_ms: float
    decode_ms: float
    tpot_ms: float?
    session_commit_ms: float?
    e2e_ms: float

    actual_output_tokens: int

    completion_status: SUCCESS | FAILED | CANCELLED
    failure_code: string?

    observed_at: timestamp
}
```

## Derived outputs

### Online `ReplicaPerformanceState`

Windowed/EWMA/quantile state materialized back to Planners:

```text
effective prefill/decode rates
TTFT P50/P95/P99
TPOT P50/P95/P99
E2E P50/P95/P99
materialization latency distributions by mode
transfer bandwidth distributions
admission rejection rate
sample count/confidence
```

### Training example

```text
TrainingExample {
    decision_features
    selected_context_mode
    realized_materialization_metrics
    realized_execution_metrics
    realized_slo_success
    realized_cost
}
```

## Learned scheduler feature categories

The training system may use:

```text
request size/output estimate
ServingProfile
engine work/capacity
RouterOutstanding state
KV local hit
transfer source/tier/bytes
context mode
state freshness
historical performance distributions
```

Do not use raw `session_id` as a predictive identity feature. Session-derived non-identifying structural features such as context length/turn-depth bucket may be used if product/privacy policy permits.

## Selection bias caveat

Production only observes the realized latency of the execution plan that actually runs. Counterfactual outcomes for unselected replicas are unobserved. Controlled exploration/shadowing/calibration are advanced training concerns; the serving contract does not require resolving them inline.
