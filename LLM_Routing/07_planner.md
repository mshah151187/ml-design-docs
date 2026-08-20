# Component — Planner

## Responsibility

The Planner is the **factual context assembler** for one quality-selected model revision. It does not own the replica scheduling policy.

It takes the exact target-model context and constructs the complete `SchedulingProblem` consumed by `SLOCostAwareReplicaScheduler`.

## Input — `PlannerRequest`

```text
PlannerRequest {
    request_id: string
    phase: int

    model_phase_plan: ModelPhasePlan
    target_context: TargetModelContextArtifact

    ttft_slo_ms: float?
    e2e_slo_ms: float?
    slo_probability_target: float
    cost_policy_id: string

    tenant_id: string
    region: string?
    policy_tags: set<string>

    request_received_at: timestamp
}
```

## State/data sources read by Planner

```text
ReplicaDiscoveryCache
ServingProfileCache / KVCompatibilityProfileCache / CostProfileCache
EngineWorkStateCache
EngineCapacityStateCache
RouterOutstandingStateCache
PerformanceStateCache
PreciseKVIndex
CostMetadataCache
```

No remote Kubernetes lookup occurs here.

## Candidate hard filtering

Before building scorer contexts, remove replicas that cannot serve safely/correctly:

```text
wrong model revision
wrong role/adapter/modality
not ready / draining / unhealthy
context length unsupported
hard policy/region mismatch
admission state REJECTING
incompatible execution capability
```

## KV request-key derivation

The target artifact provides exact token IDs. For each distinct candidate `kv_compatibility_id`, Planner/KV client derives namespace-specific canonical block keys and queries precise locality.

## Output — `SchedulingProblem`

```text
SchedulingProblem {
    request: SchedulingRequest
    candidates: list<ReplicaExecutionContext>
    built_at: timestamp
}
```

### `SchedulingRequest`

```text
SchedulingRequest {
    request_id: string
    phase: int

    model_id: string
    model_revision: string

    context_snapshot_id: string
    target_context_artifact_id: string
    token_context_digest: string
    token_ids_ref: string

    session_id: string?
    base_session_version: int?

    input_tokens: int
    expected_output_tokens: int
    max_output_tokens: int

    ttft_slo_ms: float?
    e2e_slo_ms: float?
    slo_probability_target: float

    cost_policy_id: string
    tenant_id: string
    region: string?

    request_received_at: timestamp
}
```

### `ReplicaExecutionContext`

```text
ReplicaExecutionContext {
    replica: ReplicaDescriptor
    serving_profile: ServingProfile

    engine_work: EngineWorkState
    engine_capacity: EngineCapacityState
    router_outstanding: RouterOutstandingState
    performance: ReplicaPerformanceState

    local_prefix_match: PrefixMatch
    transfer_sources: list<KVTransferSource>

    state_freshness: StateFreshness
}
```

### `EngineWorkState`

```text
EngineWorkState {
    queued_requests: int
    running_requests: int

    queued_prefill_tokens: int
    active_prefill_tokens: int

    active_decode_sequences: int
    estimated_remaining_decode_tokens: int

    replica_epoch: int
    sequence: int
    observed_at: timestamp
}
```

### `EngineCapacityState`

```text
EngineCapacityState {
    admission_state: OPEN | PRESSURED | THROTTLED | REJECTING | UNKNOWN
    available_sequence_capacity: int?

    kv_cache_utilization: float
    available_kv_blocks: int
    kv_eviction_pressure: float?

    memory_pressure: float?
    observed_at: timestamp
}
```

### `RouterOutstandingState`

```text
RouterOutstandingState {
    ready_requests: int
    ready_prefill_tokens: int
    ready_decode_tokens: int

    pre_admission_requests: int
    pre_admission_prefill_tokens: int
    pre_admission_decode_tokens: int

    state_version: int
    observed_at: timestamp
}
```

Scorer-facing outstanding work is:

```text
router_outstanding_prefill = ready_prefill + pre_admission_prefill
router_outstanding_decode  = ready_decode  + pre_admission_decode
```

PARKED sibling candidates are excluded while another destination owns a live lease.

### `ReplicaPerformanceState`

```text
ReplicaPerformanceState {
    effective_prefill_tokens_per_sec: float
    effective_decode_tokens_per_sec: float

    ttft_p50_ms: float?
    ttft_p95_ms: float?
    ttft_p99_ms: float?

    tpot_p50_ms: float?
    tpot_p95_ms: float?
    tpot_p99_ms: float?

    e2e_p50_ms: float?
    e2e_p95_ms: float?
    e2e_p99_ms: float?

    sample_count: int
    confidence: float
    observed_at: timestamp
}
```

### `PrefixMatch`

```text
PrefixMatch {
    replica_id: string
    kv_compatibility_id: string

    matched_blocks: int
    matched_tokens: int
    cache_tier: GPU | CPU | DISK | SHARED | NONE

    confidence: TRUSTED | UNCERTAIN
}
```

### `KVTransferSource`

```text
KVTransferSource {
    source_replica_id: string
    source_replica_epoch: int
    source_deployment_id: string

    kv_compatibility_id: string

    transferable_prefix_blocks: int
    transferable_prefix_tokens: int
    estimated_transfer_bytes: int
    source_tier: GPU | CPU | SHARED

    block_manifest_ref: string
    locality_version: int
    observed_at: timestamp
}
```

`block_manifest_ref` is a logical manifest; source pinning later validates that those blocks still exist under the same epoch/context.

### `StateFreshness`

```text
StateFreshness {
    engine_age_ms: float
    router_age_ms: float
    performance_age_ms: float
    kv_age_ms: float

    engine_version: string
    router_version: string
    performance_version: string
    kv_version: string
}
```

## Invariants

- Planner does not rank by policy; it assembles facts.
- All Planner instances consume the same authoritative versioned state streams, though small bounded lag is allowed.
- Context-transfer sources are factual possibilities, not a commitment to transfer.
- No destination KV capacity is reserved here.

## Next component

`SLOCostAwareReplicaScheduler.schedule(SchedulingProblem)`.
