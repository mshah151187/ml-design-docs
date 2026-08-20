# Component — ExecutionCoordinator

## Responsibility

Own request **model-phase progression**, admission/start budget, and activation of each phase's `ModelReplicaPlan`.

It does not score replicas, execute model RPCs, or run context-materialization fallbacks.

## Input — `ExecutionRouteContext`

```text
ExecutionRouteContext {
    request_id: string
    context_snapshot_id: string

    phase_plans: list<ProgressivePhaseExecutionPlan>

    request_deadline: timestamp?
}
```

Each progressive phase:

```text
ProgressivePhaseExecutionPlan {
    phase: int
    model_id: string
    model_revision: string

    target_context_artifact_ref: string

    status: PLANNING | READY | FAILED
    model_replica_plan: ModelReplicaPlan?
    planning_error: string?
}
```

The current phase is **not** authoritative in this local object. `RequestExecutionRegistry.phase` is authoritative.

## Request creation

The coordinator creates:

```text
RequestExecutionState {
    phase = 0
    state = PENDING
    phase_admission_deadline = now + configured_budget
}
```

## Phase activation

For the active phase's `ModelReplicaPlan`, enqueue each candidate destination (up to 3) into its replica queue.

The queue entry carries a reference to the full `ReplicaExecutionPlan`, including its context-materialization chain.

## Fallback rule

If registry state remains:

```text
PENDING
```

past the current phase admission/start budget, atomically attempt:

```text
phase N -> phase N+1
```

If an Executioner concurrently changes the request to:

```text
CLAIMED
MATERIALIZING_CONTEXT
EXECUTING_PRE_STREAM
STREAMING
COMMITTING_SESSION
```

the phase advance loses and V5 does not hedge automatically.

## Input/output APIs

### Input to `execute`

```text
ExecuteRequest {
    execution_route_context: ExecutionRouteContext
}
```

### Output

```text
ExecutionCoordinatorResult {
    request_id: string
    final_phase: int
    final_state: SUCCESS | FAILED | CANCELLED | REQUEST_INVALID

    winning_execution_plan_id: string?
    winning_replica_id: string?

    response_ref: string?
    failure_code: string?
}
```

## Invariants

- Same-model retry is decentralized through parked queue candidates, not a central `R1 -> R3 -> R5` loop.
- Phase advancement is atomic against `TryClaim`.
- Fallback planning can run concurrently but primary execution does not wait for fallback planning.
- Hedged inference remains separate from basic V5 phase semantics.
