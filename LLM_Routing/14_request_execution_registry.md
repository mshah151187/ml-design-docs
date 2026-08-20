# Component — RequestExecutionRegistry

## Responsibility

Own compact, request-scoped execution coordination state: active model phase, one destination execution owner, lease/fencing, execution state, and terminal outcome.

It does not store Top-K queue membership, dynamic replica load, or the durable session transcript.

## Authoritative state — `RequestExecutionState`

```text
RequestExecutionState {
    request_id: string

    phase: int
    state: PENDING
         | CLAIMED
         | MATERIALIZING_CONTEXT
         | EXECUTING_PRE_STREAM
         | STREAMING
         | COMMITTING_SESSION
         | SUCCESS
         | FAILED
         | CANCELLED
         | REQUEST_INVALID

    execution_plan_id: string?

    owner_replica_id: string?
    owner_replica_epoch: int?
    owner_executioner_id: string?

    lease_id: string?
    lease_expires_at: timestamp?

    attempt_count: int
    materialization_attempt_count: int

    phase_admission_deadline: timestamp?
    request_deadline: timestamp?

    response_ref: string?
    failure_code: string?

    version: int
    updated_at: timestamp
}
```

## API — `CreateRequest`

### Input

```text
CreateRequest {
    request_id: string
    initial_phase: int
    phase_admission_deadline: timestamp?
    request_deadline: timestamp?
}
```

### Output

```text
CreateRequestResult {
    status: CREATED | ALREADY_EXISTS
    state: RequestExecutionState
}
```

## API — `TryClaim`

### Input

```text
TryClaimRequest {
    request_id: string
    phase: int
    execution_plan_id: string

    replica_id: string
    replica_epoch: int
    executioner_id: string

    lease_ttl_ms: int
}
```

### Output

```text
TryClaimResult {
    status: CLAIMED
          | ALREADY_OWNED_BY_CALLER
          | OWNED_BY_OTHER
          | STALE_PHASE
          | STREAMING
          | TERMINAL
          | MISSING

    state: RequestExecutionState?
}
```

The transition must be atomic against `AdvancePhase`.

## API — state transitions

All mutations are conditional on expected `phase + lease_id` where ownership is required:

```text
mark_materializing_context
mark_executing_pre_stream
mark_streaming
mark_committing_session
mark_success
release_for_retry
renew_lease
mark_terminal_failure
cancel_request
```

## API — `AdvancePhase`

### Input

```text
AdvancePhaseRequest {
    request_id: string
    expected_phase: int
    next_phase: int
    new_phase_admission_deadline: timestamp?
}
```

### Rule

Allowed only from:

```text
PENDING
```

Reject if current phase is already:

```text
CLAIMED
MATERIALIZING_CONTEXT
EXECUTING_PRE_STREAM
STREAMING
COMMITTING_SESSION
```

because base V5 does not hedge an active phase.

## Lease fencing

An expired owner can be replaced pre-stream, but the new owner receives a new `lease_id`. Any old process waking later must fail all conditional mutations with its stale lease.

## Invariants

- At most one live destination execution lease exists in base V5.
- Context-materialization fallbacks for the same destination stay under that same execution lease.
- Stateful generation completion transitions through `COMMITTING_SESSION`; platform `SUCCESS` occurs only after durable session commit succeeds.
- Terminal state is immutable except for deliberately idempotent repeats.
- Registry failure must fail closed for ownership acquisition; never execute without ownership.
