# Component — RequestQueueSubsystem

## Responsibility

Own one logical FIFO candidate queue per replica and the authoritative router-side aggregate work state consumed by all Planners.

This component replaces Planner-local provisional allocation state.

## Input — `EnqueueCandidateRequest`

```text
EnqueueCandidateRequest {
    request_id: string
    phase: int

    execution_plan_id: string
    replica_execution_plan_ref: string

    destination_replica_id: string
    destination_replica_epoch: int

    candidate_rank: int

    estimated_prefill_tokens: int
    estimated_decode_tokens: int

    phase_admission_deadline: timestamp?
    enqueued_at: timestamp
}
```

## Stored queue entry — `ReplicaQueueEntry`

```text
ReplicaQueueEntry {
    queue_entry_id: string

    request_id: string
    phase: int

    execution_plan_id: string
    replica_execution_plan_ref: string

    destination_replica_id: string
    destination_replica_epoch: int
    candidate_rank: int

    estimated_prefill_tokens: int
    estimated_decode_tokens: int

    phase_admission_deadline: timestamp?
    enqueue_sequence: int
    enqueued_at: timestamp
}
```

The canonical request payload/token artifact is not duplicated K times; entries contain references.

## Logical queue states

```text
READY
    waiting in FIFO and immediately eligible to compete

PRE_ADMISSION
    popped/claimed by an Executioner but model-server execution/context admission
    has not yet crossed into backend ownership

PARKED
    sibling candidate blocked while another destination owns a live request lease

TERMINAL/STALE
    lazily discarded
```

## Router load accounting

Only `READY + PRE_ADMISSION` count toward active router scheduling load.

```text
RouterOutstandingState {
    ready_requests: int
    ready_prefill_tokens: int
    ready_decode_tokens: int

    pre_admission_requests: int
    pre_admission_prefill_tokens: int
    pre_admission_decode_tokens: int

    parked_requests: int              # observability only
    parked_prefill_tokens: int        # observability only
    parked_decode_tokens: int         # observability only

    state_version: int
    observed_at: timestamp
}
```

Scorer-facing formula:

```text
RouterOutstandingPrefill = READY.prefill + PRE_ADMISSION.prefill
RouterOutstandingDecode  = READY.decode  + PRE_ADMISSION.decode
```

## API results

### Enqueue

```text
EnqueueCandidateResult {
    status: ENQUEUED | DUPLICATE | REPLICA_QUEUE_UNAVAILABLE
    queue_entry_id: string?
    state_version: int?
}
```

### State transitions

Executioner/queue subsystem transitions logical accounting through APIs such as:

```text
mark_pre_admission(queue_entry_id)
mark_parked(queue_entry_id)
requeue_parked(queue_entry_id)
mark_backend_owned(queue_entry_id)
mark_terminal_or_stale(queue_entry_id)
```

Each transition updates queue/logical counters atomically from the caller's perspective.

## Backend handoff

At successful model-server/context admission the router removes the request from `PRE_ADMISSION` accounting and backend telemetry becomes authoritative.

There can be a small telemetry visibility gap between admission ACK and the next backend snapshot. If this becomes materially harmful, add admission watermarks/fast admission events; do not blur the meaning of router ownership.

## Invariants

- All Planner instances observe one authoritative logical queue state stream/materialization.
- Physical queue records may contain stale entries; logical active counters must not blindly equal physical depth.
- PARKED copies are retained for retry but excluded from active load.
- FIFO is the V5 base queue policy.
