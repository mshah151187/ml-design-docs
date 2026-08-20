# Component — SessionStore

## Responsibility

Own the durable, model-independent conversation truth and serialize mutations to one session. It is correctness state.

It does **not** store canonical target-model token IDs as its source of truth and it does not own replica KV cache.

## Data model

### `SessionHead`

```text
SessionHead {
    session_id: string

    committed_version: int
    latest_sequence_number: int
    latest_segment_id: int

    active_turn_request_id: string?
    active_turn_lease_id: string?
    active_turn_lease_expires_at: timestamp?

    latest_summary_checkpoint_ref: string?

    created_at: timestamp
    updated_at: timestamp
}
```

Field meanings:

| Field | Meaning |
|---|---|
| `committed_version` | Latest fully committed conversation version. |
| `latest_sequence_number` | Last committed event sequence. |
| `latest_segment_id` | Current physical partition/segment for bounded partition size. |
| `active_turn_request_id` | Request currently allowed to mutate this session. |
| `active_turn_lease_id` | Fencing token for that pending turn. |
| `active_turn_lease_expires_at` | Recovery boundary if the owner dies. |
| `latest_summary_checkpoint_ref` | Optional durable compaction checkpoint. |

### `SessionEvent`

```text
SessionEvent {
    session_id: string
    segment_id: int
    sequence_number: int
    committed_version: int

    event_id: string
    event_type: SYSTEM | USER | ASSISTANT | TOOL_CALL | TOOL_RESULT | SUMMARY | METADATA
    role: string?

    inline_content: bytes/string?
    content_ref: string?

    tool_name: string?
    tool_call_id: string?

    metadata: map
    request_id: string?
    created_at: timestamp
}
```

Large payloads may live in blob/object storage and be referenced through `content_ref`.

## Storage shape

The logical access pattern is:

```text
small strongly-fenced SessionHead
+
append-oriented ordered SessionEvents
```

A Cassandra/Scylla/DynamoDB-style partitioned store is a natural implementation:

```text
partition by (session_id, segment_id)
order by sequence_number
```

Segments prevent one unbounded session partition.

The architecture requires an atomic conditional update on `SessionHead`; it does not depend on one vendor-specific API.

## API 1 — `BeginTurn`

### Input

```text
BeginTurnRequest {
    session_id: string
    request_id: string
    expected_session_version: int?
    turn_lease_ttl_ms: int
    idempotency_key: string?
}
```

### Output

```text
BeginTurnResult {
    status: ACQUIRED | ALREADY_OWNED | VERSION_MISMATCH | BUSY | SESSION_NOT_FOUND

    session_id: string
    base_committed_version: int?

    turn_lease_id: string?
    turn_lease_expires_at: timestamp?

    session_head_version: int?
}
```

### Invariant

Only one active mutation lease exists for one session in V5.

The lease TTL should cover the request deadline plus a safety margin. A production implementation may also expose `RenewTurnLease` for unusually long operations; any renewal must be conditional on the same `request_id + turn_lease_id`.

## API 2 — `LoadContext`

### Input

```text
LoadContextRequest {
    session_id: string
    committed_version: int
    summary_checkpoint_ref: string?
    include_events_after_sequence: int?
}
```

### Output

```text
LoadedSessionContext {
    session_id: string
    committed_version: int

    summary_checkpoint_ref: string?
    summary_payload_ref: string?

    events: list<SessionEvent> | events_ref

    first_sequence_number: int
    last_sequence_number: int

    loaded_at: timestamp
}
```

## API 3 — `CommitTurn`

### Input

```text
CommitTurnRequest {
    session_id: string
    request_id: string
    turn_lease_id: string
    expected_base_version: int

    user_event: SessionEvent
    assistant_event: SessionEvent
    additional_events: list<SessionEvent>

    response_digest: string
}
```

### Output

```text
CommitTurnResult {
    status: COMMITTED | ALREADY_COMMITTED | STALE_VERSION | NOT_OWNER | SESSION_NOT_FOUND
    new_committed_version: int?
    first_sequence_number: int?
    last_sequence_number: int?
}
```

Commit is an atomic logical transition from version `N` to `N+1` for this turn.

## API 4 — `AbortTurn`

### Input

```text
AbortTurnRequest {
    session_id: string
    request_id: string
    turn_lease_id: string
    reason: string
}
```

### Output

```text
AbortTurnResult {
    status: ABORTED | ALREADY_TERMINAL | NOT_OWNER | SESSION_NOT_FOUND
}
```

## Failure semantics

- KV loss never affects session correctness.
- Target-model token artifact loss never affects session correctness.
- SessionStore unavailability prevents safe stateful progression; the platform should not silently invent conversation state.
- A stale turn lease cannot commit after a newer owner has taken over.

## Next component

`ContextInitializer` uses `BeginTurn` and `LoadContext` for session-driven requests. `SessionCommitter` uses `CommitTurn`/`AbortTurn` after execution.
