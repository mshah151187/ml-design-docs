# Component — ContextInitializer

## Responsibility

Convert either a stateless request or a session-driven request into one immutable **model-independent logical context snapshot**.

For stateful requests it acquires the session turn lease and loads the committed conversation. It does not choose a model and does not perform target-model tokenization.

## Input — `NormalizedRequest`

All fields are defined in `01_request_normalizer.md`.

The fields used directly here are:

```text
request_id
request_mode
session_id
expected_session_version
inline_messages
new_turn
attachments
tool_definitions
idempotency_key
received_at
```

## Session-mode processing

```text
NormalizedRequest(session_id=S123, new_turn=...)
    -> SessionStore.BeginTurn
    -> SessionStore.LoadContext(base_version=N)
    -> append new turn only to the immutable request snapshot
       (do not commit it yet)
```

The durable committed session remains at version `N` while inference is running.

## Stateless processing

The supplied inline messages/prompt are copied or referenced into an immutable logical snapshot. No SessionStore mutation is required.

## Output — `LogicalContextSnapshot`

```text
LogicalContextSnapshot {
    context_snapshot_id: string
    request_id: string
    request_mode: STATELESS | SESSION

    session_id: string?
    base_session_version: int?
    turn_lease_id: string?
    turn_lease_expires_at: timestamp?

    committed_context_ref: string?
    logical_events: list<LogicalEvent> | logical_events_ref
    new_turn_event: LogicalEvent?

    system_instructions_ref: string?
    tool_definitions_ref: string?
    attachment_refs: list<string>
    summary_checkpoint_ref: string?

    context_policy_id: string
    logical_context_digest: string

    created_at: timestamp
    expires_at: timestamp?
}
```

### Field semantics

| Field | Meaning |
|---|---|
| `context_snapshot_id` | Immutable identity used by all downstream phases. |
| `base_session_version` | Committed session version from which this request was constructed. |
| `turn_lease_id` | Required later to commit/abort this stateful turn. |
| `logical_events` | Model-independent ordered conversation events, potentially as a reference. |
| `new_turn_event` | Current uncommitted user turn. |
| `logical_context_digest` | Digest of the logical snapshot for idempotency/debugging; not the target-model token digest. |
| `context_policy_id` | Compaction/system-context policy used to build the snapshot. |

## Invariants

- The snapshot is immutable.
- The stateful new turn is included in the inference context but is not committed to the durable session yet.
- No target-model token IDs appear here.
- A given `context_snapshot_id` always reconstructs the same logical conversation input.

## Failure semantics

If a stateful snapshot cannot be created after `BeginTurn`, abort/release the acquired turn lease.

## Next component

`QualityModelSelector` consumes the logical snapshot plus normalized routing requirements.
