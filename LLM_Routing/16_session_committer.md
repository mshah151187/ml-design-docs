# Component — SessionCommitter

## Responsibility

After successful model generation, atomically advance durable session state by committing the pending user turn and assistant result. For a stateful request, this commit is part of the platform success path.

For stateless requests this component is a no-op. For stateful requests the RequestExecutionRegistry should be in `COMMITTING_SESSION` while this operation runs.

## Input — `SessionCommitRequest`

```text
SessionCommitRequest {
    request_id: string

    session_id: string
    base_session_version: int
    turn_lease_id: string

    context_snapshot_id: string

    user_turn_event: SessionEvent
    assistant_response_ref: string
    assistant_response_digest: string

    tool_events: list<SessionEvent>
    metadata_events: list<SessionEvent>

    execution_phase: int
    model_id: string
    model_revision: string

    completed_at: timestamp
}
```

## Processing

1. Resolve `assistant_response_ref` into a durable assistant event or durable content reference.
2. Verify request still owns the session turn lease based on `base_session_version`.
3. Append user + tool + assistant events in deterministic order.
4. Atomically advance `SessionHead.committed_version` from `N` to `N+1`.
5. Clear active turn lease.

## Output — `SessionCommitResult`

```text
SessionCommitResult {
    status: COMMITTED | ALREADY_COMMITTED | STALE_VERSION | NOT_OWNER | SESSION_NOT_FOUND

    session_id: string
    request_id: string

    previous_version: int
    new_version: int?

    first_committed_sequence: int?
    last_committed_sequence: int?
}
```

## Streaming semantics

Base V5 commits the assistant turn only after successful full completion. Client-visible partial output does not automatically become committed conversation state.

If the product wants partial-turn persistence, that is an explicit session semantic extension and must not be inferred from streaming transport alone.

## Failure semantics

If inference fails before a successful result:

```text
SessionStore.AbortTurn
```

or allow the turn lease to expire according to policy.

A successful model response that cannot be committed because the session version/lease is stale must be surfaced as a session consistency failure rather than silently appended to the wrong conversation branch.

## Completion ordering

```text
model generation complete
    -> RequestExecutionRegistry.COMMITTING_SESSION
    -> SessionCommitter.CommitTurn
    -> RequestExecutionRegistry.SUCCESS
```

The client may already have received streaming tokens, but the platform's terminal successful completion for a stateful turn is not recorded until the session commit succeeds.
