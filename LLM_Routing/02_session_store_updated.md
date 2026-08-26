# Component — SessionStore (KV-Tier Updated)

## 1. Why SessionStore Exists

Multi-turn inference needs one durable answer to:

> **What conversation state is committed and therefore safe to use for the next turn?**

KV cache cannot answer this.

A Decode worker may contain:

```text
system + user1 + assistant1 + partial assistant2
```

while the durable committed session is still:

```text
system + user1 + assistant1
```

A KV copy can also disappear, move tiers, be stale, or belong to another model revision.

Therefore:

> **SessionStore owns semantic conversation correctness. It never delegates that authority to KV residency.**

---

# 2. What Changes With an External KV Tier?

Very little in SessionStore itself.

That is a feature.

Earlier SessionStore already owned:

```text
SessionHead
ordered SessionEvents
turn lease / fencing
summary checkpoints
```

The KV-tier extension adds only an optional stable linkage to **logical context/artifact identity**, never a physical node/address.

Good:

```text
SessionHead {
    committed_version = 27
    latest_context_digest = H_context_27
}
```

Potentially useful hint:

```text
committed_model_artifact_hints[
    model_revision
] = target_context_digest
```

Bad:

```text
kv_node = "node-17"
kv_gpu_address = 0x...
kv_tier = CPU
```

Physical KV residency belongs in `KVDirectoryIndex`.

---

# 3. Why SessionStore and KV Tier Must Remain Separate

Consider:

```text
S123@27
```

The KV Tier may contain:

```text
Llama-70B KV for S123@27
Llama-8B KV for S123@27
old Llama-70B KV for S123@26
partial uncommitted S123@28 KV
```

Only SessionStore knows:

```text
27 is committed
28 is pending/uncommitted
```

Therefore the next request resolves:

```text
SessionStore
    -> committed logical version
    -> exact logical conversation

TargetModelContextMaterializer
    -> exact model-specific prefix

KVDirectoryIndex
    -> execution-state copies for that exact prefix
```

Not:

```text
SessionStore
    -> physical KV node
```

---

# 4. Data Model

## SessionHead

```text
SessionHead {
    session_id

    committed_version
    latest_sequence_number
    latest_segment_id

    active_turn_request_id?
    active_turn_lease_id?
    active_turn_lease_expires_at?

    latest_summary_checkpoint_ref?

    latest_logical_context_digest?

    created_at
    updated_at
}
```

### `latest_logical_context_digest`

This is optional metadata for stable identification/observability.

It identifies:

```text
the committed model-independent logical context
```

It does **not** itself identify target-model KV.

Different target models can materialize different KV from the same logical conversation.

---

# 5. Session Events Remain Model Independent

```text
SessionEvent {
    session_id
    segment_id
    sequence_number
    committed_version

    event_id
    event_type
    role?

    inline_content?
    content_ref?

    tool_name?
    tool_call_id?

    metadata
    request_id?
    created_at
}
```

Do not make target-model token IDs or KV block IDs canonical SessionEvents.

Why?

```text
model may change
tokenizer/template may change
summary/context policy may change
KV may be evicted
```

Conversation truth should survive all of those.

---

# 6. BeginTurn

Unchanged semantic purpose:

> Fence one active session mutation.

```text
BeginTurnRequest {
    session_id
    request_id
    expected_session_version?
    turn_lease_ttl_ms
    idempotency_key?
}
```

```text
BeginTurnResult {
    status
    base_committed_version?
    turn_lease_id?
    turn_lease_expires_at?
    session_head_version?
}
```

---

# 7. LoadContext

```text
LoadContextRequest {
    session_id
    committed_version
    summary_checkpoint_ref?
    include_events_after_sequence?
}
```

```text
LoadedSessionContext {
    session_id
    committed_version

    summary_checkpoint_ref?
    summary_payload_ref?

    events | events_ref

    first_sequence_number
    last_sequence_number

    logical_context_digest?

    loaded_at
}
```

The ContextInitializer creates the immutable logical request snapshot from this output.

---

# 8. CommitTurn

```text
CommitTurnRequest {
    session_id
    request_id
    turn_lease_id
    expected_base_version

    user_event
    assistant_event
    additional_events[]

    response_digest
    resulting_logical_context_digest?
}
```

The semantic commit is:

```text
N -> N+1
```

KV publication/checkpoint is **not** required to be part of the atomic semantic transaction.

Why?

A request should not fail semantically merely because a reusable KV checkpoint could not be written.

Recommended ordering:

```text
model generation complete
        |
        v
Commit Session N -> N+1
        |
        v
publish/checkpoint reusable KV asynchronously or best-effort
```

A stricter deployment may checkpoint before acknowledging for performance reasons, but KV persistence must never redefine semantic correctness.

---

# 9. Interaction With KVDirectoryIndex

SessionStore does not call `KVDirectoryIndex` on its own.

The request path is:

```text
ContextInitializer
    |
    | SessionStore.LoadContext
    v
LogicalContextSnapshot
    |
    v
TargetModelContextMaterializer
    |
    | exact model revision + token prefix
    v
LogicalKVIdentity / canonical block keys
    |
    v
Planner -> KVDirectoryIndex
```

This ordering prevents the system from looking up KV before the target model and exact target token prefix are known.

---

# 10. Multi-Model Sessions

Suppose a session previously ran on Llama-8B but Phase 2 now selects Llama-70B.

SessionStore still returns:

```text
same committed logical conversation
```

The target-model materializer now creates:

```text
Llama-70B exact token/context artifact
```

Then KVDirectoryIndex may find:

```text
Llama-8B KV only
```

That is not a direct cache hit.

Unless a separately qualified semantic mapper exists:

```text
re-Prefill Llama-70B
```

This is another reason SessionStore must not store one canonical KV pointer.

---

# 11. Failure Semantics

## KV Tier unavailable

```text
SessionStore healthy
TargetModelContextArtifact available/rebuildable
```

Result:

```text
performance degradation
-> fresh Prefill
```

not correctness failure.

## KVDirectoryIndex unavailable

Result:

```text
skip cache optimization
-> recompute / load-aware route
```

## SessionStore unavailable

For a stateful turn:

```text
cannot safely determine committed context/version
```

Result:

```text
do not silently continue from a random KV copy
```

This is a correctness failure.

---

# 12. Invariants

1. SessionStore is model independent.
2. SessionStore never treats physical KV residency as authoritative.
3. Only committed session versions may seed a new turn.
4. Pending/uncommitted worker KV does not advance session truth.
5. KV loss is recoverable by reconstructing target context and re-Prefilling.
6. A target-model change does not mutate historical session semantics.
7. Turn lease/version fencing remains authoritative even if KV state is newer.
8. KV checkpoint success is an optimization, not a prerequisite for semantic commit.

---

# 13. Interview Answer

> **I keep SessionStore deliberately boring even after introducing a distributed KV tier. It owns the committed logical conversation, version and turn fencing; it does not store a physical KV-node pointer. For a new turn I first load the committed semantic context, materialize the exact target-model token prefix, and only then ask the KVDirectoryIndex whether compatible execution state exists. That separation means KV can migrate between GPU, CPU and NVMe, disappear entirely, or exist for multiple model revisions without corrupting session correctness. If KV is lost, I re-Prefill. If SessionStore is lost, I cannot safely advance the session.**
