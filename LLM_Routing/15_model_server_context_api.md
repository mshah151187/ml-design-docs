# Component Contract — ModelServerContextAPI

## Responsibility

Expose the model-server primitives required by V5 context materialization and execution without leaking engine-specific APIs into the Router design.

The model server remains final authority for actual memory/admission feasibility.

## API 1 — `GetContextCapacity`

### Input

```text
GetContextCapacityRequest {
    request_id: string
    replica_id: string
    replica_epoch: int

    required_context_tokens: int
    required_kv_blocks: int?
    kv_compatibility_id: string
}
```

### Output

```text
GetContextCapacityResult {
    status: AVAILABLE | PRESSURED | INSUFFICIENT | STALE_REPLICA

    available_kv_blocks: int
    kv_cache_utilization: float
    admission_state: string

    observed_at: timestamp
}
```

This is a just-in-time check, not a long-lived reservation.

## API 2 — `PinKVPrefix`

### Input

```text
PinKVPrefixRequest {
    request_id: string

    replica_id: string
    expected_replica_epoch: int

    kv_compatibility_id: string
    token_context_digest: string

    requested_block_manifest_ref: string
    pin_ttl_ms: int
}
```

### Output

```text
PinKVPrefixResult {
    status: PINNED | BLOCKS_MISSING | STALE_EPOCH | INCOMPATIBLE | REJECTED

    pin_id: string?
    pinned_block_manifest_ref: string?

    pinned_blocks: int
    pinned_tokens: int
    pinned_bytes: int

    pin_expires_at: timestamp?
}
```

A successful pin prevents eviction of the pinned block set for the bounded pin lifetime.

## API 3 — `UnpinKVPrefix`

```text
UnpinKVPrefixRequest {
    replica_id
    replica_epoch
    pin_id
}
```

Output is idempotent success/not-found status.

## API 4 — `CreateKVImport`

### Input

```text
CreateKVImportRequest {
    request_id: string
    replica_id: string
    replica_epoch: int

    kv_compatibility_id: string
    token_context_digest: string

    expected_blocks: int
    expected_tokens: int
    expected_bytes: int
}
```

### Output

```text
CreateKVImportResult {
    status: READY | INSUFFICIENT_CAPACITY | THROTTLED | STALE_REPLICA
    destination_import_handle: string?
}
```

No import handle is created before the destination owns the execution lease.

## API 5 — `FinalizeKVImport`

### Input

```text
FinalizeKVImportRequest {
    destination_import_handle: string
    transferred_blocks: int
    transferred_tokens: int
    token_context_digest: string
}
```

### Output

```text
FinalizeKVImportResult {
    status: READY | PARTIAL_REQUIRES_RECOMPUTE | INVALID | FAILED
    destination_context_handle: string?
    valid_prefix_tokens: int
}
```

## API 6 — `RecomputeContext`

### Input

```text
RecomputeContextRequest {
    request_id: string
    replica_id: string
    replica_epoch: int

    token_ids_ref: string
    token_context_digest: string

    reusable_prefix_context_handle: string?
}
```

### Output

```text
RecomputeContextResult {
    status: READY | THROTTLED | INSUFFICIENT_CAPACITY | FAILED
    destination_context_handle: string?
    reused_tokens: int
    recomputed_tokens: int
    prefill_latency_ms: float
}
```

## API 7 — `ExecuteGeneration`

### Input

```text
ExecuteGenerationRequest {
    request_id: string
    replica_id: string
    replica_epoch: int

    destination_context_handle: string
    sampling_params_ref: string
    max_output_tokens: int
}
```

### Output/events

```text
accepted
first_token / streaming_started
token chunks
completed
retryable_pre_stream_failure
terminal_failure
```

## Telemetry emitted

The model server continuously emits:

```text
EngineWorkState
EngineCapacityState
performance observations
KV allocation/eviction/move events
context materialization timings
admission/rejection events
```

## Invariants

- Source pinning is bounded and explicit.
- Destination capacity is checked JIT and never assumed from an old Planner snapshot.
- Engine-specific details remain behind this contract.
