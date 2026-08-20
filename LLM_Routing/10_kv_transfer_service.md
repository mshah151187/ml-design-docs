# Component — KVTransferService

## Responsibility

Move already-materialized KV blocks from a pinned source to a destination import target. It is a bulk data plane, not a routing-policy component.

The concrete transport can be NIXL/RDMA/engine-native transfer/storage-backed mechanisms. The router architecture depends on the transfer contract, not a particular library.

## Preconditions

`ContextMaterializationCoordinator` has already:

```text
validated source/destination high-level compatibility
pinned source blocks
created/validated destination import capacity
```

## Input — `KVTransferRequest`

```text
KVTransferRequest {
    transfer_id: string
    request_id: string
    phase: int

    kv_compatibility_id: string
    token_context_digest: string

    source_replica_id: string
    source_replica_epoch: int
    source_endpoint: string
    source_pin_id: string
    pinned_block_manifest_ref: string

    destination_replica_id: string
    destination_replica_epoch: int
    destination_endpoint: string
    destination_import_handle: string

    expected_block_count: int
    expected_token_count: int
    expected_bytes: int

    deadline: timestamp?
}
```

## Output — `KVTransferResult`

```text
KVTransferResult {
    transfer_id: string
    status: SUCCESS | PARTIAL | FAILED

    transferred_blocks: int
    transferred_tokens: int
    transferred_bytes: int

    transfer_latency_ms: float
    effective_bandwidth_bytes_per_sec: float

    destination_import_handle: string

    failure_code: string?
    retryable: bool
}
```

## Invariants

- A transfer cannot silently cross incompatible `kv_compatibility_id`s.
- Source epoch/pin fencing prevents old/restarted source processes from serving stale manifests.
- Destination epoch/import handle prevents writing into a restarted destination generation.
- Partial transfer is never treated as complete context unless the coordinator explicitly recomputes/repairs the missing tail.
- Transport failure never destroys correctness because durable context can be recomputed.

## Observability

Emit transfer observations used later by the V5 latency/cost predictor:

```text
source/destination profile/topology
bytes
blocks/tokens
latency
bandwidth
success/failure
```
