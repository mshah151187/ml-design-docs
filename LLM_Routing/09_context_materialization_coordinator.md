# Component — ContextMaterializationCoordinator

## Responsibility

After a destination ReplicaExecutioner has won the request execution lease, prepare the exact target-model context on that destination using the ordered materialization fallback chain selected by `ReplicaScheduler`.

It owns context-materialization orchestration, not destination selection.

## Critical timing rule

**No destination KV memory is reserved while the request waits in the replica queue.**

Only after the destination wins `TryClaim` does the coordinator perform a just-in-time destination capacity check.

Source KV may be pinned briefly for a transfer attempt so the transfer observes a stable block set.

## Input — `ContextMaterializationRequest`

```text
ContextMaterializationRequest {
    request_id: string
    phase: int

    execution_plan_id: string

    owner_replica_id: string
    owner_replica_epoch: int
    execution_lease_id: string

    target_context_artifact: TargetModelContextArtifact
    materialization_chain: ContextMaterializationChain

    request_deadline: timestamp?
}
```

The owner replica must match the destination in the execution plan.

## State transition

Before work begins:

```text
CLAIMED -> MATERIALIZING_CONTEXT
```

The execution lease remains owned by the same destination during all materialization fallbacks.

## Plan 1 — `LOCAL_KV_REUSE`

1. Revalidate destination epoch and KV compatibility.
2. Query/verify that expected prefix blocks still exist.
3. JIT check destination capacity for the remaining context/tail.
4. Reuse the valid local prefix.
5. Recompute any uncached tail required to reach the exact target context.
6. Return a destination context handle.

A stale/missing local prefix does not release the execution lease immediately; advance to the next materialization plan.

## Plan 2 — `TRANSFER_KV`

1. JIT validate destination context capacity.
2. Validate source epoch, exact context digest/prefix, and `kv_compatibility_id`.
3. Ask source model server to `PinKVPrefix` for a short TTL.
4. Receive a pinned block manifest/version.
5. Create destination import context/handles.
6. Invoke `KVTransferService` for bulk movement.
7. Destination validates/registers imported KV.
8. Recompute any remaining uncached tail.
9. Always unpin source blocks in `finally`-style cleanup.

### Source race protection

The transfer plan is valid only if the source pin succeeds against the expected:

```text
source_replica_epoch
kv_compatibility_id
token_context_digest
prefix/block manifest version
```

If blocks were evicted before pinning, transfer fails cleanly and the next plan runs.

## Plan 3 — `RECOMPUTE_FROM_CONTEXT`

1. JIT validate destination capacity/admission.
2. Fetch exact token IDs through `TargetModelContextArtifact.token_ids_ref`.
3. Ask destination model server to prefill/recompute the target context.
4. Return the resulting destination context handle.

This is the correctness fallback when cache reuse/transfer fails.

## Output — `ContextMaterializationResult`

```text
ContextMaterializationResult {
    status: SUCCESS | ALL_PLANS_FAILED | STALE_EXECUTION_LEASE | DESTINATION_GONE

    request_id: string
    phase: int
    execution_plan_id: string

    destination_replica_id: string
    destination_replica_epoch: int

    mode_used: LOCAL_KV_REUSE | TRANSFER_KV | RECOMPUTE_FROM_CONTEXT | NONE
    materialization_plan_index: int?

    source_replica_id: string?
    source_replica_epoch: int?

    materialized_prefix_tokens: int
    locally_reused_tokens: int
    transferred_tokens: int
    recomputed_tokens: int
    transfer_bytes: int

    destination_context_handle: string?

    materialization_latency_ms: float
    attempts: list<ContextMaterializationAttempt>
}
```

### `ContextMaterializationAttempt`

```text
ContextMaterializationAttempt {
    plan_index: int
    mode: ContextMode
    status: SUCCESS | FAILED
    failure_code: string?
    latency_ms: float
}
```

## Failure semantics

- Failure of one materialization mode advances to the next plan while retaining the destination execution lease.
- If every plan fails, the ReplicaExecutioner releases the request to `PENDING`; parked sibling destination candidates may compete.
- Materialization failure occurs pre-stream and is therefore compatible with transparent same-phase retry.

## Next component

On success, ReplicaExecutioner proceeds to model execution using `destination_context_handle`.
