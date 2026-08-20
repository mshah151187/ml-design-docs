# Component — PreciseKVIndex

## Responsibility

Track high-cardinality exact KV-prefix residency and expose two request-path facts:

1. **How much of this exact target-model prefix is already reusable on each candidate destination?**
2. **Which other compatible replicas can act as transfer sources for the missing prefix?**

It does not choose the destination and does not pin/move KV itself.

## Namespace identity

KV identity is scoped by an explicit compatibility namespace rather than blindly by deployment ID:

```text
KVNamespace {
    model_id
    model_revision
    kv_compatibility_id
    adapter_id?
    hash_schema_version
}
```

The target-model token context/digest must also match the actual requested prefix.

## Mutation input — `KVEvent`

```text
KVEvent {
    namespace: KVNamespace

    replica_id: string
    replica_epoch: int
    sequence: int

    event_type: BLOCK_STORED | BLOCK_REMOVED | BLOCK_MOVED | CLEAR_REPLICA

    engine_block_key: string?
    request_block_key: bytes/string?

    block_token_start: int?
    block_token_count: int?
    block_bytes: int?

    cache_tier: GPU | CPU | DISK | SHARED | NONE

    observed_at: timestamp
}
```

## Ordering / confidence

Per replica, the index tracks:

```text
(replica_id, replica_epoch, sequence)
```

A sequence gap marks precise state `UNCERTAIN` until replay/snapshot reconciliation. UNCERTAIN replicas receive no exact cache credit.

## Request-path API — `FindContextResidency`

### Input

```text
FindContextResidencyRequest {
    request_id: string

    model_id: string
    model_revision: string

    token_context_digest: string
    token_ids_ref: string

    candidate_namespaces: list<CandidateKVNamespace>
}
```

Each candidate namespace:

```text
CandidateKVNamespace {
    kv_compatibility_id: string
    candidate_replica_ids: list<string>
}
```

The KV index/client derives namespace-specific canonical chained prefix block keys from exact target-model token IDs and the compatibility profile's block/hash schema.

### Output — `ContextResidencyResult`

```text
ContextResidencyResult {
    request_id: string
    token_context_digest: string

    destination_matches: list<PrefixMatch>
    transfer_sources: list<KVTransferSource>

    index_version: string
    observed_at: timestamp
}
```

### `PrefixMatch`

```text
PrefixMatch {
    replica_id: string
    replica_epoch: int
    kv_compatibility_id: string

    matched_blocks: int
    matched_tokens: int
    matched_bytes: int

    best_cache_tier: GPU | CPU | DISK | SHARED | NONE
    confidence: TRUSTED | UNCERTAIN

    block_manifest_ref: string?
}
```

### `KVTransferSource`

```text
KVTransferSource {
    source_replica_id: string
    source_replica_epoch: int
    source_deployment_id: string?

    kv_compatibility_id: string

    transferable_prefix_blocks: int
    transferable_prefix_tokens: int
    estimated_transfer_bytes: int
    source_tier: GPU | CPU | SHARED

    block_manifest_ref: string

    confidence: TRUSTED | UNCERTAIN
    locality_version: string
    observed_at: timestamp
}
```

## Sharding

High-cardinality precise KV state is not replicated into every Planner.

V5 retains model/namespace + shallow-prefix-branch sharding so a normal prefix lookup reaches one KV shard rather than scatter-gathering across all replicas or all individual block shards.

## Transfer race boundary

`block_manifest_ref` is **not a pin**. It is a decision-time statement of observed residency.

Later, after a destination wins execution ownership, `ContextMaterializationCoordinator` asks the source model server to `PinKVPrefix` using:

```text
source_replica_epoch
kv_compatibility_id
token_context_digest
block_manifest_ref
```

Only that source-side pin converts observed locality into a temporarily stable transfer source.

## Invariants

- Exact KV locality is an optimization; lookup failure falls back to recomputation/load-aware scheduling rather than inference unavailability.
- Old replica epochs never receive credit.
- Prefix continuity matters: a later matching block does not repair an earlier missing block in the reusable prefix.
- Index never performs request routing or data movement.
