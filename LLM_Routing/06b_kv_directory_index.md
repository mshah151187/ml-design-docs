# Component — KVDirectoryIndex

## 1. Why This Component Exists

The earlier `PreciseKVIndex` answered:

> Which **model replica** contains this exact reusable prefix?

Once we add an external KV tier, that question is too narrow.

The same logical KV may exist simultaneously on:

```text
Decode worker GPU
Prefill worker GPU
shared KV GPU node
pinned CPU DRAM
distributed DRAM node
local NVMe
remote persistent/offload tier
```

Therefore the index evolves into:

> **What materialized copies of this exact logical KV exist anywhere in the serving system, in what representation, and with what freshness/confidence?**

The component is renamed:

```text
PreciseKVIndex
    ->
KVDirectoryIndex
```

---

# 2. What It Owns — and Does Not Own

Owns:

```text
prefix/block identity -> residency metadata
residency freshness/version
physical representation metadata
tier/location metadata
supported exact-transform metadata/hints
```

Does not own:

```text
KV bytes
request routing policy
source pinning
destination allocation
tier eviction policy
SessionStore semantic truth
```

So:

```text
KVDirectoryIndex = metadata/control plane
KVTier           = storage/data plane
Planner          = decision policy consumer
```

---

# 3. Logical Identity vs Physical Residency

This separation is the heart of the design.

## LogicalKVIdentity

```text
LogicalKVIdentity {
    model_id
    model_revision
    adapter_id?

    tokenizer_revision
    chat_template_revision
    position_encoding_id
    attention_semantics_id

    target_context_digest
    hash_schema_version
}
```

This answers:

> What mathematical target-model execution state does this prefix represent?

## PhysicalKVRepresentation

```text
PhysicalKVRepresentation {
    representation_id

    kv_dtype
    quantization_scheme_id?

    tensor_layout
    block_size

    tp_layout_id
    pp_layout_id

    attention_backend_layout_id
    engine_cache_format_version
}
```

This answers:

> How are those mathematically correct values physically stored?

One `LogicalKVIdentity` may have many physical residencies.

---

# 4. Residency Descriptor

```text
KVResidency {
    residency_id

    logical_kv_identity
    physical_representation

    location_type:
        MODEL_WORKER
        KV_TIER_NODE

    tier:
        WORKER_GPU
        SHARED_GPU
        CPU_DRAM
        NVME
        REMOTE_STORAGE

    node_id
    replica_id?
    replica_epoch?

    storage_object_ref?
    block_manifest_ref

    prefix_blocks
    prefix_tokens
    bytes

    health
    confidence

    locality_domain?
    rack?
    zone?

    observed_at
    sequence
    residency_version
}
```

---

# 5. Why Session ID Is Not the Primary Cache Key

A session may reuse the same KV, but prefix caching also matters for stateless traffic.

Examples:

```text
shared 32K system prompt
shared RAG document prefix
few-shot prompt prefix
multi-turn session prefix
```

Therefore the fundamental index is based on:

```text
exact target-model prefix identity
```

not:

```text
session_id
```

A session is merely one producer/consumer of such prefixes.

---

# 6. Prefix Block Identity

The target-model materializer creates canonical prefix block identities from exact target tokens and a versioned hash scheme.

Conceptually:

```text
B0 = H(namespace, tokens[0:block])
B1 = H(B0, tokens[block:2block])
B2 = H(B1, ...)
```

Chaining ensures prefix order/continuity.

A match at block 17 does not compensate for a missing block 8.

Reusable prefix length is the longest continuous block chain from the beginning.

---

# 7. Mutation Events

The index subscribes to residency lifecycle events.

```text
KVDirectoryEvent {
    producer_id
    producer_epoch
    sequence

    event_type:
        RESIDENCY_ADDED
        RESIDENCY_REMOVED
        RESIDENCY_MOVED
        RESIDENCY_REPRESENTATION_CHANGED
        CLEAR_PRODUCER
        SNAPSHOT_MARKER

    residency
    observed_at
}
```

Producers may include:

```text
model servers
KV tier managers
replication workers
SSD/offload managers
```

---

# 8. Ordering and Confidence

Per producer:

```text
(producer_id, producer_epoch, sequence)
```

is monotonic.

If the index observes a gap:

```text
101
102
105
```

then affected residency becomes:

```text
UNCERTAIN
```

until replay or snapshot reconciliation.

Request-path rule:

> **UNCERTAIN residency receives no precise cache credit.**

The optimizer can degrade to another copy or recompute.

---

# 9. Request API — FindContextResidency

```text
FindContextResidencyRequest {
    request_id

    logical_kv_identity

    target_physical_representations[]
    canonical_prefix_block_keys_ref

    candidate_compute_nodes[]?
    max_results
}
```

Output:

```text
ContextResidencyResult {
    request_id

    longest_prefix_matches[]
    exact_transform_options[]

    index_version
    observed_at
}
```

---

# 10. Prefix Match

```text
PrefixResidencyMatch {
    residency_id

    matched_blocks
    matched_tokens
    matched_bytes

    tier
    node_id
    replica_id?
    replica_epoch?

    physical_representation

    transfer_bytes

    topology_hint

    confidence
    block_manifest_ref
}
```

---

# 11. Exact Transform Option

If semantic identity matches but physical representation differs:

```text
ExactTransformOption {
    source_residency_id
    target_representation_id

    transform_id

    transform_type:
        TP_RESHARD
        PP_REMAP
        HND_NHD_TRANSPOSE
        REBLOCK
        PACKING_CONVERSION

    estimated_transform_bytes
    estimated_transform_cost
    supported
}
```

The directory does not execute the transform.

It reports:

```text
a trusted transform exists
```

The ContextMaterializationCoordinator executes it through the appropriate runtime/service.

---

# 12. Cross-Model Semantic Mapping

Do not silently fold semantic mappers into exact transforms.

If future production supports a qualified cross-model mapper:

```text
SemanticMapOption {
    source_logical_kv_identity
    target_logical_kv_identity

    mapper_id
    mapper_version

    quality_profile_id
    supported_context_range
    uncertainty_class
}
```

This is logically separate because:

```text
exact transform
    -> no intended model-quality loss

semantic mapper
    -> empirical quality risk
```

Base V5 may simply return no semantic-map options.

---

# 13. Sharding

The cardinality is huge.

Do not replicate the directory into every Planner.

Shard primarily by:

```text
model semantic namespace
+
shallow prefix branch
```

so a normal exact-prefix lookup reaches a bounded set of shards.

The directory itself can maintain replicated metadata shards for availability, but the request path should avoid fanout to every KV tier/storage node.

---

# 14. Worker-Local and Tier Residency in One Result

Example:

```text
128K exact prefix

D7 worker GPU:
    90K continuous hit
    DIRECT

KVNode3 CPU:
    128K
    exact semantics
    TP2 representation
    target D requires TP4
    EXACT_TRANSFORM

NVMe Node8:
    128K
    DIRECT target representation
```

Planner can compare:

```text
reuse 90K locally + Prefill 38K
vs
restore CPU 128K + TP transform
vs
restore NVMe 128K
vs
fresh Prefill
```

This is why location must not be encoded as a binary hit/miss.

---

# 15. Session Interaction

The SessionStore does not query this index directly.

Stateful path:

```text
SessionStore
    -> committed logical context

TargetModelContextMaterializer
    -> exact target-model token context

Planner
    -> KVDirectoryIndex
```

The index may reveal a residency that originated from an earlier turn of that same session, but it treats it just like any exact compatible prefix.

---

# 16. Race Boundary

Directory metadata is not a reservation.

For worker-local source:

```text
directory says R1 has blocks
        |
        v
later ContextMaterializationCoordinator
        |
        | PinKVPrefix + epoch/version validation
        v
stable transfer source
```

For KVTier:

```text
directory says object exists
        |
        v
later Get/AcquireReadHandle
        |
        | object generation/version validation
        v
stable restore source
```

This preserves the distinction:

```text
decision-time observation
    !=
execution-time guarantee
```

---

# 17. Failure Semantics

Directory unavailable:

```text
skip exact locality optimization
-> recompute / conservative route
```

Directory stale:

```text
JIT validation fails
-> next materialization fallback
```

One residency missing:

```text
try another copy
```

All residencies missing:

```text
re-Prefill
```

No session correctness is lost.

---

# 18. Design Principles

1. Index metadata, not KV bytes.
2. Index exact target-model prefix identity, not raw text.
3. Session ID is not the universal cache key.
4. Separate semantic identity from physical representation.
5. One logical KV may have many physical copies.
6. Worker-local and KVTier residencies share one discovery contract.
7. Exact-transform existence is metadata; transform execution lives elsewhere.
8. Semantic mapping is explicitly different from exact transformation.
9. Event gaps degrade confidence rather than inventing locality.
10. Lookup failure degrades optimization, not inference correctness.
11. Residency metadata is never a pin/reservation.
12. JIT validation closes the race between planning and execution.

---

# 19. Interview Answer

> **I evolve the old PreciseKVIndex into a KVDirectoryIndex once KV can live outside model replicas. Its key abstraction is a logical KV identity—exact model revision plus exact target-token prefix—and each logical KV can have multiple physical residencies: worker GPU, shared GPU, CPU DRAM or NVMe, potentially in different TP/layout representations. The directory stores metadata and transform availability, not bytes. The Planner uses it to enumerate materialization paths, while the ContextMaterializationCoordinator later performs a JIT source/version validation before restoring or transferring. That keeps cache discovery fast without making stale metadata a correctness dependency.**
