# Component — KVTier

## 1. Why KVTier Exists

Without an external KV subsystem, expensive reusable KV is fate-shared with model workers.

```text
D7 has 128K session KV
```

If D7 becomes:

```text
overloaded
drained
restarted
scaled down
```

the system must either:

```text
keep routing to D7
transfer directly from D7
or recompute 128K
```

KVTier introduces another option:

```text
checkpoint / replicate reusable KV
into independently managed storage
```

so compute and state placement can evolve independently.

---

# 2. What KVTier Is

KVTier is a distributed storage/data-movement subsystem specialized for reconstructible inference state.

Conceptual hierarchy:

```text
HOT / FAST / EXPENSIVE / SMALLER

worker-local GPU
shared KV GPU
pinned CPU DRAM
distributed DRAM
local/remote NVMe

COLDER / CHEAPER / LARGER
```

The exact tiers are deployment-specific.

The architecture does not require every deployment to have all tiers.

---

# 3. What KVTier Is Not

It is not:

```text
SessionStore
```

It does not decide:

```text
which conversation version is committed
```

It is not:

```text
KVDirectoryIndex
```

It does not have to serve global prefix search on the request path.

It is not:

```text
Planner
```

It does not choose the best request route.

Its core job is:

> **Store, retrieve, replicate, promote, demote and delete KV objects efficiently across heterogeneous storage tiers.**

---

# 4. Why a Multi-Tier Hierarchy?

HBM is fastest but expensive and scarce.

CPU DRAM is:

```text
slower
much larger
cheaper
```

NVMe is:

```text
much larger again
slower again
```

Long-context KV has reuse distributions with a long tail.

Keeping every reusable prefix in GPU HBM would destroy serving capacity.

Therefore:

```text
hot working set
    -> GPU

warm reusable state
    -> DRAM

cold but expensive-to-recompute state
    -> NVMe / remote tier
```

This is analogous to a cache hierarchy, but the value function is special:

```text
value of cache object
≈ avoided Prefill cost
```

which grows strongly with reusable prefix length/model size.

---

# 5. Object Model

Treat a logical KV prefix as one or more immutable versioned storage objects.

```text
KVObject {
    object_id

    logical_kv_identity
    physical_representation

    prefix_start_token
    prefix_end_token

    block_manifest
    bytes

    object_generation
    checksum

    created_at
}
```

Immutability simplifies concurrency.

Instead of modifying:

```text
KV(S123)
```

in place each turn, produce:

```text
KV prefix version N
KV prefix version N+1
```

or append/reference immutable block objects.

---

# 6. Why Object/Block Immutability Matters

Consider concurrent readers while a cache is being extended.

Mutable object:

```text
reader sees partially updated 128K prefix
```

is dangerous.

Immutable/versioned objects allow:

```text
publish object only after complete write
```

so readers observe:

```text
old complete version
or
new complete version
```

never a partially materialized cache.

---

# 7. Write Path

Example Decode worker D7 finishes a turn.

```text
D7 GPU KV
    |
    | select checkpointable prefix
    v
KVTier.Put
    |
    +--> target DRAM node
    +--> optional replica
    +--> optional later NVMe offload
    |
    v
publish residency event
    |
    v
KVDirectoryIndex
```

Important ordering:

```text
write bytes completely
    ->
publish object/residency metadata
```

not the reverse.

---

# 8. Read Path

```text
ContextMaterializationCoordinator
    |
    | selected residency_id
    v
KVTier.AcquireRead
    |
    | validate object generation
    | choose topology-aware transfer path
    v
destination GPU / CPU staging
```

Then:

```text
optional exact transform
    ->
destination KV import
    ->
VALID
```

---

# 9. API Sketch

## Put

```text
PutKVRequest {
    object_id
    logical_kv_identity
    physical_representation

    source_memory_descriptors
    bytes

    desired_durability_class
    placement_hints?
}
```

```text
PutKVResult {
    status
    object_generation
    residencies[]
}
```

## AcquireRead

```text
AcquireKVReadRequest {
    object_id
    expected_generation?

    destination_topology?
    deadline?
}
```

```text
AcquireKVReadResult {
    status
    read_handle
    source_residency
    representation
    expires_at?
}
```

## Read / Transfer

```text
TransferKVObject {
    read_handle
    destination_descriptors
}
```

## ReleaseRead

```text
ReleaseKVRead(read_handle)
```

## Delete / Evict

Internal/admin/tiering lifecycle.

---

# 10. Tier Selection Is Not a Request-Side Hard Rule

Do not write:

```text
if GPU miss:
    CPU
else if CPU miss:
    SSD
```

because sometimes:

```text
SSD restore 400 tokens
```

is slower than:

```text
recompute 400 tokens
```

The Planner should receive estimated paths such as:

```text
CPU_RESTORE:
    60 ms

NVME_RESTORE:
    230 ms

RECOMPUTE:
    90 ms
```

and choose the SLO/cost-valid path.

The KVTier itself manages where objects reside; Planner decides whether a given residency is worth using for this request.

---

# 11. Promotion / Demotion Policy

This is mostly an asynchronous control-loop problem.

Signals:

```text
reuse frequency
reuse recency
object size
recompute cost
model size
prefix length
tenant priority
tier pressure
network topology
```

Example value metric:

```text
expected future avoided_prefill_cost
------------------------------------
bytes occupied in expensive tier
```

Hot:

```text
NVMe -> DRAM
```

Cold:

```text
GPU/DRAM -> NVMe
```

Pressure:

```text
evict lowest-value objects
```

This policy is not on the critical request routing path.

---

# 12. Replication

One copy is cheap in capacity but vulnerable to:

```text
node loss
rack loss
hotspot
```

Replication can improve:

```text
availability
read bandwidth
topological locality
```

but consumes storage.

Useful policy:

```text
hot/high-value prefix
    -> multiple DRAM/GPU-domain copies

cold prefix
    -> one NVMe copy

cheap-to-recompute prefix
    -> no external copy
```

No universal replication factor is required.

---

# 13. Stateless Traffic

A shared system prompt may be:

```text
64K tokens
```

requested by thousands of independent requests.

Once cached:

```text
LogicalKVIdentity(system prompt, model M)
    |
    +--> CPU DRAM copy A
    +--> CPU DRAM copy B
    +--> maybe shared GPU copy
```

Each stateless request can restore that prefix and compute only its unique suffix.

KVTier therefore provides value even without sessions.

---

# 14. Stateful Traffic

Turn N completes:

```text
SessionStore commits semantic version N
```

Reusable model-specific KV for that committed prefix can be checkpointed.

Next turn:

```text
SessionStore -> committed context
TargetModelContextArtifact -> exact target prefix
KVDirectoryIndex -> find compatible checkpoint
KVTier -> restore
```

If no checkpoint exists:

```text
re-Prefill
```

SessionStore never relies on a specific checkpoint existing.

---

# 15. Consistency Model

KV is reconstructible and immutable/versioned.

This permits weaker metadata consistency than SessionStore.

A reasonable contract:

```text
Put is atomic at object publication boundary.

Get returns:
    a complete published generation
    or NOT_FOUND.

Directory may temporarily lag actual object state.

JIT read validation detects stale directory entries.
```

We do not require globally linearizable cache residency metadata.

---

# 16. Garbage Collection

Objects need lifecycle cleanup.

Triggers:

```text
TTL
LRU/LFU/value policy
model revision retired
adapter retired
hash/schema version retired
tenant deletion
session retention policy
storage pressure
```

Deleting KV must never delete the underlying SessionStore semantic events.

---

# 17. Security / Tenant Isolation

KV can encode sensitive prompt state.

Namespace/access policy should include:

```text
tenant boundary
model/policy boundary
encryption domain if required
authorization on read
secure deletion / retention
```

Cross-tenant prefix reuse should only exist when explicitly policy-safe.

---

# 18. Failure Semantics

## Tier node fails

Directory eventually removes residency.

Other copy exists:

```text
use it
```

No copy:

```text
recompute
```

## Put fails

Request can still succeed.

Do not fail semantic session commit merely because cache checkpoint failed.

## Read fails mid-transfer

Discard partial destination import.

Try another residency or recompute.

## Directory says object exists but KVTier returns NOT_FOUND

Treat as stale metadata.

Fallback; repair directory asynchronously.

---

# 19. Observability

Track:

```text
hit rate by tier
bytes by tier
restore latency by tier
promotion/demotion bytes
eviction rate
object age
recompute avoided tokens
restore-vs-recompute wins/losses
directory stale-hit rate
read failures
write failures
network bandwidth by source/destination topology
```

Important derived metric:

```text
avoided_prefill_gpu_ms
```

rather than only raw cache hit rate.

A 100-token hit and a 128K-token hit are not equally valuable.

---

# 20. Relation to Current Systems

Mooncake is a useful concrete reference: it describes a KV-centric disaggregated architecture and a multi-level cache pool over DRAM/SSD resources, with a separate transfer engine and metadata/placement management.

LMCache similarly separates active GPU state from larger CPU/remote cache capacity and supports prefetch/offload-oriented hierarchies.

The design here intentionally keeps our router-facing contracts generic rather than copying a vendor API.

---

# 21. Design Principles

1. KVTier stores reconstructible execution state, never session truth.
2. Immutable/versioned KV objects simplify concurrent readers.
3. Publish metadata only after data is complete.
4. Worker-local GPU KV remains valid and valuable; KVTier supplements it.
5. Tier placement and per-request tier usage are separate decisions.
6. Promotion/demotion is asynchronous and value-aware.
7. Recompute is always a legal fallback.
8. Request-path restore validates object generation JIT.
9. Replication is based on value/locality, not one fixed factor.
10. Cache hit rate is weaker than avoided-Prefill value.
11. KV semantic/physical compatibility still applies regardless of storage tier.
12. Cross-tenant reuse requires explicit policy safety.

---

# 22. Interview Answer

> **I model the KV Tier as a distributed cache hierarchy specialized for reconstructible LLM execution state. HBM remains the hot working set, while shared GPU/DRAM and eventually NVMe hold progressively colder but expensive-to-recompute prefixes. The tier stores immutable/versioned KV objects and emits residency events into a separate KVDirectoryIndex. The Planner decides whether restoring a CPU or NVMe copy beats recomputation for a particular SLO; the storage system itself manages promotion, demotion, replication and eviction asynchronously. Most importantly, SessionStore remains the semantic source of truth—losing every KV copy causes re-Prefill, not session corruption.**
