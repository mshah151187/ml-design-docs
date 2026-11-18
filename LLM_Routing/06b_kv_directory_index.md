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

# 12. Lookup Process — Block Chain Walk and Representation Resolution

A single `FindContextResidency` call executes in two sequential phases.
Phase 2 only runs if Phase 1 finds at least one matching block.

---

## Phase 1 — Block Chain Walk

The shard receives the full set of canonical block keys `B0 ... Bn` upfront
in `canonical_prefix_block_keys_ref`. It does not receive them one at a time.

```text
for i = 0, 1, 2, ... n:
    probe shard index for Bi in model namespace
    if HIT:  record match, continue
    if MISS: stop
```

Each probe is a hash map lookup against the index entry:

```text
index entry:  Bi -> [residency_id, tier, node, epoch, physical_representation, confidence]
```

Walk stops at the first miss because of the chaining invariant: Bi encodes
`H(B(i-1), tokens[i*block:(i+1)*block])`. A destination cannot use blocks
B0..B(k-1) in GPU memory without the full unbroken prefix from B0.

```text
best case:   O(1)  — B0 not found, stop immediately (total miss)
worst case:  O(n)  — all blocks found, full prefix reuse
typical:     O(k)  — hits until the session-specific divergence point
```

Output of Phase 1: `matched_blocks = k` (zero if total miss).

---

## Phase 2 — Representation Resolution

Runs only when `matched_blocks > 0`.

For the matched block range `B0 ... B(k-1)`, the shard fetches all residencies
that cover those blocks. Each residency carries a `physical_representation`
(TP degree, layout, dtype, block size — the PhysicalKVRepresentation from section 3).

The shard then compares every stored representation against every
`target_physical_representations[i]` from the request.

```text
For each stored residency R with physical_representation Ps:

    For each requested representation Pt in target_physical_representations:

        if Ps == Pt:
            -> PrefixResidencyMatch (direct hit, no transform needed)
               add to longest_prefix_matches[]

        elif transform(Ps -> Pt) is registered and supported:
            -> ExactTransformOption (indirect hit, transform needed)
               add to exact_transform_options[]
               pick lowest-cost transform path if multiple source options exist

        else:
            -> skip (this residency is not useful for candidate replicas needing Pt)
```

A stored residency whose physical representation is not requested by any candidate
and has no registered transform to any requested representation is silently excluded
from the result.

---

## Result Assembly

```text
ContextResidencyResult {
    longest_prefix_matches[]    <- one entry per (residency, direct-match candidate group)
    exact_transform_options[]   <- one entry per (source residency, target representation)
    index_version
    observed_at
}
```

Both fields are populated in one pass and returned together to Planner.
The Planner uses `matched_tokens` and `estimated_transform_cost` to decide
whether restoring and transforming beats fresh Prefill for each candidate.

---

## Relationship Between the Two Phases

```text
Phase 1  answers: HOW MUCH prefix is reusable?
                  (block count, token count — same answer for all candidates)

Phase 2  answers: WHO can use it, and at what transformation cost?
                  (depends on each candidate's physical representation)
```

A full 30-block hit with a required TP reshard may lose to a 18-block hit with a
direct match if the reshard cost exceeds the Prefill cost of the remaining 12 blocks.
Planner has both numbers to make that decision.

---

# 13. Cross-Model Semantic Mapping

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

# 14. Sharding

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

# 15. Worker-Local and Tier Residency in One Result

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

# 16. End-to-End Lookup Example — Session Turn 3 Switching to Llama-70B

This walks through a concrete lookup from the moment TargetModelContextMaterializer
finishes through the result returned to Planner.

**Scenario:** Multi-turn session S123. Turns 1 and 2 were served by Llama-3B.
Phase 2 now selects Llama-70B for Turn 3 because the new question requires
implementation-level depth. The Planner needs to know whether any Llama-70B
KV already exists anywhere.

---

## 15.1 What TargetModelContextMaterializer Hands to Planner

After applying the Llama-70B chat template and tokenizer the artifact contains:

```text
TargetModelContextArtifact {
    model_revision         = "llama-70b-v3"
    tokenizer_revision     = "tok-v7"
    chat_template_revision = "llama3-chat-v2"

    token_ids_ref          = "blob://tokens-req456"
    input_token_count      = 480         <- turns 1+2 context + new user turn

    token_context_digest   = "D_70b_full_turn3"
}
```

The Planner derives canonical prefix block keys from those exact token IDs.
Block size is determined by the ServingProfile of the candidate replicas (16 tokens here).

---

## 15.2 Canonical Prefix Block Key Derivation

```text
namespace = "llama-70b-v3 / tok-v7 / llama3-chat-v2 / hash-schema-v1"

B0  = H(namespace, tokens[0:16])     <- system prompt tokens
B1  = H(B0,        tokens[16:32])    <- still system prompt
B2  = H(B1,        tokens[32:48])    <- start of user turn 1
B3  = H(B2,        tokens[48:64])    <- assistant turn 1
...
B17 = H(B16,       tokens[272:288])  <- end of assistant turn 2
B18 = H(B17,       tokens[288:304])  <- new user turn 3 begins
...
B29 = H(B28,       tokens[464:480])  <- end of user turn 3
```

Chaining is critical: B5 incorporates everything in B0..B4.
A residency that has B3 but not B2 is unreachable and never reported.

---

## 15.3 Shard Routing

The request routes to exactly one shard:

```text
shard_key = H(namespace, B0, B1)
          -> KVDirectory Shard #42
```

`namespace` separates models: Llama-70B and Llama-3B never share a shard even
with the same system prompt.

`B0 + B1` represent the first 32 tokens — almost always the system prompt.
All requests using the same model + same system prompt hash to Shard #42, so the
directory can answer a full prefix-chain lookup without fanout to other shards.

---

## 15.4 FindContextResidency Request

```text
FindContextResidencyRequest {
    request_id                      = "req-456"

    logical_kv_identity = {
        model_id                    = "llama"
        model_revision              = "llama-70b-v3"
        tokenizer_revision          = "tok-v7"
        chat_template_revision      = "llama3-chat-v2"
        target_context_digest       = "D_70b_full_turn3"
        hash_schema_version         = "hash-schema-v1"
    }

    canonical_prefix_block_keys_ref = "blob://blocks-req456"   <- B0 ... B29
    target_physical_representations = [TP4_NHD, TP2_HND]       <- what candidate D replicas use
    candidate_compute_nodes         = [D7, D3, D9]
    max_results                     = 5
}
```

---

## 15.5 Block Chain Walk Inside Shard #42

Shard #42 walks the block chain starting from B0 in the Llama-70B namespace.

**Case A — Total miss (our session; turns 1+2 were Llama-3B KV only)**

```text
B0 in llama-70b-v3 namespace -> NOT FOUND
Stop. Longest continuous match = 0 blocks.
```

No further blocks are checked. A hit at B5 without B0 is meaningless because
the first 16 tokens would be missing from the destination KV.

**Case B — Partial hit (shared system prompt)**

Suppose other requests have previously run Llama-70B with the same system prompt
and their KV was checkpointed. The system prompt ends around block 3.

```text
B0 -> FOUND  D7 GPU  (from a different session, 2 hrs ago)
B1 -> FOUND  D7 GPU
B2 -> FOUND  D7 GPU
B3 -> FOUND  D7 GPU  <- system prompt ends here
B4 -> NOT FOUND      <- session-specific content diverges here
Stop. Longest continuous match = 4 blocks = 64 tokens.
```

64 system-prompt tokens are reusable. The remaining 416 tokens need Prefill.

**Case C — Full hit (Turn 4, same session, still Llama-70B)**

After Turn 3 completes and its KV is checkpointed, the next turn's block chain
shares B0..B29 exactly.

```text
B0  -> FOUND  D7 GPU
B1  -> FOUND  D7 GPU
...
B29 -> FOUND  D7 GPU
B30 -> NOT FOUND     <- Turn 4's new user tokens not yet computed
Stop. Longest continuous match = 30 blocks = 480 tokens.
```

Entire prior context is reusable. Only Turn 4's new tokens need Prefill.

---

## 15.6 Result Returned to Planner

**Case A result:**

```text
ContextResidencyResult {
    request_id             = "req-456"
    longest_prefix_matches = []     <- empty; no Llama-70B KV exists anywhere
    exact_transform_options = []
    index_version          = 8841
    observed_at            = 2026-08-27T10:06:02Z
}
```

Planner builds one candidate: `RECOMPUTE_P_THEN_D`, uncached_prefill_tokens = 480.

**Case B result:**

```text
ContextResidencyResult {
    longest_prefix_matches = [
        PrefixResidencyMatch {
            residency_id      = "res-D7-001"
            matched_blocks    = 4
            matched_tokens    = 64
            tier              = WORKER_GPU
            node_id           = "D7"
            replica_epoch     = 12
            physical_representation = TP4_NHD    <- direct match, no transform
            confidence        = HIGH
        }
    ]
}
```

Planner builds: `LOCAL_KV_REUSE(D7, 64 tokens) + Prefill 416 tokens`.

**Case C result:**

```text
ContextResidencyResult {
    longest_prefix_matches = [
        PrefixResidencyMatch {
            matched_blocks    = 30
            matched_tokens    = 480
            tier              = WORKER_GPU
            node_id           = "D7"
            confidence        = HIGH
        }
    ]
}
```

Planner builds: `LOCAL_KV_REUSE(D7, 480 tokens)` — no Prefill needed,
only Turn 4's new tokens appended.

---

## 15.7 Why the Directory Never Returns a Llama-3B Hit for a Llama-70B Request

The `logical_kv_identity` carries `model_revision = "llama-70b-v3"`. The namespace
embedded in B0 also encodes the model revision. Llama-3B KV lives under a different
namespace and therefore on a different shard entirely. There is no code path that
returns a 3B residency in response to a 70B lookup — the namespace mismatch makes
them invisible to each other by construction.

---

## 15.8 Multi-Profile Residency Matching — Block Walk Then Representation Match

The lookup is two sequential phases.
Phase 1 finds the longest matching block prefix.
Phase 2 matches the stored physical representations against what the candidate replicas need.

**Setup — 5 candidate replicas for llama-70b-v3**

Planner runs ReplicaDiscovery and ServingMetadataRegistry lookups first:

```text
Replica  serving_profile_id    KVCompatibilityProfile  -> PhysicalRepresentation
──────── ─────────────────────────────────────────────────────────────────────────
D1       sp-h100-tp4-fp8       kv-compat-001           -> P1 (TP4, NHD, fp8)
D2       sp-h100-tp4-fp8       kv-compat-001           -> P1 (TP4, NHD, fp8)   <- same as D1
D3       sp-h100-tp2-bf16      kv-compat-002           -> P2 (TP2, HND, bf16)
D4       sp-a100-tp4-bf16      kv-compat-004           -> P4 (TP4, NHD, bf16)
D5       sp-a100-tp2-fp8       kv-compat-005           -> P5 (TP2, HND, fp8)
```

D1 and D2 share the same KVCompatibilityProfile so they share the same physical
representation P1. The request carries four distinct target representations:

```text
target_physical_representations = [P1, P2, P4, P5]
```

**Phase 1 — Block Chain Walk (same as 15.5)**

The shard walks B0 → B17 and finds a continuous match.
B18 is missing. Block walk stops.

```text
Longest matching prefix = 18 blocks = 288 tokens
```

**Phase 2 — Representation Matching for the 18-block prefix**

The shard now asks: for the residencies that hold these 18 blocks,
what physical representations are stored?

```text
Stored residencies for blocks B0..B17:

res-A:  P1 (TP4, NHD, fp8)   on D7 GPU    <- written by a previous TP4 fp8 job
res-B:  P2 (TP2, HND, bf16)  on CPU Node9  <- written by a previous TP2 bf16 job
res-C:  P3 (TP2, NHD, bf16)  on NVMe Node4 <- written by yet another config
```

Three stored representations: P1, P2, P3.
Four requested representations: P1, P2, P4, P5.

**Match table:**

```text
Stored  Requested  Outcome
──────  ─────────  ────────────────────────────────────────────────────────
P1      P1         DIRECT HIT
                   D1 and D2 can reuse res-A directly, no transform needed

P2      P2         DIRECT HIT
                   D3 can reuse res-B directly, no transform needed

P3      —          IRRELEVANT
                   No candidate replica uses P3; res-C is not reported

P1->P4  P4         EXACT TRANSFORM CANDIDATE
                   D4 needs TP4 NHD bf16. res-A has TP4 NHD fp8.
                   Same TP degree and layout, only dtype differs (fp8->bf16).
                   If a dequant/recast kernel is registered: ExactTransformOption.
                   Estimated transform cost: low (dtype cast only, no reshard).

P1->P5  P5         EXACT TRANSFORM CANDIDATE
        or         D5 needs TP2 HND fp8. res-A has TP4 NHD fp8.
P2->P5             TP reshard (4->2) + layout transpose (NHD->HND) required.
                   res-B has TP2 HND bf16 — same shape, only dtype differs.
                   P2->P5 transform is cheaper (dtype cast only).
                   Directory reports the lower-cost transform path.
```

**Result returned to Planner:**

```text
ContextResidencyResult {
    longest_prefix_matches = [

        PrefixResidencyMatch {             <- direct hit for D1, D2
            residency_id      = "res-A"
            matched_blocks    = 18
            matched_tokens    = 288
            tier              = WORKER_GPU
            node_id           = "D7"
            replica_epoch     = 12
            physical_representation = P1
            confidence        = HIGH
        },

        PrefixResidencyMatch {             <- direct hit for D3
            residency_id      = "res-B"
            matched_blocks    = 18
            matched_tokens    = 288
            tier              = CPU_DRAM
            node_id           = "Node9"
            physical_representation = P2
            confidence        = HIGH
        }
    ],

    exact_transform_options = [

        ExactTransformOption {             <- for D4 (needs P4)
            source_residency_id   = "res-A"
            target_representation = P4
            transform_type        = DTYPE_CAST   <- fp8 -> bf16, same TP4 NHD layout
            estimated_transform_bytes = 18 * 16 * model_dim * 2
            estimated_transform_cost  = low
            supported = true
        },

        ExactTransformOption {             <- for D5 (needs P5), cheaper path via res-B
            source_residency_id   = "res-B"
            target_representation = P5
            transform_type        = DTYPE_CAST   <- bf16 -> fp8, same TP2 HND layout
            estimated_transform_bytes = 18 * 16 * model_dim * 0.5
            estimated_transform_cost  = low
            supported = true
        }
    ]
}
```

**What Planner does with this:**

```text
D1, D2  -> LOCAL_KV_REUSE(res-A, 288 tokens) + Prefill 192 tokens   no transform
D3      -> RESTORE_CPU_KV(res-B, 288 tokens)  + Prefill 192 tokens   no transform
D4      -> RESTORE + DTYPE_CAST(res-A->P4)    + Prefill 192 tokens   cheap transform
D5      -> RESTORE + DTYPE_CAST(res-B->P5)    + Prefill 192 tokens   cheaper than res-A path
```

P3 (res-C on NVMe) is never surfaced — no candidate replica requested it
so the directory silently excludes it from the result.

**Key rule:**

```text
Phase 1  (block walk)        answers: HOW MUCH prefix is reusable?
Phase 2  (representation match) answers: WHO can use it, and at what transform cost?
```

These two phases compose. A 30-block full hit with a required TP reshard may be
worse than a 18-block hit with a direct match, depending on transform cost vs
Prefill cost. Planner has both numbers to make that call.

---

# 17. Session Interaction

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

# 18. Race Boundary

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

# 19. Failure Semantics

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

# 20. Design Principles

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

# 21. Interview Answer

> **I evolve the old PreciseKVIndex into a KVDirectoryIndex once KV can live outside model replicas. Its key abstraction is a logical KV identity—exact model revision plus exact target-token prefix—and each logical KV can have multiple physical residencies: worker GPU, shared GPU, CPU DRAM or NVMe, potentially in different TP/layout representations. The directory stores metadata and transform availability, not bytes. The Planner uses it to enumerate materialization paths, while the ContextMaterializationCoordinator later performs a JIT source/version validation before restoring or transferring. That keeps cache discovery fast without making stale metadata a correctness dependency.**
