# V5 Architecture Delta — External KV Tier

**Status:** Final V5 extension  
**Scope:** Only the components changed or added when reusable KV becomes an independent distributed subsystem.

---

# 1. Why This Extension Exists

The earlier V5 design treated KV cache as an execution artifact that lived primarily on model workers.

That gives three materialization paths:

```text
LOCAL_KV_REUSE
TRANSFER_KV_FROM_MODEL_WORKER
RECOMPUTE_FROM_CONTEXT
```

This works, but it creates a strong coupling:

> **Where compute happened determines where reusable execution state lives.**

For long-context, multi-turn and high-prefix-reuse workloads, that coupling becomes expensive.

A worker may be:

```text
overloaded
draining
restarted
removed by autoscaling
poorly located for the next request
```

while holding extremely valuable 128K-token KV state.

The external KV Tier breaks this coupling.

```text
compute placement
    !=
longer-lived KV residency
```

---

# 2. Final State Hierarchy

```text
Tier 1 — Durable semantic truth
================================
SessionStore

conversation events
session version
turn lease / fencing
summary checkpoints


Tier 2 — Reconstructible target-model artifact
===============================================
TargetModelContextArtifact

exact token IDs
template/tokenizer revision
context digest
canonical prefix block identities


Tier 3 — Reconstructible execution state
=========================================
KV Tier + worker-local KV

GPU KV
CPU/DRAM KV
NVMe/remote KV
exact physical representations
```

Loss semantics:

```text
SessionStore lost
    -> correctness failure

TargetModelContextArtifact lost
    -> rebuild from logical context

KV copies lost
    -> re-Prefill from target-model context
```

Core invariant:

> **KV can become durable in the storage sense without becoming authoritative in the semantic sense.**

---

# 3. Changed Component Ownership

```text
SessionStore
    "What conversation version is correct?"

TargetModelContextArtifact
    "What exact target-model token prefix represents that context?"

KVDirectoryIndex
    "What compatible materializations of that logical KV currently exist,
     and where?"

KVTier
    "Store, replicate, tier, fetch and evict the actual KV bytes."

Planner
    "Which execution topology + materialization path best satisfies SLO/cost?"

ContextMaterializationCoordinator
    "Execute the selected context-materialization chain."

ExecutionCoordinator
    "Orchestrate request/model phase and bounded multi-stage execution."
```

---

# 4. New Request-Path Mental Model

```text
LogicalContextSnapshot
        |
        v
QualityModelSelector
        |
        v
TargetModelContextArtifact
        |
        +-----------------------------+
        |                             |
        v                             v
Replica/Serving State           KVDirectoryIndex
                                      |
                     +----------------+----------------+
                     |                |                |
                     v                v                v
                 Worker GPU      Shared GPU        CPU/NVMe
                     \                |                /
                      \               |               /
                       +--------------+--------------+
                                      |
                                      v
                                   Planner
                                      |
                                      v
                           ExecutionTopologyPlan
                                      |
                                      v
                         ExecutionCoordinator
                                      |
                           terminal worker claim
                                      |
                                      v
                      ContextMaterializationCoordinator
                       /       |        |         \
                      v        v        v          v
                   local    worker    KV Tier    recompute
                    KV      transfer    restore
                                      |
                                      v
                              P/D model execution
                                      |
                                      v
                              SessionCommitter
```

---

# 5. Why SessionStore Does Not Store Physical KV Location

A tempting design is:

```text
Session S123
    kv_node = node17
    tier = CPU
```

Do not make that authoritative.

Physical residency changes independently due to:

```text
eviction
replication
promotion
demotion
worker restart
tier-node failure
rebalancing
autoscaling
```

Instead SessionStore may retain only stable semantic/artifact identity:

```text
Session S123@27
    logical_context_digest = H1
```

The target-model materializer later derives:

```text
LogicalKVIdentity(model_revision, exact_prefix_digest)
```

and the KVDirectoryIndex resolves current residencies.

---

# 6. Why KVDirectoryIndex Is Separate From KVTier

The request path needs a fast metadata question:

```text
"What exists and where?"
```

It should not discover that by contacting every KV storage node.

So:

```text
KVDirectoryIndex
    metadata/control plane

KVTier
    bytes/data plane
```

This mirrors the same design principle used earlier for ReplicaDiscovery versus model-server execution.

The directory is allowed to be stale within a bounded freshness model; the materialization coordinator performs a final source/version validation before consuming data.

---

# 7. Worker-Local KV Still Matters

Adding an external KV tier does **not** mean every request should write/read remote KV.

Fastest path remains:

```text
selected worker already has exact compatible KV
    -> LOCAL_KV_REUSE
```

A useful hierarchy is:

```text
worker-local GPU
shared/remote GPU
near CPU / pinned DRAM
NVMe / remote storage
recompute
```

But this is not a fixed priority order.

For a 200-token prefix:

```text
recompute
```

may beat SSD restore.

For a 128K prefix:

```text
remote DRAM/NVMe restore
```

can easily beat re-Prefill.

Therefore tier is an input to a cost/SLO decision, not a hard-coded policy.

---

# 8. Stateless Request

```text
request prefix
      |
      v
TargetModelContextArtifact
      |
      v
KVDirectoryIndex.FindLongestPrefix
      |
      +--> D7 local GPU: 60K
      +--> KVNode3 CPU: 120K
      +--> NVMe copy: 128K
      |
      v
Planner compares:
    D7 local reuse + recompute 68K
    restore CPU 120K + recompute 8K
    restore NVMe 128K
    fresh P Prefill 128K
```

The cache system is now location-independent.

---

# 9. Stateful Request

```text
SessionStore:
    S123 committed version = 27
    logical context = correct semantic truth
```

New turn:

```text
SessionStore.LoadContext
        |
        v
LogicalContextSnapshot
        |
        v
TargetModelContextArtifact for model M
        |
        v
KVDirectoryIndex
        |
        +--> freshest compatible prefix on D7 GPU
        +--> checkpointed compatible prefix in CPU tier
        +--> colder copy on NVMe
```

If every KV copy disappears:

```text
SessionStore
    ->
rebuild exact target context
    ->
fresh Prefill
```

The session remains correct.

---

# 10. P/D Disaggregation With KV Tier

Possible topology candidates now include:

```text
LOCAL_PD(D7)

P3 -> D7

WORKER_KV(D2) -> D7

WORKER_KV(D2) -> P3 -> D7

KV_CPU(Node9) -> D7

KV_CPU(Node9) -> P3 -> D7

KV_NVME(Node4) -> P3 -> D7

RECOMPUTE(P3) -> D7
```

This is why the Planner's output naturally evolves from a simple replica plan toward an execution topology plan.

---

# 11. InferenceExecutionGraphPlanner — Future Extraction Point

Do not create a separate service merely because we can name one.

Base V5 keeps:

```text
Planner
    -> builds ExecutionTopologyPlan
```

If later topologies expand into:

```text
Encode -> Prefill -> Decode
retrieval stages
multiple KV transform stages
speculative draft/verifier graphs
multi-model semantic KV mapping
```

then extract the graph-building policy into:

```text
InferenceExecutionGraphPlanner
```

Likewise, only introduce an `InferenceExecutionGraphExecutor` if execution orchestration becomes richer than the current ExecutionCoordinator + materialization coordinator can cleanly own.

---

# 12. Final Design Principles

1. SessionStore remains semantic truth.
2. KV Tier owns bytes, never conversation truth.
3. KVDirectoryIndex owns residency metadata, never bytes.
4. Logical KV identity is distinct from physical representation.
5. One logical KV may have several physical residencies.
6. Exact physical transforms are legal only when semantic identity matches.
7. Worker-local KV remains the lowest-latency path when available.
8. Tier selection is SLO/cost driven, not a fixed GPU>CPU>SSD rule.
9. Directory lookup is decision-time evidence; final materialization performs JIT validation.
10. Recompute remains the universal correctness fallback.
11. Planner selects topology; ContextMaterializationCoordinator executes movement/restoration.
12. `InferenceExecutionGraphPlanner` is an extraction point, not mandatory V5 complexity.
