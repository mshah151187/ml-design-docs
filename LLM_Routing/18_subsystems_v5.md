# V5 Subsystems and Data/State Planes

**Status:** Canonical V5 integration of previously designed Phase 3 subsystems with session state and context materialization.

## 1. Major planes

V5 has five logically distinct planes:

```text
A. Session State Plane
B. Serving Discovery / Static Metadata Plane
C. Dynamic Scheduling State Plane
D. Precise KV / Context-Materialization Plane
E. Feedback / Performance Learning Plane
```

They must not collapse into one giant database or one giant `ReplicaState` object.

---

## 2. Session State Plane

```text
Session API
    |
    v
SessionStore
    |-- small fenced SessionHead
    |-- append-oriented SessionEvents
    |-- summary/compaction references
    |
    v
ContextInitializer
    |
    v
immutable LogicalContextSnapshot
```

Storage characteristics:

```text
partition by session/segment
ordered append/query within segment
conditional session-head mutation
large content by object reference
```

A Cassandra/Scylla/DynamoDB-style store is a natural implementation shape.

---

## 3. Reconstructible Context Artifact Plane

```text
LogicalContextSnapshot
    |
    v
TargetModelContextMaterializer
    |
    v
ContextArtifactCache
    |
    +-- exact token IDs/ref
    +-- target model/tokenizer/template revisions
    +-- token-context digest
```

Loss of this cache causes recomputation from SessionStore/logical snapshot, not correctness loss.

---

## 4. Serving Discovery / Static Metadata Plane

```text
Kubernetes / Serving Control Plane
        |
   LIST + WATCH
        v
ReplicaDiscoveryManager
        |
        v
Planner-local ReplicaDiscoveryCache
```

Static/slow metadata:

```text
Model Registry
ServingProfile Registry
KVCompatibilityProfile metadata
Cost priors
```

No Kubernetes request is on the inference hot path.

---

## 5. Dynamic Scheduling State Plane

### Model-server telemetry

```text
Model Servers
   |
   v
Durable telemetry log
   |
   v
stream processors
   |
   +--> EngineWorkState
   +--> EngineCapacityState
   +--> ReplicaPerformanceState
   |
   v
Planner-local materialization
```

### Router queue state

```text
RequestQueueSubsystem
    |
    +--> authoritative RouterOutstandingState
    |
    v
Planner-group state stream/materialization
```

Core load invariant:

```text
EffectiveWork = EngineReportedWork + RouterOutstandingWork
```

`RouterOutstanding = READY + PRE_ADMISSION`. PARKED candidates are excluded.

---

## 6. Precise KV State Plane

```text
Model-server KV events
    |
    v
Durable ordered log
    |
    v
KV stream processor
    |
    v
Sharded PreciseKVIndex
```

The index maintains high-cardinality locality outside every Planner process.

Request-path lookup provides:

```text
local prefix match per candidate destination
transferable prefix sources
cache tier
source replica epoch
block manifest reference
confidence/version
```

Canonical KV identity is based on cache compatibility, not blindly on deployment ID:

```text
model revision
+ adapter
+ tokenizer/context identity
+ block/hash schema
+ KV representation/layout contract
```

---

## 7. Context-Materialization Data Plane

```text
ReplicaExecutioner wins destination lease
        |
        v
ContextMaterializationCoordinator
        |
        +--> LOCAL_KV_REUSE
        |
        +--> TRANSFER_KV
        |      |
        |      +--> source PinKVPrefix
        |      +--> destination CreateKVImport
        |      +--> KVTransferService
        |      +--> FinalizeKVImport
        |
        +--> RECOMPUTE_FROM_CONTEXT
               |
               +--> exact token artifact
               +--> destination RecomputeContext
```

### Source vs destination resource semantics

Source:

```text
short bounded pin during active transfer
```

Destination:

```text
NO reservation while request waits in queue
JIT capacity check only after destination wins execution claim
```

This prevents a waiting future request from degrading current work by hoarding KV capacity.

---

## 8. Request Execution Coordination Plane

```text
ExecutionCoordinator
    |
    +--> RequestExecutionRegistry
    |       phase + owner + lease + state
    |
    +--> RequestQueueSubsystem
            per-replica FIFO
```

Executioner state flow:

```text
PENDING
 -> CLAIMED
 -> MATERIALIZING_CONTEXT
 -> EXECUTING_PRE_STREAM
 -> STREAMING
 -> COMMITTING_SESSION      (stateful only)
 -> SUCCESS
```

Same-phase retry:

```text
all materialization plans fail
or retryable pre-stream backend failure
    -> release to PENDING
    -> parked sibling destination candidates re-enter competition
```

---

## 9. Feedback / Performance Learning Plane

```text
SchedulingDecisionEvent
       +
ContextMaterializationObservation
       +
ExecutionObservation
       |
       v
join / derive
       |
       +--> online ReplicaPerformanceState
       |
       +--> latency/cost predictor training examples
```

This supports the final V5 `SLOCostAwareReplicaScheduler` without putting training work in the request path.

---

## 10. Model-affine Planner groups

Ownership remains model-revision affine:

```text
ModelRevision
    -> PlannerShardGroup
       -> several warm Planner instances
```

All compatible deployments/ServingProfiles of that model revision are visible to the owning Planner group.

Planner processes continuously materialize small state locally. Precise KV remains in the specialized sharded index.

---

## 11. Final request-path data dependencies

At request time, Planner should mainly perform:

```text
local discovery lookup
local ServingProfile/state joins
one bounded precise-KV lookup
feature assembly
ReplicaScheduler inference
```

The design avoids request-time fanout to:

```text
Kubernetes
all model servers
remote state databases for every replica
```

---

## 12. V5 subsystem invariants

1. SessionStore is durable semantic truth.
2. Exact tokens are model-specific reconstructible artifacts.
3. KV is ephemeral and may be local, transferred, or recomputed.
4. Planner constructs facts; ReplicaScheduler owns policy.
5. ReplicaScheduler returns destination plans with materialization fallback chains.
6. Destination KV capacity is not reserved while queued.
7. Source blocks may be pinned only for bounded active transfer attempts.
8. RequestQueue state is global/shared among Planners and is part of scheduling load.
9. RequestExecutionRegistry owns one live destination execution lease in base V5.
10. Context materialization occurs only after destination ownership.
11. Transfer failure falls back to recompute before abandoning that destination.
12. If all destination materialization plans fail, release to PENDING for sibling retry.
13. Session commit occurs only after successful complete inference, and stateful platform `SUCCESS` occurs only after that commit.
14. Decision-time feature snapshots are preserved for learned latency/SLO feedback.
