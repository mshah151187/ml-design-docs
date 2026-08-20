# V5 End-to-End Architecture

**Status:** Canonical V5 integration contract

## 1. What V5 adds

V4 assumed a request already contained a self-contained model context. V5 supports both:

```text
Stateless request:
    full messages/prompt supplied in request

Stateful request:
    session_id + new_turn
```

For stateful requests the serving platform, not the client, owns reconstruction of the durable logical conversation context.

V5 also makes cross-replica KV transfer a first-class context-materialization path rather than a backlog optimization.

## 2. Full flow

```text
Wire Request
   |
   v
RequestNormalizer
   |
   v
NormalizedRequest
   |
   v
ContextInitializer ------------------------------+
   |                                             |
   | stateful                                    |
   +----> SessionStore                           |
   |       BeginTurn + LoadContext               |
   |                                             |
   +---------------------------------------------+
   |
   v
LogicalContextSnapshot
   |
   v
QualityModelSelector
   |
   v
ModelRoutePlan
   |
   +--> phase 0 model
   +--> phase 1 fallback model
   +--> ...
   |
   v
TargetModelContextMaterializer (per model phase)
   |
   v
TargetModelContextArtifact
   |
   v
Planner
   |--- ReplicaDiscovery
   |--- ServingProfile cache
   |--- Engine state
   |--- RouterOutstanding state
   |--- Performance state
   |--- Precise KV index
   v
SchedulingProblem
   |
   v
SLOCostAwareReplicaScheduler
   |
   v
ModelReplicaPlan
   |  candidate = destination replica
   |              + ordered context-materialization chain
   v
ExecutionCoordinator
   |
   v
RequestQueueSubsystem
   |
   v
ReplicaExecutioner
   |--- TryClaim
   v
RequestExecutionRegistry
   |
   v
ContextMaterializationCoordinator
   |--- LOCAL_KV_REUSE
   |--- TRANSFER_KV
   |--- RECOMPUTE_FROM_CONTEXT
   v
ModelServerContextAPI / KVTransferService
   |
   v
Model execution
   |
   v
COMMITTING_SESSION
   |
   v
SessionCommitter
   |
   v
Committed session version + response
```

## 3. State hierarchy

### Tier 1 — durable semantic state

```text
SessionStore
    SessionHead
    ordered SessionEvents
    pending-turn ownership/versioning
```

Loss of this state is a correctness failure.

### Tier 2 — reconstructible model artifact

```text
TargetModelContextArtifact
    exact token IDs/reference
    tokenizer/chat-template revisions
    context digest
```

If lost, rebuild from the logical session snapshot.

### Tier 3 — ephemeral execution optimization

```text
KV cache
    GPU/CPU/shared residency
    precise prefix index
    transferable blocks
```

If lost, recompute from Tier 2.

## 4. V5 scheduler objective

A scheduling candidate is no longer only a replica `R`. It is a destination replica plus an ordered context-materialization chain.

For the primary context mode `M`:

```text
Candidate = (destination replica R, context mode M)
```

The latency predictor estimates:

```text
BacklogTime(R)
+ ContextReadyTime(X, R, M)
+ Decode/GenerationTime(X, R)
```

The scheduler first enforces the request's latency/SLO probability contract, then minimizes predicted cost among feasible candidates.

## 5. Context-materialization fallback

Example destination R3:

```text
ContextMaterializationChain
    0. TRANSFER_KV(source=R1, destination=R3)
    1. RECOMPUTE_FROM_CONTEXT(destination=R3)
```

The destination does **not** reserve KV memory while the request waits in the replica queue.

Only after R3 wins the execution lease does the ContextMaterializationCoordinator perform a just-in-time capacity check.

The source may be briefly pinned during a transfer attempt.

If every context plan for R3 fails, R3 releases the execution lease and parked sibling destination candidates may compete.

## 6. Core ownership boundaries

| Component | Owns |
|---|---|
| SessionStore | durable conversation state + turn fencing |
| ContextInitializer | immutable model-independent request context |
| QualityModelSelector | quality-approved ordered model phases |
| TargetModelContextMaterializer | exact target-model tokenization/context artifact |
| Planner | factual candidate/context assembly |
| ReplicaScheduler | latency/SLO/cost policy |
| ExecutionCoordinator | model phase activation/fallback budget |
| RequestQueueSubsystem | per-replica FIFO + RouterOutstanding accounting |
| RequestExecutionRegistry | one destination execution owner/lease/phase |
| ContextMaterializationCoordinator | local reuse/transfer/recompute fallback chain |
| KVTransferService | bulk KV movement only |
| ReplicaExecutioner | claim, materialize, execute, park/retry |
| SessionCommitter | successful turn commit/version advance |
| Feedback pipeline | observations, online performance state, training data |
