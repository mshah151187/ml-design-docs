# Component — ExecutionCoordinator (Topology-Aware Update)

## 1. Why This Component Changes

Earlier V5 execution assumed:

```text
candidate = one destination replica
```

and the candidate's context was materialized after that replica won the request lease.

With P/D disaggregation, a chosen candidate may contain:

```text
optional Prefill prerequisite
+
one terminal Decode/local execution worker
```

Example:

```text
KV CPU -> P3 -> D7
```

The ExecutionCoordinator therefore needs to orchestrate a bounded topology, while preserving the one-owner/queue-arbitration invariants.

---

# 2. Base V5 Topology Constraint

To avoid premature DAG complexity, base V5 supports:

```text
optional context source
        ->
optional one Prefill stage
        ->
one terminal Decode/execution stage
```

Examples:

```text
LOCAL_PD(D7)

P3 -> D7

KV_TIER -> D7

KV_TIER -> P3 -> D7
```

Not arbitrary graphs.

---

# 3. Terminal Worker Owns the Request Lease

Keep the existing queue-arbitrated execution model.

```text
ExecutionTopologyPlan
    |
    +--> candidate terminal D7
    +--> candidate terminal D3
    +--> candidate terminal D9
```

These candidates fan into terminal worker queues.

One terminal worker wins:

```text
RequestExecutionRegistry.TryClaim
```

Only then does the system execute that topology's prerequisites.

Why terminal ownership first?

Because otherwise:

```text
P3 starts expensive Prefill
```

without knowing whether the corresponding D7 topology will actually own the request.

This could duplicate expensive work across competing topologies.

---

# 4. Execution Flow

```text
ExecutionCoordinator
    |
    | activate model phase / topology candidates
    v
RequestQueueSubsystem
    |
    v
terminal worker D7 Executioner
    |
    | TryClaim
    v
CLAIMED
    |
    v
ExecutionCoordinator / topology runner
    |
    +--> materialize context on P3 if needed
    +--> run Prefill P3
    +--> materialize P3 KV on D7
    |
    v
D7 EXECUTING_PRE_STREAM
    |
    v
STREAMING
    |
    v
COMMITTING_SESSION
    |
    v
SUCCESS
```

---

# 5. Why Not Put the Request Into Both P and D Queues?

Naive:

```text
request -> Q[P3]
request -> Q[D7]
```

creates independent ownership races.

Potential failures:

```text
P3 executes but D7 topology loses
two P workers start same request
D claims before prerequisite P work is available
complex distributed cancellation
```

Base V5 avoids this.

P is a prerequisite resource coordinated under the already-owned terminal topology.

If future scale requires independently queued P-stage work, that is a strong signal to extract a richer `InferenceExecutionGraphExecutor`.

---

# 6. State Machine

Updated request state can include topology substate:

```text
PENDING
 -> CLAIMED
 -> PREPARING_TOPOLOGY
 -> MATERIALIZING_PREFILL_CONTEXT      # optional
 -> RUNNING_PREFILL                    # optional
 -> MATERIALIZING_DECODE_CONTEXT
 -> EXECUTING_PRE_STREAM
 -> STREAMING
 -> COMMITTING_SESSION                 # stateful
 -> SUCCESS
```

Failures before streaming may still release/retry according to policy.

---

# 7. Conditional Disaggregation

The ExecutionCoordinator does not decide whether P is needed.

That is already encoded:

```text
candidate.mode = LOCAL_PD
```

or:

```text
candidate.mode = DISAGGREGATED_PD
```

Coordinator merely executes the selected topology.

This preserves the policy/execution boundary.

---

# 8. Prefill Resource Acquisition

Once terminal D7 owns the topology:

```text
acquire/dispatch bounded Prefill work to P3
```

P3 should still apply local admission/backpressure.

If P3 rejects before useful work begins:

```text
try topology's alternate P prerequisite if planned
or
fail/release topology candidate
```

Do not silently substitute arbitrary P workers unless Planner provided an equivalent fallback.

---

# 9. P Failure After Prefill Produces KV

If P3 successfully produces usable KV but the P->D transfer fails:

```text
ContextMaterializationCoordinator
    -> same-D fallback
```

Examples:

```text
retry transfer
restore checkpointed copy if published
recompute locally on D if allowed
```

Avoid discarding expensive P output prematurely if a valid alternative materialization path exists.

---

# 10. D Capacity Failure

Even though D7 won the request lease, actual model-server/KV capacity can change.

If D7 cannot JIT allocate before streaming:

```text
topology fails pre-stream
release execution lease
return request to PENDING
```

Parked sibling terminal candidates can compete.

This retains the earlier fast correction mechanism.

---

# 11. Streaming Boundary

Once D7 has emitted externally visible output:

```text
ordinary candidate retry is no longer safe
```

Do not launch a sibling topology and replay output unless a separate mid-stream recovery/hedging design explicitly supports it.

---

# 12. Session Commit

For stateful requests:

```text
Decode finishes
    ->
COMMITTING_SESSION
```

Only after SessionStore commits:

```text
N -> N+1
```

does the platform mark request success.

KV checkpoint/publication can occur after commit or best-effort around this boundary, but it does not replace the semantic transaction.

---

# 13. KVTier Checkpoint Trigger

ExecutionCoordinator may emit a post-execution signal such as:

```text
KVCheckpointCandidate {
    session_id?
    target_context_digest
    model_revision
    destination_worker
    prefix_tokens
    estimated_reuse_value
}
```

A separate KV-tier policy decides whether to persist/replicate it.

Do not synchronously checkpoint every request merely because a KVTier exists.

---

# 14. Model Phase Fallback

Phase 2 may provide:

```text
phase 0: Model A
phase 1: Model B
```

ExecutionCoordinator retains model-phase authority.

It must not advance to phase 1 while phase 0 legitimately owns/executing a topology.

Phase 1 activates only after the current phase reaches a defined no-action/terminal-failure condition.

---

# 15. Failure Classification

## Materialization-local

```text
one tier source stale
```

Handle inside ContextMaterializationCoordinator.

## Prefill-stage local

```text
P admission fails
P dies pre-stream
```

Use planned prerequisite fallback or fail topology.

## Terminal D local

```text
JIT capacity failure
backend admission rejection
```

Release terminal claim for sibling candidate.

## Model-wide

```text
all topology candidates fail due model deployment issue
```

ExecutionCoordinator may advance quality-approved model phase.

## Session commit failure

Do not call request success.

Resolve with session fencing/idempotency semantics.

---

# 16. InferenceExecutionGraphExecutor — Future Extraction Point

If topologies later require:

```text
independent P queues
Encode stage
multiple parallel prerequisites
semantic mapper service
speculative draft/verifier
branch/join execution
```

then the current ExecutionCoordinator becomes too overloaded.

Extract:

```text
ExecutionCoordinator
    -> model phase / request lifecycle

InferenceExecutionGraphExecutor
    -> execute one chosen graph
```

Base V5 does not need that service yet.

---

# 17. Design Principles

1. Terminal Decode/local worker remains one request execution owner in base V5.
2. Remote Prefill is a prerequisite of the owned topology, not an independent competing request.
3. Planner decides topology; coordinator executes topology.
4. ContextMaterializationCoordinator executes each state edge.
5. P/D/KVTier failures before streaming remain retryable according to bounded plans.
6. Destination resource reservation remains JIT.
7. Do not duplicate Prefill work across competing terminal candidates.
8. Streaming remains the normal retry boundary.
9. Session commit remains after successful inference.
10. KV checkpointing is an optimization, not semantic success.
11. Arbitrary DAG execution is deferred until complexity justifies an `InferenceExecutionGraphExecutor`.

---

# 18. Interview Answer

> **I preserve queue arbitration by making the terminal Decode or local-hybrid worker the request's execution owner. Once D7 wins the atomic lease, the coordinator executes that candidate's prerequisites—restore old KV to P3 if needed, run Prefill, hand the resulting KV to D7, then start Decode. I deliberately don't enqueue the same request independently on P and D, because that creates two ownership domains and wasted Prefill if the paired D loses. This gives me a bounded topology model today; if I later need independently scheduled Encode/Prefill/Decode DAGs, I'd extract a dedicated InferenceExecutionGraphExecutor.**
