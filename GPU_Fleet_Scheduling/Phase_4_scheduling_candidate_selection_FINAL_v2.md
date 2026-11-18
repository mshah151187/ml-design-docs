# Phase 4 — Scheduling Candidate Selection

## 1. Purpose

Phase 4 is the scheduler's **demand-selection brain**.

> **Among all pending ResourceIntents for a queue, choose the next ResourceIntent that the scheduler should spend expensive capacity-analysis work on.**

Phase 4 reads exact policy/accounting state from Phase 3 and combines it with workload context carried inside the `ResourceIntent`.

Output:

```text
SchedulingCandidateIntent
```

---

# 2. ResourceIntent Is the Work Item

There is no second independent workload-selection object.

The workload context is first-class inside `ResourceIntent`:

```text
ResourceIntent {
    intent_id
    intent_version
    resource_group_id

    action
    resource_requirements

    tenant_id
    queue_ref

    request_context {
        workflow_type
        workflow_id
        priority_class_ref
        submit_time
        deadline?
        earliest_start?
        runtime_profile_ref?
        workload_metadata?
    }
}
```

```text
ResourceRequirements = what resources are requested
request_context       = what workload/business process this request belongs to
                        and which scheduling semantics apply
```

Phase 4 orders `ResourceIntent`s directly.

---

# 3. SchedulingCandidateIntent

Phase 4 enriches the selected intent with the policy/accounting facts it already read.

```text
SchedulingCandidateIntent {
    candidate_id

    resource_intent

    scheduling_analysis {
        effective_priority
        queue_path

        entitlement {
            remaining_nominal
            borrowable_now
            lent_out_now
            reclaim_right
        }

        fair_share_state
        aging_state
        deadline_state
        ordering_reason
    }

    observed_policy_state {
        queue_policy_version
        queue_accounting_version
        fair_share_version

        queue_usage
        ancestor_usage
        borrow_lend_state
    }

    selected_at
}
```

This observed state is a snapshot, not a quota reservation.

---

# 4. Runtime Position

```text
Pending ResourceIntents
        |
        v
Phase 4 Candidate Selection
        |
        +-- Queue Policy DB
        +-- Queue Accounting DB
        +-- Fair Share
        +-- Priority / Age / Deadline
        |
        v
SchedulingCandidateIntent
        |
        v
Exact Capacity Analysis
        |
        +-- gang feasibility
        +-- resource availability
        +-- topology
        +-- fragmentation
        +-- reclaim / consolidation / preemption
        |
        v
SchedulerActionPlan
```

> **Phase 4 chooses demand. Later analysis materializes demand.**

---

# 5. Phase 4 Reads Exact Queue State

There is no cache-only restriction.

Phase 4 may read exact data from:

```text
Queue Policy Store
Queue Accounting DB
QuotaCharge / BorrowProvenance
FairShareState
Future commitments
```

The design boundary is responsibility, not data-access restriction.

---

# 6. One Logical Worker Per queue_id

Locked baseline:

> **One lease-backed logical scheduling worker per `queue_id`.**

```text
ResourceIntent SQL
      |
      | queue_id
      |
      +--> Online Worker
      +--> OfflineEval Worker
      +--> Training Worker
```

Each queue worker serializes ordering decisions for its queue.

---

# 7. Why queue_id, Not tenant_id

One tenant may have multiple policy-distinct queues:

```text
Search/Online
Search/OfflineEval
Search/Batch
```

So `queue_id` is the better policy and ownership boundary.

---

# 8. Queue Worker Lease

```text
QueueWorkerLease {
    queue_id
    worker_id
    lease_epoch
    expires_at
}
```

Only the current lease holder performs candidate selection for that queue.

Failure:

```text
worker dies -> lease expires -> new worker gets higher epoch -> rebuild state
```

---

# 9. Durable ResourceIntent Store

ResourceIntents live durably, e.g. SQL.

Conceptual lifecycle:

```text
PENDING
SCHEDULING
SATISFYING
BLOCKED
COMPLETED
CANCELLED
```

SQL is durable truth. Queue-local heaps/indexes are reconstructible acceleration state.

---

# 10. Candidate Index

```text
QueueCandidateIndex {
    queue_id
    ready_heap
    blocked_index
    aging_index
    deadline_index
    scheduling_reservation_state
}
```

The worker can rebuild it from durable intent state after failover.

---

# 11. Intent Ingestion

```text
ResourceIntent written to SQL
        |
        v
queue worker discovers it
        |
        v
materialize ordering metadata
        |
        v
insert into QueueCandidateIndex
```

Discovery may be CDC, event stream, or polling.

---

# 12. Phase 4 Inputs Per Intent

```text
request_context.priority_class_ref
request_context.submit_time
request_context.deadline
request_context.earliest_start
resource_requirements
queue/accounting snapshot
fair-share state
previous attempts
bypass count
blocking reason
```

---

# 13. Phase 3 Participation

Phase 4 reads exact Phase 3 state:

```text
remaining_nominal
remaining_hard_limit
borrowable_now
lent_out_now
reclaim_right
fair_share_state
ancestor usage
```

This is not a separate precomputed entitlement stage over every intent.

---

# 14. Candidate Selection Question

For each pending intent:

> **Given current queue policy, accounting, priority, age, deadline, and fairness state, how deserving and actionable is this ResourceIntent relative to the other pending intents in the queue?**

Then select one intent for exact capacity analysis.

---

# 15. Hard Eligibility

Examples of immediate blocking:

```text
intent terminal/superseded
earliest_start in future
queue CLOSED
hard policy violation
workflow dependency blocked
retry cooldown active
```

These go to a blocked index.

---

# 16. Queue-Accounting Eligibility

A candidate may be temporarily blocked because:

```text
hard limit reached
ancestor hard limit reached
borrowing prohibited
no shared excess
future reservation prevents borrowing
```

When accounting changes, wake it.

---

# 17. Ranking Inputs

Among eligible candidates:

```text
priority class
waiting time
deadline urgency
FIFO
fair-share position
nominal vs borrowed state
starvation / bypass count
scheduling reservation state
```

---

# 18. Priority Class

Platform-controlled bounded classes:

```text
P0_CRITICAL_RECOVERY
P1_ONLINE_PRODUCTION
P2_RESERVED_DEADLINE
P3_STANDARD
P4_BATCH
P5_OPPORTUNISTIC
```

Clients reference allowed classes via `request_context`.

---

# 19. FIFO, Aging, Deadline

FIFO is a tie-breaker, not the whole policy.

Aging:

```text
effective_priority = base_priority + bounded_aging(wait_time)
```

Deadline urgency:

```text
slack = deadline - now - estimated_runtime
```

Low slack increases urgency within allowed policy boundaries.

---

# 20. Fair Share in Candidate Selection

Phase 4 may directly compare exact quota/fair-share state.

Example:

```text
Intent A requires 20 borrowed H100
Intent B fits nominal headroom
```

Depending policy, B may rank ahead because it consumes guaranteed entitlement rather than contested shared excess.

---

# 21. Borrowing Example

Queue Q:

```text
borrowable_now = 25 H100
```

Pending:

```text
A needs 10 borrowed H100
B needs 20 borrowed H100
```

Both can look valid against the same observed state. If B wins due to priority/deadline/age, Phase 4 emits B.

A has not reserved 10 H100.

---

# 22. Preserve Observed State

If Phase 4 read:

```text
usage = 128
borrowable_now = 25
fair_share = 0.17
accounting_version = 742
```

embed those facts in `SchedulingCandidateIntent` so the next phase can reuse them.

---

# 23. Snapshot Semantics

The provided state is:

```text
observed truth at selection time
```

not:

```text
reserved entitlement
```

Final ownership commit still revalidates correctness-critical assumptions.

---

# 24. Cross-Queue Race

Online and OfflineEval workers may both read:

```text
20 H100 shared excess
```

and both select candidates.

That is acceptable.

At quota commit, only legal concurrent usage succeeds. The loser refreshes and replans.

```text
selection-time exact read = decision quality
commit-time atomic validation = correctness
```

---

# 25. Ordering Policy

Prefer explainable lexicographic ordering:

```text
1. hard eligibility
2. priority class
3. quota/fair-share state
4. deadline urgency
5. aging / bypass state
6. FIFO
7. deterministic intent_id tie breaker
```

Avoid one opaque global weighted score where possible.

---

# 26. Head-of-Line Blocking

```text
A needs 64 H100
B needs 8 H100
```

If A repeatedly fails exact capacity analysis, downstream feedback marks it blocked/bypassed so B can progress.

---

# 27. Starvation Prevention

Use:

```text
aging
bypass count
SchedulingReservation
```

for old/large intents.

---

# 28. SchedulingReservation

Internal anti-starvation construct:

```text
SchedulingReservation {
    intent_id
    protected_since
    required_resource_envelope
    expected_ready_horizon?
    backfill_deadline?
}
```

Different from public Phase 2 `RESERVE`.

---

# 29. Backfill

While an old/large intent is protected, lower-ranked work may temporarily use capacity if it will not delay the protected candidate.

Use runtime estimates, reclaimability, and safety margin.

---

# 30. Single Candidate Baseline

For the baseline design, each queue worker selects one `SchedulingCandidateIntent` per iteration.

This keeps shared-quota assumptions simple.

Later, a small K can be added for downstream parallelism.

---

# 31. Candidate Claim

Before handoff:

```text
PENDING -> SCHEDULING
```

via an atomic durable claim.

```text
SchedulingClaim {
    intent_id
    queue_id
    worker_id
    claim_epoch
    expires_at
    intent_version
}
```

---

# 32. Claim Semantics

A claim means one worker owns the scheduling attempt.

It does **not** reserve quota or GPUs.

---

# 33. Fencing Epoch

If worker A loses its queue lease and worker B takes over:

```text
A epoch = 51
B epoch = 52
```

writes from stale epoch 51 are rejected.

TTL gives liveness; epoch gives correctness.

---

# 34. Failover

New worker:

```text
1. acquire queue lease
2. read pending/scheduling intents from SQL
3. recover expired claims
4. rebuild QueueCandidateIndex
5. resume selection
```

---

# 35. Exact Capacity Analysis Feedback

Downstream may return:

```text
NO_FREE_CAPACITY
NO_TOPOLOGY_FIT
FRAGMENTATION_BLOCKED
RECLAIM_REQUIRED
PREEMPTION_REQUIRED
STALE_ACCOUNTING
TEMPORAL_BLOCK
```

Phase 4 then requeues, blocks, ages, reserves, or waits for an event.

---

# 36. Event-Driven Wakeup

Examples:

```text
QuotaReleased
SharedExcessChanged
ResourceReleased
NewCapacityAdded
ReservationExpired
DeadlineThresholdReached
QueuePolicyChanged
```

Avoid hot-loop polling.

---

# 37. Candidate State Machine

```text
PENDING
   |
   v
READY
   |
   | selected
   v
SCHEDULING
   |
   +-- temporary downstream failure -> BLOCKED / READY
   +-- capacity commit success       -> SATISFYING
   +-- worker failure                -> claim expiry -> READY
```

---

# 38. SQL as Durable Truth

Conceptual fields:

```text
intent_id
queue_id
state
intent_version
scheduling_owner
claim_epoch
claim_expires_at
created_at
updated_at
```

The local heap is not authoritative.

---

# 39. Determinism and Explainability

Selection should be deterministic for the same state.

Persist traces:

```text
SchedulingSelectionTrace {
    selected_intent_id
    candidate_count
    queue_policy_version
    accounting_version
    effective_priority
    nominal/borrow state
    fair_share state
    age
    deadline slack
    ordering_reason
}
```

---

# 40. Concurrency Model

Within queue:

```text
one logical ordering owner
```

Across queues:

```text
many workers concurrently
```

Shared accounting races are resolved at atomic commit.

---

# 41. Atomic Resource/Quota Commit

Downstream final commit revalidates:

```text
intent still current?
queue policy still valid?
quota still available?
borrow still legal?
physical resources still free?
```

Then logically atomically create:

```text
ResourceAllocated + QuotaCharge
```

If stale, refresh and requeue.

---

# 42. Production Components

```text
ResourceIntentStore
QueueWorkerLeaseManager
QueueSchedulingWorker
QueueCandidateIndex
SchedulingPolicyEngine
Phase3AccountingClient
SchedulingClaimManager
SchedulingReservationManager
SelectionAuditLog
```

---

# 43. Selection Algorithm

```text
select_next(queue_id, now):

    phase3 = accounting.get_queue_snapshot(queue_id)

    candidates = candidate_index.ready_candidates()

    for intent in candidates:
        if hard_blocked(intent, phase3, now):
            block(intent)
            continue

        key = compute_ordering_key(intent, phase3, now)

    winner = best candidate

    claim = try_claim(winner)
    if !claim.success:
        refresh and retry

    return SchedulingCandidateIntent(
        winner,
        phase3 observed state,
        ordering analysis
    )
```

Incremental heap/index maintenance avoids rescoring every candidate each cycle.

---

# 44. Core Invariants

1. Phase 4 orders `ResourceIntent`s, not a separate workload object.
2. Workload details are first-class `request_context` inside ResourceIntent.
3. Phase 4 may read exact Phase 3 queue/accounting state.
4. Phase 3 is not a mandatory pre-processing stage over every intent.
5. One lease-backed logical worker owns ordering per `queue_id`.
6. Different queue workers may select concurrently.
7. Shared quota races are resolved by atomic quota commit.
8. `SchedulingCandidateIntent` carries observed policy/accounting state forward.
9. Observed state is not a quota reservation.
10. Final ownership commit revalidates correctness-critical state.
11. Phase 4 chooses demand; later analysis chooses exact capacity.
12. Placement/capacity failure feeds back into ordering state.
13. Priority does not automatically imply preemption permission.
14. SQL is durable truth; queue-local indexes are reconstructible acceleration state.

---

# 45. Staff+ Coding Exercise

Implement:

```text
queue worker lease
durable ResourceIntent claim
per-queue priority index
priority + aging + deadline ordering
exact Phase 3 accounting reads
borrow/fair-share-aware ranking
SchedulingCandidateIntent creation
failover
capacity-analysis feedback
event-driven wakeup
```

Suggested APIs:

```text
class QueueSchedulingWorker:
    select_next(queue_id, now) -> SchedulingCandidateIntent | None

class Phase3AccountingClient:
    get_snapshot(queue_id) -> QueueSchedulingPolicySnapshot

class SchedulingClaimManager:
    try_claim(intent_id, worker_id, lease_epoch) -> ClaimResult
```

---

# 46. Interview Summary: Zoom Level 2

This is the version to review right before an interview.

> **Phase 4 is the scheduling-candidate selection layer. Given many pending `ResourceIntent`s in one queue, the queue worker reads exact queue/accounting state, filters out intents that cannot compete now, ranks the remaining intents using policy and workload context, atomically claims the winner, and emits a `SchedulingCandidateIntent` for expensive capacity analysis.**

The whole phase is:

```text
Pending ResourceIntents
        |
        v
Read exact Queue / Accounting state
        |
        v
Filter currently ineligible intents
        |
        v
Rank remaining intents
        |
        v
Pick highest-ranked ResourceIntent
        |
        v
Atomic Claim
        |
        v
SchedulingCandidateIntent
        |
        v
Exact Capacity Analysis
```

The key boundary is:

> **Phase 4 chooses the demand we will work on next. It does not choose GPUs.**

---

## 46.1 What Does Phase 4 Read?

The queue worker has pending `ResourceIntent`s:

```text
Intent A
Intent B
Intent C
...
```

Each intent already contains first-class workload context:

```text
ResourceIntent {
    intent_id
    action
    resource_requirements

    tenant_id
    queue_ref

    request_context {
        workflow_type
        priority_class_ref
        submit_time
        deadline?
        earliest_start?
        runtime_profile_ref?
    }
}
```

Phase 4 also reads exact Phase 3 state:

```text
Queue Policy

Queue Accounting DB {
    nominal_usage
    borrowed_usage
    lent_out_usage
    remaining_hard_limit
    ancestor_usage
}

Borrow / Lend state

Fair Share state

Future commitments / reservation constraints
```

So selection uses:

```text
request/workload context
+
current queue/accounting truth
```

---

## 46.2 Step 1 — Filter Intents That Cannot Compete Now

Before ranking, remove intents that clearly cannot be selected right now.

Examples:

```text
intent cancelled / superseded

earliest_start is in the future

queue is CLOSED

hard limit would be violated

ancestor queue limit would be violated

borrowing is required but borrowing is disabled

no shared excess currently exists

retry cooldown is active
```

These intents move to a blocked state/index and are reconsidered when relevant state changes.

The output of this step is:

```text
currently selectable ResourceIntents
```

---

## 46.3 Step 2 — Rank the Remaining ResourceIntents

Now ask:

> **Among the intents that can currently compete, which one deserves the next scheduling attempt?**

A clean interview ordering policy is lexicographic:

```text
1. Priority class

2. Entitlement / fair-share position
      nominal entitlement preferred
      versus borrowed/shared-excess demand

3. Deadline urgency

4. Aging / starvation state

5. FIFO submit time

6. Stable intent_id tie-breaker
```

Conceptually:

```text
OrderingKey(intent) = (
    priority_class_rank,
    entitlement_rank,
    deadline_urgency,
    aging_rank,
    submit_time,
    intent_id
)
```

The exact formula is policy-specific.

The important design choice is:

> **Use explainable policy tiers instead of one opaque global score.**

---

## 46.4 Concrete Selection Example

Suppose one queue has:

```text
nominal headroom:
    0 H100

borrowable shared excess:
    25 H100
```

Pending intents:

```text
Intent A
    needs 10 H100
    priority = STANDARD
    waiting = 30 min

Intent B
    needs 20 H100
    priority = ONLINE_PRODUCTION
    waiting = 5 min

Intent C
    needs 8 H100
    priority = STANDARD
    waiting = 2 min
    deadline is near
```

All three are currently individually policy-legal because:

```text
10 <= 25
20 <= 25
8  <= 25
```

That does **not** mean 38 H100 have been reserved.

It only means each can currently compete against this accounting snapshot.

The ordering may become:

```text
B
    highest priority class

C
    lower class but urgent deadline

A
    lower class, older
```

So:

```text
winner = Intent B
```

At this point Phase 4 is done comparing A/B/C for this scheduling attempt.

---

## 46.5 Step 3 — What Is the Atomic Claim?

After selecting B, the queue worker performs a durable transition:

```text
PENDING
    ->
SCHEDULING
```

Conceptually:

```text
SchedulingClaim {
    intent_id = B

    queue_id

    worker_id

    queue_worker_lease_epoch

    claim_epoch

    expires_at

    expected_intent_version
}
```

The claim succeeds only if:

```text
intent.state == PENDING

intent.version == expected_version

worker still owns the queue lease
```

If it succeeds:

```text
this worker owns the current scheduling attempt for B
```

If it fails:

```text
another worker/failover changed B
    ->
refresh and select again
```

### The claim does NOT reserve capacity

It does not reserve:

```text
quota

borrowable capacity

GPUs
```

It only prevents duplicate work on the same `ResourceIntent`.

The later quota/resource ownership commit remains the correctness boundary.

---

## 46.6 Step 4 — What Is SchedulingCandidateIntent?

After the claim succeeds, Phase 4 emits an enriched object:

```text
SchedulingCandidateIntent {
    candidate_id

    resource_intent {
        intent_id
        intent_version

        resource_group_id
        action
        resource_requirements

        tenant_id
        queue_ref

        request_context
    }

    scheduling_analysis {
        effective_priority

        queue_path

        entitlement {
            remaining_nominal
            borrowable_now
            lent_out_now
            reclaim_right
        }

        fair_share_state

        aging_state
        deadline_state

        ordering_reason
    }

    observed_policy_state {
        queue_policy_version
        queue_accounting_version
        fair_share_version

        queue_usage
        ancestor_usage
        borrow_lend_state
    }

    selected_at
}
```

Interpretation:

> **“This is the `ResourceIntent` we chose to work on next, and here are the exact policy/accounting facts that drove the decision.”**

It does not mean:

```text
capacity has been reserved

the request is guaranteed to succeed
```

---

## 46.7 Why Pass the Observed State Forward?

Suppose Phase 4 already read:

```text
Queue Q usage:
    128 H100

borrowable_now:
    25 H100

fair_share_score:
    0.17

accounting_version:
    742
```

Do not throw that away.

The next phase receives those facts inside the `SchedulingCandidateIntent`.

It can immediately begin exact capacity analysis:

```text
Intent B needs 20 H100

observed borrowable budget = 25

now determine:
    legal ResourceRequirement alternative
    exact free capacity
    gang feasibility
    topology
    fragmentation
    reclaim / consolidation / preemption options
```

The next phase reads additional fleet state only where needed.

---

## 46.8 Snapshot Semantics

The state passed forward is:

```text
observed state
```

not:

```text
reserved state
```

Example:

```text
Phase 4 observed:
    borrowable_now = 25
```

Another queue worker may consume 20 before this candidate commits.

Therefore final ownership commit still revalidates:

```text
quota still legal?

borrowable shared excess still exists?

physical resources still free?
```

So:

```text
Phase 4 exact read
    = best current decision input

Atomic ownership commit
    = correctness guarantee
```

---

## 46.9 Distributed Selection Model

Baseline:

```text
one lease-backed logical worker per queue_id
```

Example:

```text
Online queue
    -> Worker A

OfflineEval queue
    -> Worker B

Training queue
    -> Worker C
```

Within one queue:

```text
ordering is logically serialized
```

Across queues:

```text
workers run concurrently
```

Sibling workers may read the same parent shared-excess state optimistically.

Final quota/resource commit resolves races.

---

## 46.10 Candidate Index

The worker should not query and sort the entire SQL table for every decision.

Durable state lives in SQL.

The worker maintains reconstructible local indexes:

```text
QueueCandidateIndex {
    ready_heap

    blocked_by_reason

    deadline timers

    aging metadata

    scheduling reservation state
}
```

The ready heap is ordered by the Phase 4 ordering key.

---

## 46.11 Aging, Starvation, and Head-of-Line Blocking

Priority alone is insufficient.

Example:

```text
Intent A
    very large request
    repeatedly fails exact capacity analysis

Intent B/C/D
    small requests
    keep succeeding
```

If A is always bypassed:

```text
A can starve
```

Phase 4 tracks:

```text
waiting time
bypass count
```

and can eventually create an internal:

```text
SchedulingReservation
```

for A.

Safe temporary work may still backfill if it will not delay A.

---

## 46.12 Capacity-Analysis Feedback

Phase 4 chooses an intent.

The next phase may return:

```text
NO_FREE_CAPACITY

NO_TOPOLOGY_FIT

FRAGMENTATION_BLOCKED

RECLAIM_REQUIRED

TEMPORAL_BLOCK
```

Phase 4 uses that feedback to:

```text
requeue now

block until event

increase bypass count

age candidate

create scheduling reservation
```

So Phase 4 is a persistent scheduling loop, not a one-shot sorter.

---

## 46.13 Full Example End to End

Initial:

```text
ResourceIntent SQL:
    A, B, C are PENDING

Queue Q worker:
    owns queue lease epoch 51
```

Selection:

```text
1. Read Queue Accounting DB

   nominal headroom = 0
   borrowable_now = 25
   accounting_version = 742


2. Filter

   A legal
   B legal
   C legal


3. Rank

   B > C > A


4. Select B


5. Atomic claim

   B:
       PENDING -> SCHEDULING

   claim_epoch = 104


6. Build

   SchedulingCandidateIntent(B)


7. Pass B to exact capacity analysis
```

A and C remain pending.

If B later commits 20 borrowed H100:

```text
Queue Accounting DB:
    version 742 -> 743

borrowable:
    25 -> 5
```

When A or C is considered later, Phase 4 reads the new exact state.

---

## 46.14 Zoom Level 2 Diagram

```text
                         ResourceIntent SQL
                    A      B      C      ...
                     \     |     /
                      \    |    /
                       v   v   v
                  Queue Candidate Index
                           |
             +-------------+-------------+
             |                           |
             v                           v
      ResourceIntent Context       Phase 3 State
      priority / age /             Queue Accounting DB
      deadline / workflow          quota / borrow / lend
                                   fair share / hierarchy
             \                           /
              \                         /
               +-----------+-----------+
                           |
                           v
                    Eligibility Filter
                           |
                           v
                    Ordering Policy
              priority -> entitlement
              -> deadline -> aging -> FIFO
                           |
                           v
                    Select Intent B
                           |
                           v
                     Atomic Claim
                PENDING -> SCHEDULING
                           |
                           v
              SchedulingCandidateIntent
              {
                  ResourceIntent B
                  request_context
                  effective priority
                  entitlement snapshot
                  fair-share state
                  accounting versions
                  ordering reason
              }
                           |
                           v
                  Exact Capacity Analysis
              gang / free capacity / topology
              fragmentation / reclaim /
              consolidation / preemption
                           |
                           v
                  SchedulerActionPlan
```

---

## 46.15 Interview Sound Bite

> **“For each queue I have one logical scheduling worker. It reads the queue's exact accounting and fair-share state, filters pending ResourceIntents that cannot currently compete, and ranks the rest using platform priority, entitlement position, deadline, aging, and FIFO. It atomically claims the highest-ranked intent by moving it from PENDING to SCHEDULING, then emits a SchedulingCandidateIntent containing the selected ResourceIntent plus the exact policy/accounting snapshot that drove the decision. Only then do I spend expensive work on gang, topology, fragmentation, and disruption analysis.”**


# 47. Interview Summary: Zoom Level 3

At whole-system zoom, Phase 4 is one box:

> **Phase 4 reduces many pending `ResourceIntent`s to the next `SchedulingCandidateIntent` the physical-capacity scheduler should work on.**

It reads:

```text
ResourceIntent request_context
    priority
    age
    deadline
    workflow semantics

Phase 3 queue/accounting state
    quota
    borrow/lend
    hierarchy
    fair share
```

It performs:

```text
filter
    ->
rank
    ->
select
    ->
atomic claim
```

and emits:

```text
SchedulingCandidateIntent {
    selected ResourceIntent
    scheduling analysis
    observed queue/accounting snapshot
    versions
}
```

The claim means:

```text
this worker owns the current scheduling attempt
```

not:

```text
quota or GPUs are reserved
```

Single-box view:

```text
        Pending ResourceIntents
                 |
                 v
      +--------------------------+
      |         PHASE 4          |
      | Scheduling Candidate     |
      | Selection                |
      |                          |
      | reads:                   |
      | - request_context        |
      | - Queue Accounting DB    |
      | - fair share             |
      |                          |
      | does:                    |
      | - filter                 |
      | - rank                   |
      | - select                 |
      | - atomic claim           |
      +------------+-------------+
                   |
                   v
        SchedulingCandidateIntent
                   |
                   v
          Exact Capacity Analysis
```

One-sentence version:

> **“Phase 4 picks the next ResourceIntent to work on by combining exact queue entitlement/fair-share state with workload priority, age, and deadline; it atomically claims that intent and passes an enriched SchedulingCandidateIntent to physical capacity analysis.”**

The full-system boundary is:

```text
Phase 3:
    maintain exact entitlement/accounting truth

Phase 4:
    choose which demand goes next

Next phase:
    determine whether and how that selected demand
    can be physically materialized
```
