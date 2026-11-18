# Phase 3 — Queue Accounting and Entitlement Policy Subsystem

## 1. Purpose

Phase 3 is the **authoritative queue/accounting and entitlement-policy subsystem** used by the scheduling decision engine.

Its responsibility is:

> **Maintain and expose the exact policy/accounting state required to answer who is entitled to how much capacity, what is borrowed/lent, and how shared excess should be divided.**

Phase 4 reads this subsystem directly while selecting the next `ResourceIntent`.

```text
ResourceIntent SQL
      |
      v
Phase 4 Scheduling Candidate Selection
      |
      +---- reads Phase 3 Queue Policy
      +---- reads Phase 3 Queue Accounting DB
      +---- reads borrow/lend + fair-share state
      |
      v
SchedulingCandidateIntent
```

Phase 3 provides authoritative state, derived policy state, query APIs, and commit APIs. It does **not** independently choose which pending intent should run next.

---

# 2. Core Mental Model

Phase 3 owns four concerns:

```text
1. Queue hierarchy
2. Quota accounting
3. Borrow / lend provenance
4. Fair-share state
```

The central component is:

```text
Queue Accounting DB
    = authoritative scheduler-owned accounting
      for current entitlement usage
```

It answers:

```text
How much quota does this queue own?
How much is currently consumed?
How much is borrowed?
How much is lent out?
Which ResourceAllocated objects produced those charges?
What ancestor limits apply?
What is the current fair-share position?
```

---

# 3. Queue Hierarchy

```text
root
 |
 +-- Search
 |    |
 |    +-- GenAI
 |    |    +-- Online
 |    |    +-- OfflineEval
 |    |
 |    +-- Relevance
 |
 +-- Training
      +-- Foundation
      +-- FineTune
```

A request belongs to one leaf queue, e.g.:

```text
Search/GenAI/Online
```

but policy is enforced along:

```text
Online -> GenAI -> Search -> root
```

> **A child queue cannot violate any ancestor quota, hard limit, or sharing policy.**

---

# 4. Queue Policy

```text
QueuePolicy {
    queue_id
    parent_queue_id?

    state: OPEN | DRAINING | CLOSED

    quotas[]: QuotaPolicy

    fair_share_policy
    admission_policy

    policy_version
}
```

Clients supply `queue_ref`; they do not author quota or fair-share weights.

---

# 5. Quota Policy

```text
QuotaPolicy {
    resource_key

    nominal_quota
    hard_limit?
    borrowing_limit?
    lending_limit?
    fair_share_weight?

    policy_version
}
```

Semantics:

```text
nominal_quota  = baseline reclaimable entitlement
hard_limit     = maximum concurrent usage
borrowing_limit= maximum shared excess usage
lending_limit  = maximum unused nominal entitlement exposed to siblings
```

---

# 6. Resource Dimensions

```text
ResourceKey {
    resource_type: GPU | CPU | HOST_MEMORY | RDMA_NIC | NVME | ...
    flavor?
    pool?
}
```

Examples:

```text
GPU/H100_SXM_80GB
GPU/H200_SXM_141GB
CPU/x86_64
```

```text
ResourceVector {
    ResourceKey -> quantity
}
```

Do not silently equate H100 and H200. Phase 2 alternatives define legal fungibility.

---

# 7. Nominal Quota

```text
Search nominal quota = 128 H100
```

means Search has baseline entitlement up to 128 H100. It does not mean 128 exact GPUs are permanently reserved.

Unused entitlement may be lendable.

---

# 8. Hard Limit

```text
nominal = 128
hard_limit = 192
```

means:

```text
0..128   normal entitlement
129..192 shared excess / borrowing only
>192     forbidden
```

Ancestor limits still apply.

---

# 9. Borrowing

Borrowing is derived accounting state, not a client action.

A queue may exceed nominal quota only when:

```text
borrowing is allowed
shared excess exists
ancestor limits allow it
fair-share policy permits it
```

Example:

```text
Training nominal = 128
Training usage   = 128
borrowable       = 32

Training gets +16
=> borrowed_usage = 16
```

---

# 10. Lending

Example:

```text
OfflineEval nominal = 100
OfflineEval usage   = 40
unused              = 60
```

If policy permits lending, some or all of that 60 becomes shared excess at the parent.

```text
OfflineEval unused entitlement
      |
      v
GenAI shared excess pool
      |
      v
Online may borrow
```

This is logical policy capacity, not physical GPU ownership.

---

# 11. Hierarchical Borrowing

Borrowing occurs within a sharing scope.

```text
Search
 |
 +-- GenAI
 |    +-- Online
 |    +-- OfflineEval
 |
 +-- Relevance
```

Unused OfflineEval quota can become shared excess within GenAI. Recursive sharing can happen at higher ancestors according to policy.

---

# 12. Fair Share

Fair share answers:

> **When multiple sibling queues want shared excess, who should receive the next opportunity?**

Recommended policy order:

```text
1. Restore under-nominal entitlement.
2. Distribute shared excess fairly.
3. Enforce hard/borrowing limits throughout.
```

---

# 13. Dominant Resource Fairness

For excess usage:

```text
share(q,r) = excess_usage(q,r) / shared_capacity(r)

dominant_share(q) = max_r share(q,r)

weighted_share(q) = dominant_share(q) / queue_weight(q)
```

Lower weighted dominant share generally gets the next shared opportunity.

---

# 14. Recursive Fair Share

```text
root:          Search vs Training
inside Search: GenAI vs Relevance
inside GenAI:  Online vs OfflineEval
```

This avoids a giant global DRF calculation.

---

# 15. Queue Accounting DB

This database is on the critical scheduling path.

```text
QueueUsageState {
    queue_id

    usage_by_resource
    nominal_usage_by_resource
    borrowed_usage_by_resource
    lent_out_usage_by_resource
    reserved_usage_by_resource
    descendant_usage_by_resource

    usage_version
    updated_at
}
```

Phase 4 reads this exact state while comparing candidates.

---

# 16. Per-Allocation Accounting Provenance

```text
QuotaCharge {
    resource_allocated_id

    consumer_queue_id
    resource_key
    quantity

    charge_type: NOMINAL | BORROWED | RESERVED

    sharing_scope_queue_id?
    entitlement_source_queue_id?

    policy_epoch
}
```

This answers who is borrowing, how much, and which concrete allocations created the charge.

---

# 17. Why Provenance Is Written at Allocation Time

> **Borrowing is explicit accounting provenance, not inferred telemetry.**

The scheduler already knows why an allocation was admitted, so write the `QuotaCharge` when `ResourceAllocated` ownership commits.

---

# 18. Allocation Lifecycle Drives Accounting

```text
ALLOCATE
  -> ResourceAllocated
  -> QuotaCharge
  -> QueueUsageState increment
```

```text
RELEASE / MARK_LOST
  -> close/remove QuotaCharge
  -> QueueUsageState decrement
```

Quota usage follows ownership, not GPU utilization.

---

# 19. Reclaim Rights

Example:

```text
Search nominal = 128
Search usage   = 96
Batch borrowed = 24
Search new demand = +32
```

Phase 3 can derive:

```text
reclaim_right = up to 24 H100
```

It does not choose exact victims. Concrete victim selection remains later resource-transition analysis.

---

# 20. Borrow Provenance

```text
BorrowProvenance {
    borrower_queue_id
    sharing_scope_queue_id
    resource_key
    quantity
    entitlement_source_queue_id?
    admitted_at
    reclaim_deadline_policy?
}
```

---

# 21. Future Commitments

```text
FutureQuotaCommitment {
    queue_id
    resource_vector
    valid_from
    valid_until
    source_reservation_id
}
```

Future commitments can reduce lendable capacity before the deadline.

---

# 22. Policy Query APIs

```text
get_queue_policy(queue_id)
get_queue_path(queue_id)
get_queue_usage(queue_id)
get_hierarchy_usage(queue_id)
get_shared_excess(parent_queue_id)
get_fair_share_state(parent_queue_id)
get_borrow_provenance(queue_id)
get_reclaim_right(queue_id, resource_vector)
```

---

# 23. Combined Scheduling Policy Snapshot

```text
QueueSchedulingPolicySnapshot {
    queue_id
    queue_path[]

    usage_state

    remaining_nominal
    remaining_hard_limit
    borrowable_now
    lent_out_now
    reclaim_right
    fair_share_state

    policy_version
    accounting_version
    fair_share_version
}
```

This is an exact observed snapshot, not a reservation.

---

# 24. Phase 3 Boundary

Phase 3 does **not** choose which `ResourceIntent` runs next and does **not** choose exact GPUs.

```text
Phase 3 = policy/accounting truth
Phase 4 = choose next demand
Later   = materialize exact capacity
```

---

# 25. Queue Worker Model

Locked baseline:

> **One lease-backed logical scheduling worker per `queue_id`.**

```text
ResourceIntent SQL
      |
      +--> Online worker
      +--> OfflineEval worker
      +--> Training worker
```

Each worker owns ordering for its queue and reads shared Phase 3 accounting state.

---

# 26. Queue Worker Lease

```text
QueueWorkerLease {
    queue_id
    worker_id
    lease_epoch
    expires_at
}
```

On failure, another worker acquires a higher epoch and rebuilds state.

---

# 27. Cross-Queue Concurrency

Sibling workers may read the same shared-excess state concurrently.

That is acceptable.

```text
selection = optimistic
quota consumption = atomic
```

If stronger sibling arbitration is needed later, add it then; do not start with global locks.

---

# 28. Atomic Quota Commit

```text
atomic {
    verify leaf + ancestor limits
    verify borrowing still legal
    verify shared excess still available
    update QueueUsageState
    create QuotaCharge
}
```

If stale, fail and replan.

---

# 29. Consistency Model

Strong / authoritative:

```text
QueuePolicy
QueueUsageState
QuotaCharge
FutureQuotaCommitment
quota commit
```

Derived / recomputable:

```text
fair-share score
historical fairness
queue pressure
```

---

# 30. Scalability

Do not scan all allocations or queues per query.

Maintain aggregates and indexes so cost depends on:

```text
queue depth
resource dimensions
sibling count
```

not total fleet size.

---

# 31. Production Components

```text
QueuePolicyStore
QueueAccountingDB
QuotaChargeStore
HierarchyUsageIndex
BorrowLendCalculator
FairShareEngine
FutureCommitmentIndex
QuotaCommitter
AccountingAuditLog
```

---

# 32. Reconciliation

Periodically compare:

```text
ResourceAllocated registry
vs
QuotaCharge store
```

Detect and repair missing/orphan/wrong charges idempotently.

---

# 33. Core Invariants

1. Phase 3 owns queue/accounting truth, not candidate ordering.
2. Quota follows scheduler ownership, not utilization telemetry.
3. Every allocation has explicit quota provenance.
4. Borrowed usage is identifiable without reconstructing history.
5. Child queues cannot bypass ancestor policy.
6. Nominal quota is reclaimable entitlement, not dedicated hardware.
7. Shared excess is governed by borrow/lend policy and fair share.
8. Phase 4 may read exact Phase 3 state directly.
9. Concurrent queue workers may make optimistic decisions, but quota commit is atomic.
10. Exact GPU placement and concrete preemption victims are outside Phase 3.
11. Fair-share state is derived; accounting state is authoritative.

---

# 34. Staff+ Coding Exercise

Implement:

```text
hierarchical queue accounting
borrow/lend calculation
weighted DRF
atomic quota commit
per-allocation provenance
queue worker lease
reconciliation
```

Suggested APIs:

```text
class QueueAccountingService:
    get_snapshot(queue_id) -> QueueSchedulingPolicySnapshot
    try_commit(charge, expected_version) -> CommitResult

class FairShareEngine:
    get_state(parent_queue_id) -> FairShareState
```

---

# 35. Interview Summary: Zoom Level 2

> **Phase 3 is the queue-accounting and entitlement-policy subsystem. It maintains hierarchical quotas, exact current usage, borrowing/lending provenance, and fair-share state. Phase 4 reads this exact state while choosing the next ResourceIntent.**

Critical pieces:

```text
Queue Policy Store
Queue Accounting DB
Borrow/Lend + Fair Share
QuotaCharge provenance
```

The Queue Accounting DB is on the critical path because candidate selection needs exact answers such as current usage, borrowable capacity, lent-out capacity, parent limits, and fair-share position.

## 35.1 Zoom Level 2 Diagram

```text
                     Queue Policy Store
                            |
                            v
                  +---------------------+
                  |      PHASE 3        |
                  | Queue Accounting /  |
                  | Entitlement Policy  |
                  +----------+----------+
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
   Queue Accounting DB                 Fair Share Engine
            |
            +-- nominal usage
            +-- borrowed usage
            +-- lent-out usage
            +-- ancestor usage
            +-- QuotaCharge provenance
            |
            v
 QueueSchedulingPolicySnapshot
            |
            v
      Phase 4 Candidate Selection
```

## 35.2 Hierarchical Borrowing Example

```text
                     Search
                       |
              +--------+--------+
              |                 |
           GenAI            Relevance
              |
        +-----+------+
        |            |
     Online      OfflineEval
```

```text
Online nominal = 300
Online usage   = 300

OfflineEval nominal = 100
OfflineEval usage   = 40
```

OfflineEval has 60 unused. If lending is allowed, GenAI exposes up to 60 shared excess. Online may borrow from that scope. If multiple siblings compete for excess, weighted fair share decides the next opportunity.

## 35.3 Key Design Decisions

```text
Quota != utilization
Nominal quota != dedicated hardware
Borrowing is explicit accounting
Fair share applies to shared excess
Borrow provenance is stored at allocation time
Queue hierarchy is enforced root-to-leaf
Quota commit is atomic
```

## 35.4 Zoom Level 2 Contract

Input/query:

```text
queue_id
resource vector
```

Output:

```text
QueueSchedulingPolicySnapshot {
    remaining_nominal
    borrowable_now
    lent_out_now
    reclaim_right
    fair_share_state
    versions
}
```

---

# 36. Interview Summary: Zoom Level 3

> **Phase 3 is the authoritative queue-policy/accounting box. It exposes exact quota, borrowing/lending, and fair-share state to candidate-selection logic.**

```text
                  +----------------------+
                  |      PHASE 3         |
                  | Queue Accounting &   |
                  | Entitlement Policy   |
                  | quotas / hierarchy   |
                  | borrow / lend        |
                  | fair share           |
                  | accounting DB        |
                  +----------+-----------+
                             |
                             v
                  Phase 4 Candidate Selection
```

One sentence:

> **I keep queue entitlement in an authoritative accounting subsystem that tracks nominal, borrowed, lent-out, and hierarchical usage; candidate-selection workers read that exact state, while final quota consumption is atomically committed.**
