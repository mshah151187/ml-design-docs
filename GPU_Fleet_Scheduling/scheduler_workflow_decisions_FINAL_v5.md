# GPU Fleet Scheduler — Public Actions to Internal Workflow Decisions

## 1. Purpose

This document defines how the GPU Fleet Scheduler translates its **public resource actions** into one or more **internal scheduling actions**.

Public actions describe the outcome an upstream actor wants:

```text
ACQUIRE
RESIZE
RELEASE
RESERVE
REPLACE
```

Internal actions are scheduler mechanisms:

```text
ALLOCATE
RECLAIM
PREEMPT
CONSOLIDATE
BACKFILL
PROTECT_CAPACITY
CONTROLLER_DRAIN
CONTROLLER_WAIT_FOR_RELEASE
MARK_LOST
RELEASE
```

The key principle is:

> **A public action is not a direct internal command. It enters a policy and feasibility decision process that produces an internal action plan.**

Example:

```text
ACQUIRE 4 TP8-H100 instances
```

may resolve to:

```text
ALLOCATE
```

or:

```text
RECLAIM -> CONTROLLER_WAIT_FOR_RELEASE -> ALLOCATE
```

or:

```text
PREEMPT -> CONTROLLER_DRAIN -> CONTROLLER_WAIT_FOR_RELEASE -> ALLOCATE
```

or:

```text
CONSOLIDATE -> ALLOCATE
```

or:

```text
PENDING
```

depending on quota, priority, reservations, topology, fragmentation, and disruption policy.

---

# 2. Scheduler Decision Boundary

```text
ResourceIntent
     |
     v
Intent Validation
     |
     v
Entitlement / Admission Resolution
     |
     v
Feasibility Analysis
     |
     v
Existing Ownership / Borrowing / Reservation Analysis
     |
     v
Disruption / Reclaim / Consolidation Policy
     |
     v
SchedulerActionPlan
     |
     v
Atomic Resource State Transitions
     |
     v
ResourceIntentStatus
```

The scheduler decides **what resource transitions should happen**.

Domain controllers decide **how application state safely moves**.

The allocation subsystem commits **exclusive ownership changes**.

---

# 3. Public Actions

```text
ACQUIRE
    Obtain additional resource instances.

RESIZE
    Converge a ResourceGroup to a desired instance count.

RELEASE
    Relinquish selected `ResourceAllocated` objects or an entire `ResourceAllocation`.

RESERVE
    Protect resource capacity for a time window.

REPLACE
    Obtain replacement capacity for an existing/lost `ResourceAllocated`.
```

---

# 4. Canonical Internal Scheduler Actions

This decision engine uses exactly the same internal-action vocabulary as the Phase 2 design:

```text
ALLOCATE
RECLAIM
PREEMPT
CONSOLIDATE
BACKFILL
PROTECT_CAPACITY
RELEASE
MARK_LOST
```

## 4.1 ALLOCATE
Assign physical resources to one logical instance and create a `ResourceAllocated`.

## 4.2 RECLAIM

> **Restore an entitlement by identifying borrowed/lent capacity that must be returned to its entitled owner.**

`RECLAIM` is not itself the physical eviction primitive.

It is an entitlement-driven decision that answers:

```text
Which borrowed allocations must give capacity back?
How much capacity must be recovered?
By what deadline?
```

If the borrower does not release voluntarily, reclaim may result in one or more separate `PREEMPT` actions.

Conceptually:

```text
RECLAIM entitlement deficit
    |
    v
identify borrowed ResourceAllocated victims
    |
    +-- borrower can release gracefully
    |       -> controller drain/checkpoint
    |       -> RELEASE
    |
    +-- borrower does not release / deadline reached
            -> PREEMPT victim
            -> controller grace if policy allows
            -> RELEASE
```

After reclaimed capacity becomes free, some *other* pending scheduling decision may later issue:

```text
ALLOCATE
```

to the entitled owner.

`ALLOCATE` is therefore **not part of the semantic definition of RECLAIM**.

## 4.3 PREEMPT

> **Select/mark a specific `ResourceAllocated` for displacement because effective scheduler policy permits that allocation to be evicted.**

`PREEMPT` is the scheduler's eviction decision primitive.

It does **not** imply who receives the freed capacity afterward.

Its workflow is only:

```text
PREEMPT ResourceAllocated
    |
    v
optional controller grace / drain / checkpoint
    |
    v
RELEASE
```

Once the resources are free, the scheduler performs a separate scheduling cycle.

That later cycle may issue:

```text
ALLOCATE
```

for whichever pending demand wins admission, entitlement, priority, and placement.

Therefore:

```text
PREEMPT != PREEMPT + ALLOCATE
```

and:

```text
RECLAIM
    may choose PREEMPT as a mechanism

PREEMPT
    does not imply RECLAIM
```

Examples:

```text
Quota reclaim:
    RECLAIM
        -> PREEMPT borrower if necessary
        -> RELEASE
```

```text
Priority eviction:
    PREEMPT low-priority allocation
        -> RELEASE
```

The reason for the eviction differs even though the same `PREEMPT` primitive may be used.

## 4.4 CONSOLIDATE

> **Rearrange movable `ResourceAllocated` objects so that fragmented free capacity becomes a useful feasible placement.**

`CONSOLIDATE` is not itself one atomic ownership mutation. It is a scheduler optimization decision that produces a bounded movement/restart plan.

Typical trigger:

```text
pending demand:
    1 × TP8 H100 ResourceInstance

fleet:
    16 H100 free in aggregate

but:
    only 2 free GPUs in each NVSwitch domain
```

The scheduler may discover that moving a few flexible allocations could produce one clean 8-GPU domain.

Typical steps:

```text
CONSOLIDATE
    |
    v
1. Identify blocked high-value ResourceRequirements
    |
    v
2. Verify fragmentation is the actual blocker
       aggregate capacity is sufficient
       but no legal topology placement exists
    |
    v
3. Find movable ResourceAllocated candidates
       low disruption cost
       restartable / checkpointable
       placement alternatives exist
    |
    v
4. Generate a bounded set of consolidation plans
       which allocations move?
       where can they move?
       what useful domain becomes free?
    |
    v
5. Score plans
       disruption cost
       movement/startup cost
       topology value created
       SLA risk
       number of victims/moves
    |
    v
6. Select a plan only if benefit exceeds threshold
    |
    v
7. Materialize destination capacity for moved work if needed
       -> ALLOCATE replacement ResourceAllocated
    |
    v
8. Controller performs restart/checkpoint/handoff
    |
    v
9. RELEASE old ResourceAllocated
    |
    v
10. Re-evaluate the originally blocked demand
        -> separate ALLOCATE if it now wins scheduling
```

A simpler restart-based consolidation may instead be:

```text
CONSOLIDATE
    ->
select movable allocations
    ->
controller checkpoint/stop
    ->
RELEASE old placements
    ->
re-schedule those requirements elsewhere
```

Important:

```text
CONSOLIDATE
    does not imply that the originally blocked demand
    automatically receives the newly created capacity
```

Once capacity becomes free, normal admission/priority/fair-share ordering is applied again unless the implementation explicitly reserves the resulting domain for the triggering plan.

The scheduler should avoid continuous defragmentation. Consolidation should be bounded by:

```text
minimum fragmentation benefit
maximum disruption budget
cooldown / hysteresis
maximum moves per cycle
```


## 4.5 BACKFILL

> **Use temporarily idle capacity for work that can safely vacate before a known future capacity need.**

`BACKFILL` is an opportunistic utilization decision.

Typical trigger:

```text
reservation:
    128 H100 required at 09:00

current time:
    06:00

capacity:
    idle right now
```

Rather than leaving the GPUs unused for three hours, the scheduler may place suitable short or reclaimable work.

Typical steps:

```text
BACKFILL
    |
    v
1. Identify temporarily idle capacity with a future commitment
       reservation
       gang accumulation
       maintenance deadline
    |
    v
2. Compute reclaim deadline
       future_need_time
       - drain/checkpoint/startup safety margin
    |
    v
3. Find candidate ResourceRequirements
       expected to finish before deadline
       or policy allows reclaim/preemption
       compatible with the temporary capacity
    |
    v
4. Evaluate risk
       runtime uncertainty
       checkpoint/drain time
       startup variance
       reservation criticality
    |
    v
5. Select safe backfill demand
    |
    v
6. ALLOCATE ResourceAllocated
       mark provenance/policy as BACKFILL
       record reclaim deadline
    |
    v
7. Run normally while capacity is temporarily available
    |
    v
8. Before reclaim deadline:
       if work completed
           -> RELEASE normally

       else if future capacity is now needed
           -> RECLAIM backfill allocation
           -> optional PREEMPT if it misses release deadline
           -> RELEASE
```

Backfill therefore combines:

```text
temporary ALLOCATE now
+
future reclaimability contract
```

The backfilled client does not decide that it is protected from reclaim. Its effective policy comes from the scheduler/control plane.

A good safety invariant is:

> **Backfill must never make the protected future commitment less likely to be satisfied.**


## 4.6 PROTECT_CAPACITY

> **Prevent a capacity/topology envelope from being consumed by unrelated future allocations because it is committed to a protected demand.**

`PROTECT_CAPACITY` is primarily used for reservations, but can also support gang accumulation or planned maintenance/capacity handoff.

There are two useful forms.

```text
Abstract protection:
    reserve "8 H100 in one NVSwitch domain"
    without choosing exact GPUs yet

Concrete protection:
    reserve node42/gpu0..7 specifically
```

Typical steps:

```text
PROTECT_CAPACITY
    |
    v
1. Resolve protected ResourceRequirements
       required instance count
       acceptable ResourceRequirement alternatives
       topology constraints
       time window
    |
    v
2. Determine protection mode
       abstract envelope
       or concrete resources
    |
    v
3. Check conflicts
       existing reservations
       higher guarantees
       maintenance
       already-owned allocations
    |
    v
4. If capacity is currently free:
       mark capacity/envelope protected
    |
    +-- if capacity is currently borrowed/backfilled:
    |       schedule RECLAIM before protection deadline
    |
    v
5. Exclude protected capacity from unrelated admission/placement
    |
    v
6. At consumption time:
       matching ACQUIRE/RESIZE may allocate against the protection
    |
    v
7. On reservation expiry/cancellation:
       remove protection
       re-evaluate pending demand
```

For a future reservation, the system may intentionally combine:

```text
PROTECT_CAPACITY abstractly
    ->
BACKFILL temporarily
    ->
RECLAIM before deadline
    ->
PROTECT_CAPACITY concretely
```

`PROTECT_CAPACITY` does not itself create a running `ResourceAllocated`. It changes what future allocation decisions may consume.


## 4.7 RELEASE

> **End scheduler ownership of one or more `ResourceAllocated` objects and return their physical resources to schedulable capacity.**

`RELEASE` is the actual ownership-ending resource transition.

Typical client-initiated flow:

```text
Public RELEASE
    |
    v
1. Validate requester owns the ResourceAllocated
    |
    v
2. Check current allocation state
       already RELEASED -> idempotent success
       LOST -> finalize lost cleanup
       ALLOCATED -> continue
    |
    v
3. If runtime-safe shutdown is required:
       mark release pending / RELEASING
       notify owning controller
    |
    v
4. Controller performs domain-specific work
       serving: drain requests
       training: checkpoint/stop ranks
       batch: finish/requeue shard
    |
    v
5. Controller confirms releasable
    |
    v
6. Atomically remove ownership
    |
    v
7. Mark ResourceAllocated RELEASED
    |
    v
8. Return physical resources to schedulable pool
    |
    v
9. Re-evaluate:
       pending ACQUIRE
       pending RESIZE-up
       waiting gangs
       reservation gaps
```

Internal scheduler workflows also eventually use `RELEASE` after:

```text
RECLAIM
PREEMPT
CONSOLIDATE
REPLACE
maintenance evacuation
lease expiry
```

Important invariant:

> **A resource is not considered free merely because the scheduler intends to release it. It becomes free only after the ownership transition commits.**


## 4.8 MARK_LOST

> **Record that a `ResourceAllocated` can no longer be treated as usable because some underlying physical resource disappeared or failed.**

Typical triggers:

```text
node failure
GPU XID / fatal fault
NIC/fabric isolation making the allocation unusable
allocation lease loss
control-plane confirmation that worker is gone
```

Typical steps:

```text
MARK_LOST
    |
    v
1. Receive authoritative/credible failure signal
    |
    v
2. Mark affected physical resources non-schedulable if appropriate
    |
    v
3. Find ResourceAllocated objects using those resources
    |
    v
4. Transition each affected ResourceAllocated:
       ALLOCATED -> LOST
    |
    v
5. Remove it from healthy/effective allocated count
    |
    v
6. Notify owning workflow controller
       runtime is gone / no graceful drain possible
    |
    v
7. Recompute ResourceAllocation status
       desired = 50
       healthy allocated drops 50 -> 49
    |
    v
8. If ResourceRequirements remain under-satisfied:
       derive new pending capacity demand
    |
    v
9. Normal scheduling cycle may later:
       ALLOCATE replacement capacity
```

`MARK_LOST` is deliberately different from `RELEASE`.

```text
RELEASE:
    orderly ownership termination

MARK_LOST:
    ownership/runtime is invalidated by failure
```

There may be separate cleanup for stale leases, processes, or device state, but the scheduler must stop counting the lost allocation toward satisfied capacity immediately once the loss is authoritative.

`MARK_LOST` also does **not** semantically include `ALLOCATE`.

Replacement capacity is handled by a separate scheduling decision.


# 4A. Controller Workflow Steps

The following names may appear in decision flows, but they are **not scheduler actions**:

```text
CONTROLLER_DRAIN
CONTROLLER_CHECKPOINT
CONTROLLER_CONTROLLER_WAIT_FOR_RELEASE
CONTROLLER_WAIT_FOR_READY
CONTROLLER_HANDOFF
```

They are coordination steps executed by the owning workflow controller/runtime.


# 5. Shared Decision Inputs

Every action is resolved using a common decision context.

```text
DecisionContext {
    normalized_resource_intent

    resource_group_state
    existing_allocations

    FleetResourceSnapshot
    FleetPerformanceIndex?

    queue_state
    quota_state
    fair_share_state

    reservation_state
    borrowing_lending_state

    priority_state

    resolved_disruption_capabilities
    effective_disruption_policies

    pending_demand

    current_time
}
```

---

# 5A. Structure Assumptions Used by This Decision Engine

This decision engine assumes the final Phase 2 model:

```text
ResourceGroup
    -> ResourceRequirements
         -> primary_requirement: ResourceRequirement
         -> alternative_requirements[]: ResourceRequirement
         -> min_instances
         -> desired_instances
         -> max_instances
         -> partial_satisfaction_allowed
         -> cross_instance_constraints
         -> TemporalRequirements?

ResourceRequirement
    -> ResourceInstance
    -> RuntimeTemporalProfile?

ResourceAllocation
    -> aggregate progress for one ResourceRequirements
    -> resources_allocated[]: ResourceAllocated

ResourceAllocated
    -> one concrete physical allocation satisfying
       one logical ResourceInstance copy
```

Important:

```text
ResourceInstance
    is client/platform-defined.

ResourceAllocated
    is scheduler-created.

ResourceAllocation
    is aggregate scheduler status.
```

The scheduler may choose the primary `ResourceRequirement` or a declared alternative, but it may not invent a new realization.

Disruption semantics are also split:

```text
DisruptionCapabilities
    = resolved from trusted workflow/runtime metadata

DisruptionPolicy
    = derived by scheduler/control-plane policy
```

The requesting client does not directly set `preemption_allowed`, `reclaim_allowed`, or disruption criticality.

---

# 6. Shared Helper Predicates

The algorithms below use the following conceptual predicates.

```text
is_valid(intent)

is_supported(requirements)

is_admitted(intent)

is_within_entitlement(intent)

free_feasible_capacity(requirements, count)

borrowed_capacity_reclaimable_for(intent)

preemption_candidates(intent)

consolidation_can_create_fit(requirements)

intent.resource_requirements.partial_satisfaction_allowed

reservation_conflict(requirements, time_window)

safe_backfill_candidate(intent, reservation)

replacement_before_release_required(intent)

allocation_is_owned_by(intent.owner)

allocation_is_releasable(allocation)

allocation_is_lost(allocation)
```

These are conceptual, not fixed APIs.

---

# 7. Scheduler Decision Brain

The scheduler's central reasoning path is not simply:

```text
ResourceIntent
    ->
find free GPUs
```

It is a layered decision process.

The scheduler must answer three increasingly expensive questions:

```text
1. Is this demand entitled to consume capacity?

2. Can it be satisfied non-disruptively?

3. If not, what is the cheapest legal resource transition
   that can create a feasible placement?
```

This section is the canonical mental model for the scheduler decision engine.

---

## 7.1 Stage 1 — Ownership / Entitlement Analysis

The first question is:

> **Is this ResourceRequirements entitled to consume the requested capacity now?**

This analysis uses policy state such as:

```text
tenant
queue
quota
guaranteed / nominal quota
hard limits
fair share
borrowing
lending
priority/admission
reservations
```

The output is not necessarily an exact GPU assignment.

It is an entitlement classification such as:

```text
IN_QUOTA
BORROWING_ALLOWED
NEEDS_RECLAIM
OVER_QUOTA
WAIT_FOR_FAIR_SHARE
ADMISSION_BLOCKED
```

Example:

```text
Search guaranteed quota:
    128 H100

Search currently owns:
    96 H100

Search requests:
    +32 H100

Result:
    request is within guaranteed entitlement
```

However, those 32 H100 may currently be lent to another tenant.

Therefore:

```text
entitled capacity
    !=
currently free physical capacity
```

Ownership/entitlement analysis determines **who is allowed to consume capacity**.

Exact physical placement remains a later decision.

---

## 7.2 Stage 2 — Non-Disruptive Placement Feasibility

Once the demand is admitted/entitled, the scheduler should first attempt the cheapest path:

> **Can the request be satisfied using currently free schedulable capacity without disturbing any existing ResourceAllocated?**

This analysis includes:

```text
resource flavor
GPU count
CPU / RAM / NIC / storage
health
drain / maintenance state
reservation/protection state
topology constraints
gang atomicity
cross-instance constraints
fragmentation
```

For a TP8 request:

```text
100 free H100
```

does not imply feasibility if no NVSwitch domain contains 8 schedulable H100.

Therefore the scheduler must evaluate:

```text
FreeCapacityAnalysis
    +
TopologyFeasibility
    +
FragmentationAnalysis
```

together.

If a legal placement exists:

```text
choose best non-disruptive placement
    ->
ALLOCATE
```

This is always the preferred path.

---

## 7.3 Stage 3 — Diagnose Why Non-Disruptive Placement Failed

If no direct fit exists, the scheduler should classify the blocker before taking disruptive action.

Typical blocker classes:

```text
INSUFFICIENT_AGGREGATE_CAPACITY

ENTITLED_CAPACITY_LENT_OUT

FRAGMENTATION_BLOCKS_TOPOLOGY

LOWER_PRIORITY_OWNERSHIP_BLOCKS_DEMAND

CAPACITY_PROTECTED_BY_RESERVATION

TOPOLOGY_CONSTRAINT_UNSATISFIED

TEMPORAL_CONSTRAINT_UNSATISFIED
```

This diagnosis determines which transition analysis is relevant.

Examples:

```text
ENTITLED_CAPACITY_LENT_OUT
    -> Reclaim Analysis

FRAGMENTATION_BLOCKS_TOPOLOGY
    -> Consolidation Analysis

LOWER_PRIORITY_OWNERSHIP_BLOCKS_DEMAND
    -> Preemption Analysis

PROTECTED_BY_RESERVATION
    -> wait / reservation policy

INSUFFICIENT_AGGREGATE_CAPACITY
    -> PENDING
```

The scheduler should not jump directly from "no free fit" to `PREEMPT`.

---

## 7.4 Stage 4 — Resource Transition Analysis

If non-disruptive placement is impossible, the scheduler evaluates whether changing existing allocations can create a feasible placement.

This stage considers:

```text
RECLAIM
PREEMPT
CONSOLIDATE
replacement / movement plans
```

The core question is:

> **What set of existing ResourceAllocated objects should change so that the requested capacity becomes feasible, at minimum legal system cost?**

Inputs include:

```text
candidate ResourceAllocated objects

ownership provenance:
    borrowed?
    lent?
    protected?
    quota owner?

effective DisruptionPolicy

DisruptionCapabilities:
    drain?
    checkpoint?
    resumable?

runtime progress
remaining runtime estimate
checkpoint/restart cost

warm-state value

topology
fragmentation

failure-domain effects

deadline / TemporalRequirements

capacity created by each transition
```

Conceptual candidate:

```text
TransitionCandidate {
    affected_resources_allocated[]

    proposed_internal_actions[]

    useful_capacity_created

    resulting_topology

    disruption_cost

    fragmentation_delta

    policy_cost

    estimated_time_to_capacity
}
```

The scheduler selects the lowest-cost legal candidate.

---

## 7.5 Reclaim Analysis

Suppose an entitled owner needs capacity that is currently lent to a borrower.

Example:

```text
Search guarantee:
    128 H100

Search owns:
    96

Batch borrowed:
    32 from Search's unused entitlement

Search now requests:
    4 × TP8-H100
```

The scheduler should not simply reclaim any 32 H100.

It must ask:

```text
Which borrowed ResourceAllocated objects can be reclaimed
so that useful TP8-capable capacity is created
with minimum disruption?
```

Candidate A:

```text
reclaim 8 GPUs scattered across 8 NVSwitch domains
```

Result:

```text
8 free GPUs
but no TP8 fit
```

Candidate B:

```text
reclaim one complete borrowed TP8-compatible domain
```

Result:

```text
one immediately usable TP8 placement
```

Candidate B is better even if both reclaim the same GPU count.

So reclaim victim selection should consider:

```text
ReclaimCost =
    disruption_cost
  + lost_work_cost
  + checkpoint/restart_cost
  + warm_state_loss
  + number_of_victims
  + topology/fragmentation penalty
```

while maximizing:

```text
useful_capacity_created
```

`RECLAIM` may later use `PREEMPT` against a borrower that does not release by its reclaim deadline.

---

## 7.6 Consolidation Analysis

Fragmentation may block a request even when enough aggregate capacity exists.

Example:

```text
pending:
    TP8 H100

aggregate free:
    16 H100

but:
    only 2 free per NVSwitch domain
```

The scheduler should ask:

```text
Can a small number of low-cost moves create a complete feasible domain?
```

This can be preferable to generic preemption.

A possible plan:

```text
move/restart 2 flexible allocations
    ->
release one complete NVSwitch domain
    ->
TP8 becomes feasible
```

Consolidation scoring must consider:

```text
movement cost
restart cost
disruption cost
warm-state loss
topology value created
future fragmentation improvement
```

Fragmentation is not considered only during disruption.

It should influence:

```text
normal free placement scoring
reclaim victim selection
consolidation decisions
preemption victim selection
```

throughout the decision engine.

---

## 7.7 Preemption Analysis

If capacity is owned by allocations whose effective policy allows displacement, the scheduler may consider `PREEMPT`.

Preemption asks:

```text
Which ResourceAllocated objects can legally be displaced,
and which victim set creates the required feasible footprint
at minimum cost?
```

The cost model can include:

```text
priority gap
lost work
checkpoint cost
restart cost
warm-state loss
number of victims
time to release
topology usefulness
fragmentation delta
SLA risk
```

`PREEMPT` does not imply a later `ALLOCATE`.

It only selects/marks victims for displacement.

After resources are actually released, a separate scheduling cycle determines who receives the newly free capacity.

---

## 7.8 Unified Transition Cost

A useful conceptual objective is:

```text
PlanCost =
    DisruptionCost
  + LostWorkCost
  + RestartOrCheckpointCost
  + WarmStateLoss
  + PlacementCommunicationCost
  + FragmentationPenalty
  + FairnessPolicyPenalty
  + TimeToCapacity
```

subject to hard constraints:

```text
entitlement legal
preemption legal
reservation respected
topology feasible
ResourceInstance atomicity respected
TemporalRequirements satisfied
```

The scheduler may compare plans such as:

```text
Plan A:
    reclaim one borrowed TP8 allocation

Plan B:
    move 2 flexible allocations + reclaim 4 GPUs

Plan C:
    preempt 3 low-priority allocations

Plan D:
    wait 45 seconds for a job to finish
```

and choose the lowest-cost legal plan.

---

## 7.9 Final Scheduler Decision Flow

```text
                     ResourceIntent
                          |
                          v
              Ownership / Entitlement
                          |
                          v
              Non-Disruptive Placement
                          |
             +------------+-------------+
             |                          |
           FIT                        NO FIT
             |                          |
             v                          v
         ALLOCATE              Diagnose blocker
                                        |
                 +----------------------+-------------------+
                 |                      |                   |
              borrowed             fragmented          priority
                 |                      |                   |
                 v                      v                   v
              RECLAIM              CONSOLIDATE          PREEMPT
                 \                     |                   /
                  \                    |                  /
                   +--------- ResourceTransition --------+
                              Optimization
                                  |
                                  v
                         SchedulerActionPlan
```

A more explicit version:

```text
ResourceIntent
    |
    v
Validate / Normalize
    |
    v
Ownership & Entitlement Analysis
    |
    +-- not admitted / no entitlement
    |       -> PENDING / REJECTED by policy
    |
    v
Try Non-Disruptive Placement
    |
    +-- feasible
    |       -> choose best placement
    |       -> ALLOCATE
    |
    v
Diagnose Blocking Reason
    |
    +-- lent/borrowed entitlement
    |       -> RECLAIM analysis
    |
    +-- fragmentation
    |       -> CONSOLIDATE analysis
    |
    +-- lower-priority ownership
    |       -> PREEMPT analysis
    |
    +-- reservation/protected capacity
    |       -> WAIT / reservation policy
    |
    +-- insufficient aggregate capacity
    |       -> PENDING
    |
    v
Build Transition Candidates
    |
    v
Score Legal Plans
    |
    v
Choose Minimum-Cost Plan
    |
    v
SchedulerActionPlan
    |
    v
Execute Internal Actions + Controller Coordination
    |
    v
Atomic Ownership Transitions
    |
    v
Re-evaluate ResourceAllocation / Pending Demand
```

---

## 7.10 Core Mental Model

The scheduler brain can therefore be summarized as:

```text
Entitlement
    determines WHO may consume capacity.

Placement Feasibility
    determines WHERE demand can run without disruption.

Resource Transition Analysis
    determines WHAT existing allocations should change
    when no non-disruptive placement exists.

SchedulerActionPlan
    records the chosen legal transition path.
```

This is the central decision-making core of the GPU Fleet Scheduler.


# 8. ACQUIRE — Detailed Decision Algorithm

## 8.1 Semantics

```text
ACQUIRE(ResourceRequirements)
```

means:

> Reconcile the scheduler's `ResourceAllocation` toward `ResourceRequirements.desired_instances`, materializing logical instances through the primary `ResourceRequirement` or declared alternatives.

It does not mean:

```text
ALLOCATE N immediately
```

---

## 8.2 Pseudocode

```text
function handle_acquire(intent, ctx):

    if not is_valid(intent):
        return REJECTED(INVALID_INTENT)

    if not is_supported(intent.resource_requirements):
        return REJECTED(UNSUPPORTED_REQUIREMENTS)

    requirements = intent.resource_requirements
    allocation = lookup_or_create_resource_allocation(requirements)

    existing = allocation.allocated_instances
    needed = requirements.desired_instances - existing

    if needed <= 0:
        return SATISFIED(existing)

    admission = evaluate_admission(intent, ctx)

    if admission == DENY_PERMANENT:
        return REJECTED(ADMISSION_DENIED)

    if admission == WAIT:
        return PENDING(QUOTA_OR_POLICY_WAIT)

    plan = []

    # Step 1: evaluate primary + declared alternatives, including
    # demand-level TemporalRequirements and each variant's RuntimeTemporalProfile.
    feasible_variants = rank_legal_requirement_variants(
        intent.resource_requirements,
        ctx
    )

    # Step 2: use free feasible resources first
    free_fit = find_free_feasible_instances(
        feasible_variants,
        needed,
        ctx
    )

    if free_fit.count > 0:
        plan += ALLOCATE(free_fit)
        needed -= free_fit.count

    if needed == 0:
        return execute(plan)

    # Step 3: reclaim capacity that belongs to this entitlement
    reclaimable = find_reclaimable_borrowed_capacity(
        intent,
        needed,
        ctx
    )

    if reclaimable.count > 0:
        plan += RECLAIM(reclaimable)
        # controller protocol: drain/checkpoint and wait until releasable
        needed -= projected_instances_after_reclaim(reclaimable)

    if needed <= 0:
        plan += ALLOCATE(after_reclaim_capacity)
        return execute(plan)

    # Step 4: solve fragmentation if enough aggregate capacity exists
    if consolidation_can_create_fit(
        intent.resource_requirements,
        needed,
        ctx
    ):
        moves = choose_low_disruption_consolidation_plan(intent, ctx)

        plan += CONSOLIDATE(moves)
        plan += CONTROLLER_CONTROLLER_WAIT_FOR_RELEASE(moves)

        newly_created_fit = expected_fit_after_consolidation(moves)

        plan += ALLOCATE(newly_created_fit)
        needed -= newly_created_fit.count

    if needed <= 0:
        return execute(plan)

    # Step 5: consider preemption
    victims = find_legal_preemption_candidates(intent, needed, ctx)

    if victims.count > 0:
        plan += PREEMPT(victims)
        # controller protocol: optional grace/drain/checkpoint until releasable

        expected = projected_capacity_after_preemption(victims)

        plan += ALLOCATE(expected)
        needed -= expected.count

    if needed <= 0:
        return execute(plan)

    # Step 6: partial satisfaction
    if intent.resource_requirements.partial_satisfaction_allowed and plan_allocates_any(plan):
        execute(plan)

        return PARTIALLY_SATISFIED(
            allocated = requested - needed,
            pending = needed
        )

    # Step 7: cannot satisfy now
    return PENDING(NO_LEGAL_CAPACITY_PATH)
```

---

# 9. ACQUIRE — Example: Free Capacity

Request:

```text
ACQUIRE 2 TP8-H100 instances
```

Fleet:

```text
NVSwitch domain A:
    8 free H100

NVSwitch domain B:
    8 free H100
```

Plan:

```text
ALLOCATE A
ALLOCATE B
```

No reclaim/preemption.

---

# 10. ACQUIRE — Example: Guaranteed Capacity Is Lent Out

Tenant Search:

```text
guaranteed quota:
    128 H100

currently used:
    96 H100

32 H100 from Search guarantee currently borrowed by Batch
```

Search submits:

```text
ACQUIRE 4 TP8-H100 instances
```

Need:

```text
32 H100
```

Free fleet capacity:

```text
0
```

Decision:

```text
if Search demand is within guaranteed entitlement
and Batch allocation is consuming Search's lent capacity:

    select Batch reclaim victims

    RECLAIM victims
    CONTROLLER_DRAIN victims
    CONTROLLER_WAIT_FOR_RELEASE

    once 32 H100 return:
        ALLOCATE Search 4 × TP8
```

Important:

```text
RECLAIM
```

is used before generic `PREEMPT` because this is entitlement restoration.

---

# 11. ACQUIRE — Example: Fragmentation

Request:

```text
1 TP8-H100
```

Fleet:

```text
16 H100 free total

but:
    2 free in each of 8 NVSwitch domains
```

No legal TP8 placement exists.

If there are movable low-priority allocations:

```text
if aggregate capacity >= 8
and consolidation benefit > disruption cost:

    select movable allocations

    CONSOLIDATE

    create one clean 8-GPU NVSwitch domain

    ALLOCATE TP8
else:
    PENDING
```

---

# 12. ACQUIRE — Example: Preemption

Request:

```text
critical production TP8
```

No free capacity.

No reclaimable borrowed capacity.

No acceptable consolidation plan.

But:

```text
8 H100 currently owned by low-priority batch whose effective policy permits preemption
```

Decision:

```text
if production priority > victim priority
and effective_disruption_policy(victim).preemption_allowed:

    PREEMPT batch allocation
    CONTROLLER_DRAIN
    CONTROLLER_WAIT_FOR_RELEASE
    ALLOCATE production TP8
else:
    PENDING
```

---

# 13. RESIZE — Detailed Decision Algorithm

## 13.1 Semantics

```text
RESIZE(ResourceGroup, desired_instances)
```

means:

> Converge the ResourceGroup toward the desired cardinality.

---

## 13.2 Pseudocode

```text
function handle_resize(intent, ctx):

    if not is_valid(intent):
        return REJECTED(INVALID_INTENT)

    current = effective_instance_count(intent.resource_group)

    desired = intent.resource_requirements.desired_instances

    delta = desired - current

    if delta == 0:
        return SATISFIED(current)

    if delta > 0:
        derived_acquire = derive_acquire_intent(
            resource_group = intent.resource_group,
            count = delta,
            requirements = intent.resource_requirements,
            inherited_policy = intent
        )

        return handle_acquire(derived_acquire, ctx)

    # delta < 0
    remove_count = abs(delta)

    candidates = choose_release_candidates(
        group = intent.resource_group,
        count = remove_count,
        objectives = [
            minimize_disruption,
            improve_fragmentation,
            preserve_availability_constraints,
            prefer_expiring_or_policy_preemptible_instances
        ]
    )

    if candidates.count < remove_count:
        return PARTIALLY_SATISFIED(
            reason = INSUFFICIENT_RELEASABLE_ALLOCATIONS
        )

    plan = []

    for allocation in candidates:

        if effective_disruption_policy(allocation).graceful_release_required:
            # controller protocol: drain/checkpoint until releasable

        plan += RELEASE(allocation)

    return execute(plan)
```

---

# 14. RESIZE Up

Example:

```text
current = 8
desired = 12
```

Derived action:

```text
ACQUIRE 4
```

Possible internal paths:

```text
ALLOCATE 4
```

or:

```text
RECLAIM -> ALLOCATE 4
```

or:

```text
PREEMPT -> ALLOCATE 4
```

or:

```text
PARTIAL 2 + PENDING 2
```

---

# 15. RESIZE Down

Example:

```text
current = 12
desired = 8
```

Decision:

```text
select 4 release candidates
```

Candidate scoring may consider:

```text
disruption cost
warm-state value
topology balance
failure-domain balance
fragmentation benefit
lease expiry
age/minimum runtime
```

For serving:

```text
choose_release_candidates
    ->
CONTROLLER_DRAIN
    ->
controller removes from router
    ->
controller waits for in-flight requests
    ->
RELEASE
```

For disposable batch:

```text
choose_release_candidates
    ->
RELEASE / terminate
```

---

# 16. RELEASE — Detailed Decision Algorithm

## 16.1 Semantics

```text
RELEASE(resource_allocated_ids)
```

means:

> End ownership of the specified `ResourceAllocated` objects. Releasing an entire `ResourceAllocation` is shorthand for releasing all of its active `ResourceAllocated` entries.

---

## 16.2 Pseudocode

```text
function handle_release(intent, ctx):

    for resource_allocated_id in intent.release_spec.resource_allocated_ids:

        allocation = lookup_resource_allocated(resource_allocated_id)

        if allocation does not exist:
            mark_idempotent_success_if_already_released()
            continue

        if not allocation_is_owned_by(intent.owner):
            return REJECTED(NOT_OWNER)

        if allocation.state == RELEASED:
            continue

        if allocation.state == LOST:
            finalize_lost_allocation_cleanup(allocation)
            continue

        if allocation.requires_runtime_confirmation:
            transition(allocation, RELEASING)

            if not controller_has_confirmed_safe_release(allocation):
                return PENDING(WAITING_FOR_CONTROLLER_RELEASE)

        atomic_free(allocation)

    reevaluate_pending_demand()

    return SATISFIED
```

---

# 17. RELEASE Can Trigger Other Scheduler Work

A release may immediately unblock:

```text
pending ACQUIRE
pending RESIZE-up
pending gang
reservation materialization
quota reclaim completion
```

So:

```text
Customer RELEASE
    ->
RELEASE
    ->
scheduler event:
        reevaluate pending queue
    ->
ALLOCATE next eligible demand
```

A public action from one actor can therefore trigger internal actions for another actor.

---

# 18. RESERVE — Detailed Decision Algorithm

## 18.1 Semantics

```text
RESERVE(ResourceRequirements, reservation_window)
```

means:

> Guarantee that sufficient compatible capacity will be available during the requested reservation window.

---

## 18.2 Pseudocode

```text
function handle_reserve(intent, ctx):

    if not is_valid_reservation(intent):
        return REJECTED(INVALID_RESERVATION)

    if not is_supported(intent.resource_requirements):
        return REJECTED(UNSUPPORTED_REQUIREMENTS)

    if reservation_conflicts_with_higher_guarantees(intent, ctx):
        return REJECTED(RESERVATION_POLICY_CONFLICT)

    required_instances = intent.resource_requirements.desired_instances

    future_instance_capacity = forecast_feasible_instance_capacity(
        resource_requirements = intent.resource_requirements,
        at = intent.reservation_spec.valid_from
    )

    if future_instance_capacity < required_instances:
        reclaimable_instances = future_reclaimable_instance_capacity(intent, ctx)

        if future_instance_capacity + reclaimable_instances < required_instances:
            return PENDING_OR_REJECTED(UNABLE_TO_GUARANTEE)

    reservation = create_reservation_record(intent)

    if reservation_start_is_far_away(intent):

        protect_abstract_capacity_envelope(reservation)

        if safe_backfill_exists(reservation):
            allow_backfill_until(reclaim_deadline(reservation))

    else:
        concrete = choose_concrete_capacity(reservation)

        if currently_occupied_by_borrowed_or_backfill(concrete):
            schedule_reclaim_before(reservation.valid_from)

        protect_capacity(concrete)

    return ACCEPTED(reservation)
```

---

# 19. Reservation Reclaim Workflow

Example:

```text
Reservation:
    128 H100 at 09:00

Current time:
    08:40

32 of those H100 are being used by batch backfill
```

Decision:

```text
if reclaim deadline reached:

    identify backfill victims

    RECLAIM

    CONTROLLER_DRAIN

    CONTROLLER_WAIT_FOR_RELEASE

    if release completes:
        PROTECT_CAPACITY

    else if deadline expires and policy allows:
        PREEMPT / force terminate

    reservation becomes READY
```

---

# 20. REPLACE — Detailed Decision Algorithm

## 20.1 Semantics

```text
REPLACE(old_resource_allocated)
```

means:

> Establish a new `ResourceAllocated` satisfying the same logical requirement for an existing `ResourceAllocated` that is lost, draining, degraded, or intentionally superseded.

---

# 21. Planned REPLACE

Typical reasons:

```text
rollout
maintenance
hardware drain
proactive topology movement
```

Pseudocode:

```text
function handle_replace(intent, ctx):

    old = lookup(intent.replacement_spec.old_resource_allocated)

    if old does not exist:
        return REJECTED(UNKNOWN_ALLOCATION)

    requirements = replacement_requirements(old, intent)

    if old.state == LOST:
        return handle_unplanned_replace(old, requirements, intent, ctx)

    if replacement_before_release_required(intent):

        acquire = derive_acquire_for_replacement(requirements)

        acquire_result = handle_acquire(acquire, ctx)

        if acquire_result not in {SATISFIED, PARTIALLY_SATISFIED_ALLOWED}:
            return PENDING(REPLACEMENT_CAPACITY_UNAVAILABLE)

        replacement = acquire_result.new_resource_allocated

        notify_owner_replacement_ready(
            old_resource_allocated = old,
            new_resource_allocated = replacement
        )

        wait for owner handoff confirmation

        CONTROLLER_DRAIN(old)
        CONTROLLER_WAIT_FOR_RELEASE(old)
        RELEASE(old)

        return SATISFIED(replacement)

    else:

        CONTROLLER_DRAIN(old)
        CONTROLLER_WAIT_FOR_RELEASE(old)
        RELEASE(old)

        acquire = derive_acquire_for_replacement(requirements)

        return handle_acquire(acquire, ctx)
```

---

# 22. Unplanned REPLACE

Typical reasons:

```text
node failure
GPU failure
allocation LOST
```

Pseudocode:

```text
function handle_unplanned_replace(old, requirements, intent, ctx):

    MARK_LOST(old)

    acquire = derive_acquire_for_replacement(requirements)

    result = handle_acquire(acquire, ctx)

    if result == SATISFIED:
        notify_owner_replacement_available(result.allocation)

    return result
```

No graceful release is possible.

---

# 23. REPLACE Internal Paths

A planned replacement can become:

```text
ALLOCATE replacement
    ->
handoff
    ->
RELEASE old
```

or:

```text
RECLAIM
    ->
ALLOCATE replacement
    ->
handoff
    ->
RELEASE old
```

or:

```text
PREEMPT
    ->
ALLOCATE replacement
    ->
handoff
    ->
RELEASE old
```

or:

```text
PENDING
```

The public action remains simply:

```text
REPLACE
```

---

# 24. SchedulerActionPlan

The result of resolution is conceptually an explicit transition plan.

```text
SchedulerActionPlan {
    plan_id
    resource_intent_id

    based_on_snapshot_version

    steps[] {
        action_type

        target_allocations[]
        target_resources[]

        prerequisites[]

        deadline?

        retry_policy?
    }

    final_expected_state
}
```

Example:

```text
Plan P91

Public request:
    Search ACQUIRE 4 TP8-H100

Internal steps:

1. RECLAIM Batch B12
2. RECLAIM Batch B18
3. CONTROLLER_DRAIN [B12, B18]
4. CONTROLLER_WAIT_FOR_RELEASE [B12, B18]
5. ALLOCATE -> ResourceAllocated Search D9
6. ALLOCATE -> ResourceAllocated Search D10
7. ALLOCATE -> ResourceAllocated Search D11
8. ALLOCATE -> ResourceAllocated Search D12
```

The implementation does not necessarily need to persist a full DAG for every trivial allocation.

But this is the correct conceptual model.

---

# 25. Action Plan Concurrency

Plans are based on a fleet snapshot.

```text
plan snapshot version = V100
```

While the plan executes:

```text
another scheduler shard may allocate resources
node health may change
victim may release early
reservation may be cancelled
```

Therefore:

> **Planning never implies ownership.**

Every ownership-changing step must revalidate atomically.

Example:

```text
ALLOCATE node42/gpu0..7
```

must use:

```text
compare / lease / CAS / transactional claim
```

against current ownership state.

If claim fails:

```text
invalidate affected plan step
refresh state
replan
```

---

# 26. Scheduler-Triggered Decisions

Not all internal workflows begin with a fresh public action.

The scheduler also reacts to control-plane events.

Important triggers:

```text
capacity released
resource failure
health degradation
maintenance/drain
reservation deadline
guaranteed quota deficit
high-priority pending demand
fragmentation threshold
new capacity arrival
lease expiry
intent cancellation
```

---

# 27. Scheduler Trigger: Capacity Released

Event:

```text
allocation A becomes RELEASE
```

Algorithm:

```text
function on_capacity_released(resources):

    candidate_intents =
        pending_intents_compatible_with(resources)

    ordered =
        apply_queue_quota_fairshare_priority(candidate_intents)

    for intent in ordered:

        if resources can satisfy intent:
            ALLOCATE
            break or continue while resources remain
```

---

# 28. Scheduler Trigger: Resource Failure

Event:

```text
GPU/node becomes UNHEALTHY
```

Algorithm:

```text
function on_resource_failure(resource):

    mark_resource_unschedulable(resource)

    affected = allocations_using(resource)

    for allocation in affected:

        MARK_LOST(allocation)

        notify_owner(allocation, RESOURCE_LOST)

        group = allocation.resource_group

        if group.desired_capacity > healthy_group_capacity(group):

            derive replacement demand

            attempt:
                ALLOCATE
                else RECLAIM -> ALLOCATE
                else PREEMPT -> ALLOCATE
                else PENDING
```

---

# 29. Scheduler Trigger: Maintenance / Drain

Event:

```text
node/rack enters DRAINING
```

Algorithm:

```text
function on_maintenance_drain(domain, deadline):

    mark_domain_no_new_resource_allocateds(domain)

    affected = allocations_in(domain)

    for allocation in affected:

        policy = effective_disruption_policy(allocation)

        if policy.requires_replacement_before_release:

            REPLACE(allocation)

        elif policy.checkpoint_capable:

            CONTROLLER_DRAIN(allocation, deadline)
            CONTROLLER_WAIT_FOR_RELEASE
            RELEASE

            optionally reacquire later

        elif policy.preemption_allowed:

            CONTROLLER_DRAIN or PREEMPT
            RELEASE

        else:

            escalate maintenance conflict
```

The scheduler owns the resource decision.

The controller owns the safe drain mechanics.

---

# 30. Scheduler Trigger: Reservation Deadline

Event:

```text
reservation reclaim deadline reached
```

Algorithm:

```text
function on_reservation_reclaim_deadline(reservation):

    gap = required_capacity(reservation)
          - currently_protected_capacity(reservation)

    if gap <= 0:
        return

    victims = allocations_using_reserved_future_capacity(reservation)

    victims = rank_by(
        borrowed_first,
        backfill_first,
        lowest_disruption_cost,
        shortest_remaining_runtime
    )

    RECLAIM(victims)
    CONTROLLER_DRAIN(victims)
    CONTROLLER_WAIT_FOR_RELEASE

    if capacity now sufficient:
        PROTECT_CAPACITY
        mark reservation READY

    elif deadline passed and forced_preemption_allowed:
        PREEMPT remaining victims
        PROTECT_CAPACITY

    else:
        mark reservation AT_RISK
```

---

# 31. Scheduler Trigger: Guaranteed Quota Reclaim

Event:

```text
Tenant A has admitted demand
but part of A's guarantee is currently lent to Tenant B
```

Algorithm:

```text
function restore_guarantee(tenant_A):

    deficit = guaranteed_entitlement(A)
              - currently_owned_or_reserved(A)

    if deficit <= 0:
        return

    borrowers = allocations_using_A_lent_capacity()

    victims = rank_reclaim_candidates(borrowers)

    RECLAIM until deficit is recovered

    then allocate A's pending admitted demand
```

This is distinct from generic priority preemption.

---

# 32. Scheduler Trigger: High-Priority Pending Demand

Event:

```text
new high-priority intent admitted
```

Algorithm:

```text
if free feasible capacity exists:
    ALLOCATE

elif reclaimable borrowed capacity exists:
    RECLAIM -> ALLOCATE

elif consolidation can cheaply create fit:
    CONSOLIDATE -> ALLOCATE

elif legal preemption victims exist:
    PREEMPT -> CONTROLLER_WAIT_FOR_RELEASE -> ALLOCATE

else:
    PENDING
```

---

# 33. Scheduler Trigger: Fragmentation

Event:

```text
large/high-value intent remains pending
and aggregate free capacity is sufficient
but no legal topology placement exists
```

Algorithm:

```text
function maybe_consolidate(intent):

    if aggregate_capacity < required_capacity:
        return NO_OP

    if legal_feasible_domain_exists:
        return ALLOCATE

    movable = find_movable_allocations()

    plans = generate_bounded_consolidation_candidates(movable)

    best = argmin(
        disruption_cost
        + migration_cost
        + SLA_risk
        - placement_value
    )

    if best.benefit > consolidation_threshold:
        CONSOLIDATE(best)
        ALLOCATE(intent)
    else:
        PENDING
```

Consolidation should be bounded.

Avoid constant cluster reshuffling.

---

# 34. Scheduler Trigger: New Capacity Joins

Event:

```text
node becomes HEALTHY + SCHEDULABLE
```

Algorithm:

```text
update FleetResourceSnapshot

candidate_pending =
    pending intents matching new flavor/topology

order candidates using:
    admission
    quota
    fair share
    priority

ALLOCATE highest-ranked feasible demand
```

Actors should not need to resubmit persistent `ResourceIntent`s.

---

# 35. Scheduler Trigger: Lease Expiry

Event:

```text
allocation lease expires
```

Algorithm:

```text
if owner renewed lease:
    continue

elif graceful release required:
    CONTROLLER_DRAIN
    CONTROLLER_WAIT_FOR_RELEASE
    RELEASE

elif effective_disruption_policy(allocation).preemption_allowed:
    revoke / terminate
    RELEASE
```

Then reevaluate pending demand.

---

# 36. Scheduler Trigger: Intent Cancellation

Event:

```text
ResourceIntent cancelled
```

Algorithm:

```text
cancel pending unsatisfied demand

cancel unmaterialized reservations if policy permits

for allocations owned exclusively by cancelled intent:

    if graceful release required:
        CONTROLLER_DRAIN
        CONTROLLER_WAIT_FOR_RELEASE

    RELEASE
```

---

# 37. Victim Selection

Whenever the scheduler must reclaim, consolidate, or preempt, it needs victim selection.

Conceptual score:

```text
VictimCost =
    disruption_cost
  + lost_work_cost
  + checkpoint_cost
  + warm_state_loss
  + SLA_risk
  + topology_side_effect
  + fairness_penalty
```

Possible positive preference for eviction:

```text
borrowed capacity
allocation whose effective policy permits preemption
lease nearing expiry
short drain time
already checkpointed
low priority
low warm-state value
```

Different internal actions use different eligibility rules.

---

# 38. Reclaim Victim Selection

Eligibility:

```text
must be using borrowed/lent capacity
```

Ranking may prefer:

```text
same lender entitlement
lowest disruption
shortest revoke time
best topology recovery
```

Priority is secondary to entitlement provenance.

---

# 39. Preemption Victim Selection

Eligibility:

```text
effective_disruption_policy(victim).preemption_allowed == true
requester/victim effective priority and policy permit displacement
```

Ranking may consider:

```text
priority gap
disruption cost
amount of capacity recovered
topology usefulness
number of victims
age / minimum runtime
checkpointability
```

Avoid killing ten small jobs if one low-cost victim can create the required topology shape.

---

# 40. Consolidation Candidate Selection

Eligibility:

```text
movable/restartable allocation
```

Goal:

```text
recover useful topology domains
```

not merely maximize total free GPUs.

Example:

```text
freeing one complete NVSwitch domain
```

may be more valuable than creating the same number of scattered GPUs.

---

# 41. Partial Satisfaction

Public actions can be partially satisfied only if allowed.

Example:

```text
ResourceRequirements.desired_instances = 4
ResourceAllocation.allocated_instances = 2
```

If:

```text
partial_satisfaction_allowed = true
```

scheduler may return:

```text
PARTIALLY_SATISFIED

ResourceAllocation.allocated_instances = 2
ResourceAllocation.pending_instances = 2
```

If one instance itself is a strict gang:

```text
TP8
```

then that individual instance must still be atomically whole.

Partial satisfaction applies across ResourceInstances, not inside an indivisible gang unless the requirements explicitly allow elasticity.

---

# 42. Pending Intent Lifecycle

Persistent intents remain active.

```text
PENDING
    |
    +-- capacity release
    +-- quota change
    +-- reclaim completion
    +-- node recovery
    +-- new capacity
    +-- reservation expiration
    |
    v
reevaluate
```

The actor should not need to repeatedly submit the same desired state.

This is one reason `ResourceIntent` is versioned and durable.

---

# 43. Policy Boundary

The decision process is intentionally split between:

```text
mechanism
```

and:

```text
policy
```

Mechanism knows how to:

```text
allocate
reclaim
preempt
consolidate
reserve
release
```

Policy decides:

```text
who is entitled
who may borrow
who gets reclaimed
who may preempt whom
which deadline dominates
how much disruption is acceptable
```

Phase 3 will define much of the entitlement/fair-share policy used by these decision branches.

---

# 43A. Updating ResourceAllocation During Decision Execution

Every successful concrete placement creates a `ResourceAllocated` and updates the aggregate `ResourceAllocation`.

Example:

```text
ResourceRequirements.desired_instances = 50

after scheduling:
    ResourceAllocation.allocated_instances = 47
    ResourceAllocation.pending_instances = 3

    ResourceAllocation.resources_allocated = [
        ResourceAllocated #1,
        ...
        ResourceAllocated #47
    ]
```

A pending decision is therefore not merely a queue item; it is remaining unsatisfied cardinality on a durable `ResourceAllocation`.

`REPLACE` normally replaces one `ResourceAllocated`.

`RELEASE` can release selected `ResourceAllocated` entries or the entire aggregate allocation.

---

# 44. Error / Status Semantics

Useful scheduler statuses:

```text
SATISFIED

PARTIALLY_SATISFIED

PENDING
    NO_FEASIBLE_CAPACITY
    WAITING_FOR_QUOTA
    WAITING_FOR_RECLAIM
    WAITING_FOR_PREEMPTION
    WAITING_FOR_DRAIN
    WAITING_FOR_RESERVATION
    WAITING_FOR_TOPOLOGY_FIT

REJECTED
    INVALID_INTENT
    UNSUPPORTED_REQUIREMENTS
    POLICY_DENIED
    UNKNOWN_ALLOCATION
    NOT_OWNER
```

A caller should be able to distinguish:

```text
wrong request
```

from:

```text
valid request waiting for capacity
```

---

# 45. Public Action to Internal Workflow Summary

| Public Action | Typical Internal Workflow |
|---|---|
| `ACQUIRE` | `ALLOCATE`; or `RECLAIM -> WAIT -> ALLOCATE`; or `CONSOLIDATE -> ALLOCATE`; or `PREEMPT -> WAIT -> ALLOCATE`; or `PENDING` |
| `RESIZE` up | Derive `ACQUIRE(delta)` |
| `RESIZE` down | `choose_release_candidates -> CONTROLLER_DRAIN -> RELEASE` |
| `RELEASE` | `ResourceAllocated: RELEASING -> RELEASED/RELEASE -> reevaluate pending demand` |
| `RESERVE` | `PROTECT_CAPACITY`; optionally `BACKFILL -> RECLAIM -> PROTECT_CAPACITY` |
| `REPLACE` planned | Acquire replacement `ResourceAllocated` through any legal capacity path -> handoff -> release old |
| `REPLACE` lost | `MARK_LOST -> ACQUIRE replacement` |

---

# 46. Scheduler-Triggered Event Summary

| Trigger | Internal Decision |
|---|---|
| Capacity released | Reevaluate pending demand -> `ALLOCATE` |
| GPU/node failure | `MARK_LOST -> REPLACE / ACQUIRE` |
| Maintenance/drain | replacement/drain decision based on disruption policy |
| Reservation deadline | `RECLAIM/PREEMPT -> PROTECT_CAPACITY` |
| Guaranteed quota deficit | `RECLAIM` lent capacity |
| High-priority demand | free -> reclaim -> consolidate -> preempt -> pending |
| Fragmentation | optional bounded `CONSOLIDATE -> ALLOCATE` |
| New capacity | reevaluate pending -> `ALLOCATE` |
| Lease expiry | drain/revoke -> `RELEASE` |
| Intent cancellation | cancel pending + release owned resources |

---

# 47. Core Invariants

1. **Public actions specify outcomes, never internal mechanisms.**
2. **`ResourceInstance` is defined by the client/platform; the scheduler creates `ResourceAllocated`, not logical instances.**
3. **`ResourceAllocation` is aggregate progress for one `ResourceRequirements`.**
4. **The scheduler may select only the primary or declared alternative `ResourceRequirement`s.**
5. **Free feasible capacity is preferred when policy and temporal constraints allow it.**
6. **Reclaim and preemption are semantically different.**
7. **`RECLAIM` is entitlement restoration and may emit one or more `PREEMPT` actions against borrowers.**
8. **`PREEMPT` only decides displacement of a `ResourceAllocated`; it never implies a subsequent `ALLOCATE`.**
9. **Reclaim restores entitlement; preemption is the eviction primitive used when policy allows displacement.**
10. **Clients do not author disruption policy; it is derived by the scheduler/control plane.**
11. **Disruption capabilities constrain how a derived disruption decision may safely execute.**
10. **Consolidation is optional and must justify its disruption cost.**
11. **Gang atomicity inside one `ResourceInstance` is preserved even when the overall `ResourceRequirements` is partially satisfied.**
12. **Actors/controllers own domain-safe drain/checkpoint/handoff.**
13. **Scheduler owns which `ResourceAllocated` objects must move and why.**
14. **Exclusive physical ownership changes are always atomic.**
15. **Plans are based on snapshots but never assume snapshot ownership.**
16. **Pending ResourceAllocations are durable and reevaluated on relevant fleet events.**
17. **Scheduler-triggered workflows can occur without a new customer API call.**
18. **Demand-level `TemporalRequirements` and variant-specific `RuntimeTemporalProfile`s participate in scheduling feasibility/ranking where relevant.**
19. **Internal action eligibility is policy constrained; actions are not arbitrary heuristics.**
20. **Temporary scarcity produces `PENDING`, not request invalidation.**
21. **Phase 3 quota/fair-share policy feeds directly into these decision branches.**

---

# 48. Interview-Level Summary

> **The scheduler exposes a small public outcome vocabulary—Acquire, Resize, Release, Reserve, and Replace—but resolves each request through a policy-driven decision tree. A client-defined `ResourceRequirements` asks for N logical instances, with one primary and zero or more alternative singular `ResourceRequirement`s; each singular requirement contains the client-defined `ResourceInstance`. The scheduler never invents that logical unit. Instead, every successful physical placement creates a `ResourceAllocated`, and the aggregate `ResourceAllocation` tracks satisfied versus pending instance count. Acquire first checks admission and existing allocation state, ranks declared requirement alternatives using feasibility and temporal information, then tries free capacity, reclaim of borrowed entitlement, bounded consolidation, legal preemption, and finally pending. Resize-up reduces to Acquire; resize-down selects `ResourceAllocated` entries for drain/release. Reserve establishes a future logical-instance capacity guarantee and may use backfill plus timed reclaim. Replace composes acquisition and release around one concrete `ResourceAllocated`. Disruption capabilities come from trusted workflow/runtime metadata, while effective disruption policy is derived by the scheduler/control plane. The same engine also reacts to failures, maintenance, reservation deadlines, quota reclaim, fragmentation, and new capacity without requiring a new client request.**
