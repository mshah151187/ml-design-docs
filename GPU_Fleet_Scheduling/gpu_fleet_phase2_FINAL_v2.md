# Phase 2 — Resource Intent, Scheduler Actions, and Actor Contracts

## 1. Why This Phase Exists

Phase 1 answered:

> **What resources exist, how are they connected, and what is schedulable?**

Phase 2 answers:

> **What resource outcome is an upstream system asking the GPU Fleet Scheduler to achieve?**

The GPU Fleet Scheduler sits below several complex workflows:

- Serving Deployment
- Serving Autoscaling
- Model Rollout / Upgrade
- Training
- Batch / Evaluation
- Interactive Development
- Capacity Reservation
- Failure Recovery

Each workflow understands its own domain semantics. The GPU scheduler should not.

It should understand a common **resource language**:

```text
resource owner
resource requirements
quantity
atomicity
topology
availability
priority
preemptibility
time
desired state
```

Core principle:

> **The GPU Fleet Scheduler is a resource broker, not a workflow engine.**

---

# 2. System Boundary

```text
                 HIGHER-LEVEL WORKFLOWS
================================================================

 Serving Deployment Controller
 Serving Autoscaler
 Rollout / Upgrade Controller
 Training Job Controller
 Batch / Evaluation Controller
 Interactive Session Controller
 Capacity Reservation Controller
 Failure / Recovery Controller

          domain semantics
                |
                | translate
                v

                 ResourceIntent API

================================================================
                   GPU FLEET SCHEDULER

 Resource Intent Normalization / Validation
 Tenant / Queue / Quota Admission
 Gang / Resource Requirements Scheduling
 Topology-Aware Placement
 Fragmentation Optimization
 Reclaim / Preemption / Consolidation
 Reservation / Allocation Commit

================================================================
                      EXECUTION

 Serving Runtime / Deployment Reconciler
 Training Runtime
 Batch Runtime
 Kubernetes / Container Runtime
```

The scheduler does not need to know:

```text
"this is step 4 of a Llama rollout"
"this training job is in epoch 17"
"TTFT caused the autoscaler to scale"
"this evaluation is testing a new checkpoint"
```

It only needs the resource consequences.

---

# 3. Demand Plane vs Supply Plane

Not every fleet interaction is a `ResourceIntent`.

## Demand Plane

Consumers request resources:

```text
Deployment Controller
Autoscaler
Training Controller
Batch Controller
Reservation Controller
Recovery Controller
```

They use the `ResourceIntent API`.

## Supply Plane

Fleet systems modify what resources exist or are schedulable:

```text
Node Discovery
Health Controller
Maintenance Controller
Capacity Operator
Topology Controller
```

They update `FleetResourceState / FleetControl API`.

Example:

```text
Maintenance Controller:
    Drain rack R7 by 18:00
```

This is a supply-side state transition, not a consumer resource request.

---

# 4. Public Scheduler Action Semantics

The public resource language contains five primary actions:

```text
ACQUIRE
RESIZE
RELEASE
RESERVE
REPLACE
```

These describe **resource outcomes**.

They are intentionally different from internal scheduler mechanisms such as:

```text
ALLOCATE
BACKFILL
CONSOLIDATE
RECLAIM
PREEMPT
```

An actor asks for an outcome. The scheduler decides the mechanism.

---

# 5. ACQUIRE

## Definition

> **Obtain a new set of resource instances matching one of the resource requirementss.**

Typical use:

```text
new deployment
fixed-size training job
interactive GPU session
one-shot batch allocation
```

Example:

```text
ACQUIRE
    4 instances

Each instance:
    TP8
    8 × H100
    same NVSwitch domain
```

## Semantics

`ACQUIRE` is tied to an immutable/idempotent intent identity.

Retrying the same:

```text
intent_id = I123
ACQUIRE 4
```

must not allocate another four.

Possible result:

```text
SATISFIED
PARTIALLY_SATISFIED
PENDING
REJECTED
```

For a strict fixed gang, partial satisfaction may be forbidden. For elastic capacity, partial satisfaction may be legal.

---

# 6. RESIZE

## Definition

> **Converge a long-lived ResourceGroup toward a desired capacity target.**

Typical use:

```text
serving autoscaling
elastic training
elastic batch workers
manual service scaling
```

Example:

```text
ResourceGroup:
    search/llama70b/decode

current allocated:
    8

target:
    min = 4
    desired = 12
    max = 24
```

Desired-state is safer than a delta:

```text
desired = 12
```

rather than:

```text
+4
```

Retries remain idempotent.

### Scale Up

```text
allocated = 8
desired = 12
```

Scheduler attempts to acquire four additional instances.

### Scale Down

```text
allocated = 12
desired = 8
```

For disruption-sensitive online services, scheduler should usually produce `ReleaseCandidate[]`; the owning controller drains safely and then issues `RELEASE`.

---

# 7. RELEASE

## Definition

> **Relinquish ownership of one or more existing allocations.**

Typical use:

```text
scale-down after drain
training completion
batch completion
interactive lease end
old replica after rollout handoff
```

Example:

```text
RELEASE {
    allocation_ids = [A17, A18]
}
```

After successful release, allocation ownership ends and resources become schedulable again.

The actor owns domain-safe shutdown:

```text
Serving Controller:
    drain traffic
    -> RELEASE

Training Controller:
    checkpoint
    -> RELEASE
```

---

# 8. RESERVE

## Definition

> **Protect a resource quantity/shape for a future or immediate time window without necessarily starting the consuming workload yet.**

Typical use:

```text
planned product launch
large training campaign
scheduled rollout
failure reserve
capacity guarantee
```

Example:

```text
RESERVE {
    valid_from = 09:00
    valid_until = 12:00

    quantity:
        16 TP8-capable H100 instances
}
```

Reservation may be abstract or concrete depending on timing and implementation.

Before reservation start, backfill may consume reclaimable capacity if it cannot delay the reservation.

---

# 9. REPLACE

## Definition

> **Obtain equivalent/new capacity for an existing allocation or resource instance whose placement must be removed or superseded.**

Typical use:

```text
node failure
node drain
rolling model upgrade
bad GPU
topology evacuation
serving replica repair
```

Example:

```text
REPLACE {
    old_allocation = A17
    replacement_requirements = same TP8-H100 shape
    handoff_policy = REPLACEMENT_BEFORE_RELEASE
}
```

This differs from `RELEASE + ACQUIRE` because online systems often need replacement first to preserve availability.

---

# 10. Internal Scheduler Actions

Public actions describe the **resource outcome** requested by an upstream actor:

```text
ACQUIRE
RESIZE
RELEASE
RESERVE
REPLACE
```

The scheduler satisfies those outcomes using this canonical set of **internal scheduler actions**:

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

A public action is not a one-to-one alias for one internal action.

Examples:

```text
ACQUIRE
    -> ALLOCATE
```

```text
ACQUIRE
    -> RECLAIM
    -> controller drain/checkpoint
    -> RELEASE
    -> ALLOCATE
```

```text
ACQUIRE
    -> PREEMPT
    -> controller grace/drain
    -> RELEASE
    -> ALLOCATE
```

```text
ACQUIRE
    -> CONSOLIDATE
    -> ALLOCATE
```

The canonical internal actions are scheduler-owned resource decisions or ownership-state transitions. Domain-specific steps such as draining requests, checkpointing training, waiting for readiness, or handing off traffic are controller/runtime workflow steps and are **not** scheduler internal actions.

## 10.1 ALLOCATE

> **Atomically assign schedulable physical resources to one requested logical `ResourceInstance`, creating one `ResourceAllocated`.**

Preconditions include feasibility, admission/entitlement, topology constraints, and successful exclusive ownership commit.

## 10.2 RECLAIM

> **Recover capacity currently consumed through borrowing/lending because the entitled owner now needs it.**

`RECLAIM` is entitlement restoration. It is not generic priority preemption.

A reclaim decision may require the owning controller to drain/checkpoint before the scheduler performs `RELEASE`.

## 10.3 PREEMPT

> **Select an existing `ResourceAllocated` for displacement because effective scheduler policy and priority allow another demand to take precedence.**

`PREEMPT` is a priority/policy action. Effective disruption policy is platform-derived, not client-authored.

## 10.4 CONSOLIDATE

> **Rearrange movable allocations to reduce fragmentation and create a useful feasible resource footprint.**

Example: aggregate free H100 capacity exists, but no complete TP8-capable NVSwitch domain is free.

## 10.5 BACKFILL

> **Temporarily place suitable work into otherwise-idle capacity that must remain reclaimable for a known future need.**

Typical future needs include reservations, large gangs, or planned maintenance windows.

## 10.6 PROTECT_CAPACITY

> **Mark a capacity/topology envelope as unavailable to unrelated future allocations because it is committed to a reservation or protected future demand.**

Protection may be abstract or concrete.

## 10.7 RELEASE

> **End scheduler ownership of one or more `ResourceAllocated` objects and return their physical resources to schedulable capacity.**

A controller may first drain/checkpoint, but the scheduler-owned transition is the release of resource ownership.

## 10.8 MARK_LOST

> **Record that a `ResourceAllocated` is no longer usable because its underlying physical resources failed or disappeared.**

`MARK_LOST` commonly derives replacement demand.

## 10.9 Controller Steps Are Not Scheduler Actions

These may appear in workflow diagrams:

```text
REQUEST_DRAIN
WAIT_FOR_RELEASE
CHECKPOINT
WAIT_FOR_READY
TRAFFIC_HANDOFF
```

but they are controller/runtime protocol steps, not part of the internal scheduler-action enum.

The ownership split is:

```text
Actor:
    asks for a public resource outcome

Scheduler:
    chooses internal resource actions

Domain Controller:
    performs safe drain/checkpoint/handoff when needed

Allocation subsystem:
    atomically commits ownership transitions
```


# 11. Core Scheduler Structures

Phase 2 introduces the scheduler-native structures shared by all actors:

```text
ResourceIntent
ResourceGroup
ResourceRequirements
ResourceInstance
CrossInstanceConstraints
ResourceAllocation
ResourceAllocated
ResourceReservation
PlacementConstraint
AvailabilityConstraint
DisruptionPolicy
TemporalRequirements
ResourceIntentStatus
```

The most important ownership boundary is:

> **The client defines what one usable `ResourceInstance` means. The scheduler does not invent that logical unit.**

The scheduler works on `ResourceRequirements`, which ask for one or more copies of a client-defined `ResourceInstance`.

---

# 12. ResourceGroup

## Definition

> **A stable long-lived workflow, job, deployment, or business-process ownership scope.**

Examples:

```text
serving/llama70b-prod
training/job-9182
evaluation/run-723
rollout/llama70b-v18
notebook/session-4481
```

Conceptually:

```text
ResourceGroup {
    resource_group_id

    owner_ref
    requester_type

    tenant_id
    queue_ref

    resource_requirements[]

    generation
}
```

A `ResourceGroup` can contain one or more independent `ResourceRequirements`.

Example serving group:

```text
ResourceGroup:
    serving/llama70b-prod

ResourceRequirements:
    prefill
    decode
```

Example training group:

```text
ResourceGroup:
    training/job-9182

ResourceRequirements:
    trainer-gang
    auxiliary-gpu-workers?
```

`ResourceGroup` answers:

> **Which larger workflow or long-lived business process owns these resource requests?**

---

# 13. ResourceRequirements

## Definition

> **The complete scheduler-facing demand for one logical type of capacity inside a ResourceGroup.**

`ResourceRequirements` is plural because it can contain one primary acceptable requirement plus zero or more alternative acceptable requirements.

Conceptually:

```text
ResourceRequirements {
    requirement_id

    primary_requirement: ResourceRequirement
    alternative_requirements[]: ResourceRequirement

    min_instances
    desired_instances
    max_instances

    partial_satisfaction_allowed

    cross_instance_constraints[]

    temporal_requirements?

    # classification / references used by platform policy
    workflow_class_ref?
    priority_class_ref?

    cost_policy_ref?
    warm_state_key?
}
```

It answers:

> **How many logical instances do I want, and what are the acceptable ways the scheduler may materialize each instance?**

Example:

```text
ResourceRequirements {
    requirement_id = decode

    primary_requirement = H200_TP4

    alternative_requirements = [
        H100_TP8
    ]

    min_instances = 20
    desired_instances = 50
    max_instances = 100

    partial_satisfaction_allowed = true
}
```

The scheduler therefore sees:

```text
"I want 50 logical Decode instances.

Prefer this ResourceRequirement.

If necessary, these alternative ResourceRequirements are also valid."
```

The scheduler does not invent alternatives.

---

# 14. ResourceRequirement

## Definition

> **One concrete, client/platform-qualified way to materialize one logical ResourceInstance.**

Conceptually:

```text
ResourceRequirement {
    requirement_variant_id

    resource_instance

    variant_specific_constraints[]?

    runtime_temporal_profile?

    performance_class?
    cost_class?
}
```

A `ResourceRequirement` describes one valid physical execution contract for one logical instance.

Example primary requirement:

```text
ResourceRequirement H200_TP4 {
    resource_instance {
        member_count = 4

        per_member_resources {
            GPU = 1 × H200
            CPU = 8
            RAM = 50GB
        }

        gang {
            atomic = true
            min_members = 4
            desired_members = 4
            max_members = 4
        }

        intra_instance_placement {
            REQUIRED:
                SAME_NVSWITCH_DOMAIN
        }
    }
}
```

Alternative:

```text
ResourceRequirement H100_TP8 {
    resource_instance {
        member_count = 8

        per_member_resources {
            GPU = 1 × H100
            CPU = 8
            RAM = 50GB
        }

        gang {
            atomic = true
            min_members = 8
            desired_members = 8
            max_members = 8
        }

        intra_instance_placement {
            REQUIRED:
                SAME_NVSWITCH_DOMAIN
        }
    }
}
```

Both are qualified ways to produce the same logical capacity unit.

---

# 15. ResourceInstance

## Definition

> **The caller-defined logical allocation unit inside one ResourceRequirement.**

The client/platform already knows what one useful unit means.

Examples:

```text
one TP4 Decode replica
one TP8 Decode replica
one entire 64-GPU training gang
one independent 1-GPU batch worker
```

Conceptually:

```text
ResourceInstance {
    instance_definition_id

    members[]

    gang_semantics

    intra_instance_constraints[]
}
```

A homogeneous form may be represented compactly as:

```text
ResourceInstance {
    member_count = 4

    per_member_resources {
        GPU = 1 × H200
        CPU = 8
        RAM = 50GB
    }

    gang {
        min_members = 4
        desired_members = 4
        max_members = 4
        atomic = true
    }

    intra_instance_placement {
        REQUIRED:
            SAME_NVSWITCH_DOMAIN
    }
}
```

The scheduler does not decide that four GPUs constitute one replica. The caller/platform defined that semantic unit.

---

# 16. Primary vs Alternative Requirements

The client provides:

```text
ResourceRequirements {
    primary_requirement
    alternative_requirements[]
}
```

The semantic contract is:

```text
primary_requirement
    = preferred/default way to satisfy one logical instance

alternative_requirements
    = other explicitly qualified ways that are also acceptable
```

Example:

```text
primary:
    4 × H200, TP4

alternatives:
    8 × H100, TP8
```

The scheduler may choose an alternative because of:

```text
availability
quota
fragmentation
topology
cost
```

but it must never invent:

```text
6 × H100, TP6
```

unless that appears as another declared `ResourceRequirement`.

This keeps hardware/runtime flexibility explicit while preserving the scheduler boundary.

---

# 17. Instance Count Belongs to ResourceRequirements

The requested cardinality is common across the primary and alternatives, so it belongs on the plural wrapper.

```text
ResourceRequirements {
    min_instances
    desired_instances
    max_instances
    partial_satisfaction_allowed
}
```

Example:

```text
desired_instances = 50
```

means:

> Materialize 50 logical instances, using the primary requirement when possible and declared alternatives when policy permits.

It does **not** mean:

```text
50 H200_TP4
+
50 H100_TP8
```

The alternatives compete to satisfy the same logical instance demand.

For a fixed 64-GPU training job:

```text
min_instances = 1
desired_instances = 1
max_instances = 1
partial_satisfaction_allowed = false
```

For elastic batch workers:

```text
min_instances = 10
desired_instances = 100
max_instances = 100
partial_satisfaction_allowed = true
```

---

# 18. Intra-Instance vs Cross-Instance Constraints

## Intra-Instance Constraints

Belong in the singular `ResourceRequirement` through its `ResourceInstance`.

Examples:

```text
all TP ranks in same NVSwitch domain
all members allocated atomically
GPU-to-NIC affinity
per-member CPU/RAM
same fabric block for a large gang
```

These constraints can differ by alternative.

Example:

```text
H200_TP4:
    4 GPUs in one NVSwitch domain

H100_TP8:
    8 GPUs in one NVSwitch domain
```

## Cross-Instance Constraints

Belong in plural `ResourceRequirements` because they apply to the logical replicas regardless of which requirement variant materialized them.

Examples:

```text
spread 50 Decode instances across racks
no more than 5 logical instances per rack
minimum 3 zones
anti-affinity across logical instances
```

Example:

```text
ResourceRequirements {
    desired_instances = 50

    cross_instance_constraints {
        min_distinct_racks = 10
        max_instances_per_rack = 5
    }
}
```

This is an important separation:

```text
ResourceRequirement / ResourceInstance
    = how one logical unit may be built

ResourceRequirements
    = how many units are needed and how those units relate to one another
```

# 19. PlacementConstraint

Placement constraints used **within one ResourceInstance** describe required or preferred locality.

```text
PlacementConstraint {
    scope

    topology_level

    relationship

    strength:
        REQUIRED
        PREFERRED
}
```

Examples:

```text
REQUIRED:
    all TP4 members in same NVSwitch domain

PREFERRED:
    members close to a preferred NIC / rail
```

Hard constraints determine feasibility.

Soft constraints affect ranking only.

---

# 20. Availability / Cross-Instance Constraints

Communication locality and failure isolation use the same topology hierarchy for different reasons.

Example:

```text
inside each TP4 ResourceInstance:
    PACK into one NVSwitch domain

across 50 ResourceInstances:
    SPREAD across racks
```

Conceptually:

```text
CrossInstanceConstraint {
    topology_level

    min_distinct_domains?
    max_instances_per_domain?

    affinity?
    anti_affinity?

    strength:
        REQUIRED
        PREFERRED
}
```

---

# 21. Disruption Capabilities vs Disruption Policy

The client should **not** directly tell the scheduler:

```text
"you may preempt me"
"my disruption cost is LOW"
"you may reclaim my GPUs whenever you want"
```

Those are policy decisions.

Instead, separate:

```text
DisruptionCapabilities
    = what the runtime/workflow is technically capable of doing

DisruptionPolicy
    = what the platform/scheduler is allowed to do
```

## DisruptionCapabilities

These capabilities come from the workflow type, runtime profile, controller, or platform metadata.

Conceptually:

```text
DisruptionCapabilities {
    graceful_drain_supported

    checkpoint_supported

    resumable

    replacement_before_release_supported

    max_graceful_shutdown_time?

    checkpoint_cost_class?

    restart_cost_class?
}
```

Examples:

### Online serving replica

```text
graceful_drain_supported = true
checkpoint_supported = false
resumable = true
replacement_before_release_supported = true
```

### Training job

```text
graceful_drain_supported = true
checkpoint_supported = true
resumable = true
```

### Disposable batch worker

```text
graceful_drain_supported = false
checkpoint_supported = false
resumable = true
restart_cost = LOW
```

These describe **runtime mechanics**, not permission to disrupt.

## DisruptionPolicy

`DisruptionPolicy` is derived by the platform from information such as:

```text
workflow / job class
tenant policy
queue policy
priority class
SLO / criticality class
reservation status
whether capacity is borrowed
whether the allocation is backfill
minimum-runtime guarantees
maintenance/emergency policy
```

Conceptually:

```text
DisruptionPolicy {
    preemption_allowed

    reclaim_allowed

    graceful_release_required

    minimum_runtime?

    revoke_notice_period?

    disruption_cost_class:
        LOW
        MEDIUM
        HIGH
        CRITICAL

    force_termination_allowed?
}
```

This object is scheduler/control-plane owned.

The client cannot self-declare policy such as:

```text
preemption_allowed = false
disruption_cost = CRITICAL
reclaim_allowed = false
```

simply by setting fields in `ResourceIntent`.

Likewise, a client cannot opt into reclaiming capacity that the scheduler policy says must remain protected.

## Why Both Are Needed

The scheduler may decide:

```text
this allocation should be reclaimed
```

but execution depends on capabilities.

For example:

```text
DisruptionPolicy:
    reclaim_allowed = true

DisruptionCapabilities:
    checkpoint_supported = true
```

can produce:

```text
REQUEST_CHECKPOINT
    ->
WAIT_FOR_SAFE_RELEASE
    ->
FREE
```

Whereas:

```text
reclaim_allowed = true
checkpoint_supported = false
restart_cost = LOW
```

may produce:

```text
REQUEST_TERMINATION
    ->
FREE
```

The policy answers:

> **May I disrupt this allocation?**

The capabilities answer:

> **If I must disrupt it, what safe transition mechanisms are available?**

This distinction is important for serving, training, batch, and any workflow-based system.

# 22. Temporal Semantics

Time has two different meanings and should not be collapsed into one generic temporal structure.

## 22.1 TemporalRequirements — Demand-Level Timing

`TemporalRequirements` belongs on the plural `ResourceRequirements`.

It describes **when the logical demand needs capacity**, independent of which primary/alternative `ResourceRequirement` the scheduler chooses.

Conceptually:

```text
TemporalRequirements {
    earliest_start?

    required_by?
    deadline?

    valid_from?
    valid_until?
}
```

Examples:

```text
batch evaluation:
    deadline = 20:00

scheduled capacity:
    valid_from = 09:00
    valid_until = 12:00

workflow step:
    earliest_start = 14:00
```

These constraints apply to the logical demand as a whole.

If the scheduler chooses H200 instead of H100, the caller's deadline does not change.

---

## 22.2 RuntimeTemporalProfile — Requirement-Variant Timing

A singular `ResourceRequirement` may have runtime-specific timing characteristics.

Conceptually:

```text
RuntimeTemporalProfile {
    expected_startup_time?

    expected_runtime?

    expected_graceful_release_time?

    expected_checkpoint_time?

    confidence?
}
```

Example:

```text
primary ResourceRequirement:
    8 × H200
    expected_runtime = 2h

alternative ResourceRequirement:
    16 × H100
    expected_runtime = 3.5h
```

If the overall `TemporalRequirements.deadline = 18:00` and current time is 15:00:

```text
H200:
    estimated finish ~= 17:00
    deadline feasible

H100:
    estimated finish ~= 18:30
    deadline infeasible
```

This gives the scheduler useful information for deadline-aware offline scheduling without changing the logical demand.

The distinction is:

> **`TemporalRequirements` says when the caller needs the capacity/result. `RuntimeTemporalProfile` describes time characteristics of one particular way to satisfy it.**

Reservation timing is demand/action-level state, not a property of one accelerator alternative.

---

# 23. ResourceAllocation

## Definition

> **The scheduler's aggregate status and result while attempting to satisfy one ResourceRequirements object.**

This is not merely one concrete GPU placement.

It represents progress against the complete requested instance count.

Conceptually:

```text
ResourceAllocation {
    allocation_id

    resource_group_id
    requirement_id

    desired_instances
    allocated_instances
    pending_instances

    state:
        PENDING
        PARTIALLY_SATISFIED
        SATISFIED
        RELEASING
        RELEASED
        FAILED

    resources_allocated[]

    observed_intent_version
    allocation_generation
}
```

Example:

```text
ResourceRequirements:
    desired ResourceInstances = 50
```

Scheduler currently succeeds for 47:

```text
ResourceAllocation {
    desired_instances = 50

    allocated_instances = 47
    pending_instances = 3

    state = PARTIALLY_SATISFIED

    resources_allocated = [
        ...
        47 entries
        ...
    ]
}
```

So the clean relationship is:

```text
ResourceRequirements
        |
        | scheduler processing
        v
ResourceAllocation
```

---

# 24. ResourceAllocated

## Definition

> **One concrete physical fleet allocation satisfying one copy of the client-defined ResourceInstance.**

Conceptually:

```text
ResourceAllocated {
    resource_allocated_id

    resource_instance_ordinal

    selected_resource_requirement_id

    assigned_resources[]

    assigned_topology_domains[]

    state:
        RESERVED
        ALLOCATED
        DRAINING
        RELEASING
        RELEASED
        LOST

    allocation_epoch

    lease?
}
```

Example for one TP4 replica:

```text
ResourceAllocated #17 {
    resource_instance_ordinal = 17

    assigned_resources = [
        node42/gpu0
        node42/gpu1
        node42/gpu2
        node42/gpu3
    ]

    assigned_topology_domains = [
        node42/NVSwitch0
        rack12
    ]

    state = ALLOCATED
}
```

If the requirement asks for 50 ResourceInstances and the scheduler can place 47:

```text
ResourceAllocation.resources_allocated.size = 47
```

This terminology makes the boundary explicit:

```text
ResourceInstance
    client-defined logical unit

ResourceAllocated
    one scheduler-materialized physical copy

ResourceAllocation
    aggregate scheduler status/result for the full ResourceRequirements
```

---

# 25. Correctness Boundary

Exclusive physical ownership lives at `ResourceAllocated`.

For every active `ResourceAllocated`:

```text
assigned_resources[]
```

must be atomically owned.

Invariant:

> **No exclusive physical GPU may belong to two active ResourceAllocated objects at the same time.**

`ResourceAllocation` is aggregate progress state.

`ResourceAllocated` contains the concrete ownership mapping.

---

# 26. ResourceIntent — Common API Contract

```text
ResourceIntent {
    intent_id
    intent_version
    idempotency_key

    requester_id
    owner_ref

    tenant_id
    queue_ref

    resource_group_id

    action:
        ACQUIRE
        RESIZE
        RELEASE
        RESERVE
        REPLACE

    resource_requirements?

    release_spec?
    replacement_spec?
    reservation_spec?

    created_at
}
```

The public contract contains resource demand and workflow classification references.

During normalization, the scheduler/control plane resolves additional effective metadata such as:

```text
NormalizedResourceIntent {
    ...
    resolved_disruption_capabilities
    effective_disruption_policy
    effective_priority
    effective_queue_policy
}
```

Most caller-visible resource detail now lives naturally inside:

```text
ResourceRequirements
    ->
ResourceInstance
    ->
    ->
CrossInstanceConstraints
    ->
DisruptionPolicy
    ->
TemporalRequirements
```

rather than making `ResourceIntent` itself an overly large flat object.

---

# 27. Public Actions Under the Revised Model

## ACQUIRE

```text
ACQUIRE ResourceRequirements
```

Example:

```text
ResourceInstance:
    TP4 H100 replica

capacity fields:
    desired = 50
```

Scheduler creates/reconciles a `ResourceAllocation`.

---

## RESIZE

Update the `capacity fields` of an existing requirement.

Example:

```text
desired:
    50 -> 70
```

Scheduler reconciles:

```text
ResourceAllocation:
    allocated 50 -> target 70
```

---

## RELEASE

Release:

```text
an entire ResourceAllocation
```

or selected:

```text
ResourceAllocated[]
```

after the owning controller safely drains them.

---

## RESERVE

Reserve future capacity corresponding to:

```text
ResourceRequirements
```

without necessarily materializing all exact `ResourceAllocated` objects immediately.

---

## REPLACE

Targets a concrete:

```text
ResourceAllocated
```

and asks the scheduler to produce another physical allocation satisfying the same logical `ResourceInstance` contract.

This is especially clean for:

```text
node failure
maintenance
rollout
hardware degradation
```

---

# 28. Full Serving Example

```text
ResourceGroup {
    id = serving/llama70b-prod

    requirements = [
        decode
    ]
}
```

Decode:

```text
ResourceRequirements {
    requirement_id = decode

    ResourceInstance {
        member_count = 4

        per_member_resources {
            GPU = 1 × H100_SXM_80GB
            CPU = 8
            RAM = 50GB
        }

        gang {
            atomic = true
            min_members = 4
            desired_members = 4
        }

        intra_instance_placement {
            REQUIRED:
                SAME_NVSWITCH_DOMAIN
        }
    }
    min_instances = 20
    desired_instances = 50
    max_instances = 100
    partial_satisfaction_allowed = true

    CrossInstanceConstraints {
        min_distinct_racks = 5
        max_instances_per_rack = 10
    }
    TemporalRequirements {
        # usually empty for continuously-running serving capacity
    }

    # Effective disruption capabilities/policy are resolved by the
    # workflow control plane and scheduler policy, not authored here.

}
```

Scheduler sees:

```text
Need 50 logical ResourceInstances.

Each ResourceInstance:
    complete atomic TP4 H100 replica.
```

Suppose 47 can currently be placed:

```text
ResourceAllocation {
    requirement_id = decode

    desired_instances = 50
    allocated_instances = 47
    pending_instances = 3

    state = PARTIALLY_SATISFIED

    resources_allocated = [
        ResourceAllocated #0 -> node42/gpu0..3
        ResourceAllocated #1 -> node81/gpu0..3
        ...
        ResourceAllocated #46 -> ...
    ]
}
```

The remaining three stay pending.

---

# 29. Full Training Example

Training controller defines one complete training job as one `ResourceInstance`:

```text
ResourceRequirements {
    requirement_id = trainer

    ResourceInstance {
        member_count = 64

        per_member_resources {
            GPU = 1 × H100
            CPU = ...
            RAM = ...
        }

        gang {
            atomic = true
            min_members = 64
            desired_members = 64
        }

        intra_instance_constraints {
            TP subgroups packed tightly
            preferred same fabric block
        }
    }

    capacity fields {
        min_instances = 1
        desired_instances = 1
        max_instances = 1

        partial_satisfaction_allowed = false
    }
}
```

The scheduler either materializes:

```text
1 ResourceAllocated containing all 64 GPUs
```

or leaves the requirement pending.

It never reports:

```text
0.5 ResourceInstance satisfied
```

for an atomic job.

---

# 30. Final Canonical Vocabulary

```text
ResourceGroup
    = long-lived workflow / job / deployment / business-process ownership scope


ResourceRequirements
    = one logical demand:
      how many logical instances are needed,
      plus primary and alternative acceptable realizations


ResourceRequirement
    = one concrete client/platform-qualified way
      to materialize one logical ResourceInstance


ResourceInstance
    = caller-defined logical allocation unit inside one
      ResourceRequirement

      contains:
        member resources
        gang / atomicity semantics
        intra-instance topology


CrossInstanceConstraints
    = placement / availability relationships among
      the logical instances requested by ResourceRequirements


TemporalRequirements
    = when the logical demand needs capacity/result;
      lives on ResourceRequirements


RuntimeTemporalProfile
    = startup/runtime/checkpoint/release timing characteristics
      of one ResourceRequirement alternative


DisruptionCapabilities
    = workflow/runtime mechanics available for safe interruption;
      resolved from trusted workflow/runtime metadata


DisruptionPolicy
    = scheduler/control-plane policy derived from workflow class,
      queue, tenant, priority, borrowing/reservation state, etc.;
      not authored by the requesting client


ResourceAllocation
    = scheduler's aggregate status/result while satisfying
      one ResourceRequirements


ResourceAllocated
    = one concrete physical fleet allocation satisfying
      one logical instance using one selected ResourceRequirement
```

Visual model:

```text
ResourceGroup
     |
     +-- ResourceRequirements
     |       |
     |       +-- primary_requirement: ResourceRequirement
     |       |       |
     |       |       +-- ResourceInstance
     |       |       +-- RuntimeTemporalProfile?
     |       |
     |       +-- alternative_requirements[]: ResourceRequirement
     |       |       |
     |       |       +-- ResourceInstance
     |       |       +-- RuntimeTemporalProfile?
     |       |
     |       +-- min_instances
     |       +-- desired_instances = 50
     |       +-- max_instances
     |       +-- partial_satisfaction_allowed
     |       |
     |       +-- CrossInstanceConstraints
     |       +-- TemporalRequirements?
     |
     +-- ResourceRequirements
             ...

                     |
                     v

          Intent Normalization / Policy Resolution
                     |
                     +-- effective priority
                     +-- disruption capabilities
                     +-- disruption policy
                     |
                     v

                 Scheduler
                     |
                     v

             ResourceAllocation
                     |
                     +-- ResourceAllocated #1
                     +-- ResourceAllocated #2
                     +-- ...
                     +-- ResourceAllocated #47
                     |
                     +-- pending = 3
```

# 31. Final Phase 2 Structure Invariants

1. **`ResourceGroup` represents the long-lived owning workflow/job/business process.**
2. **A ResourceGroup may contain one or more `ResourceRequirements`.**
3. **`ResourceRequirements` is the logical-demand wrapper: it owns instance count, cross-instance policy, demand-level temporal requirements, one primary requirement, and zero or more alternatives.**
4. **Each singular `ResourceRequirement` is one concrete client/platform-qualified way to satisfy one logical instance.**
5. **`ResourceInstance` is defined by the client/platform inside a ResourceRequirement, not invented by the scheduler.**
6. **A ResourceInstance defines the gang/atomicity and intra-instance topology of one usable unit.**
7. **Alternative ResourceRequirements compete to satisfy the same logical instance demand; their counts are not additive.**
8. **Intra-instance constraints and runtime timing characteristics may differ by ResourceRequirement alternative.**
9. **Cross-instance constraints and demand deadlines belong to ResourceRequirements because they apply across logical instances independent of the selected alternative.**
10. **`min_instances`, `desired_instances`, `max_instances`, and `partial_satisfaction_allowed` live directly on ResourceRequirements.**
11. **Clients do not author disruption policy.**
12. **Trusted workflow/runtime metadata supplies disruption capabilities; platform policy derives effective disruption policy.**
13. **`ResourceAllocation` is aggregate scheduler progress/result for one ResourceRequirements.**
14. **`ResourceAllocated` is one concrete physical allocation satisfying one logical instance using one selected ResourceRequirement.**
15. **Exclusive resource ownership is enforced at `ResourceAllocated`.**
16. **The scheduler may select among declared ResourceRequirement alternatives but may not invent or relax an undeclared alternative.**
17. **The detailed public-action → internal-action decision engine lives separately in `scheduler_workflow_decisions_final.md`.**

# 32. Serving Deployment Controller

Workflow:

```text
1. User requests model deployment.
2. Serving control plane validates model/revision.
3. ServingProfiles are selected.
4. Baseline replica counts are determined.
5. Resource requirementss are compiled.
6. GPU Fleet Scheduler is called.
7. Allocations return.
8. Runtime starts workers.
9. Weights load / distributed runtime initializes.
10. Health checks pass.
11. Replicas register with router.
```

Scheduler touch point: after resource compilation, before physical worker startup.

Mapping:

```text
ACQUIRE
```

or initial long-lived:

```text
RESIZE desired = baseline_capacity
```

---

# 33. Serving Autoscaler

Workflow:

```text
1. Observe backlog / TTFT / TPOT / utilization.
2. Compute desired replica count.
3. Call GPU Scheduler.
4. New allocations arrive or remain pending.
5. Deployment controller starts additions.
6. For scale-down, selected replicas drain.
7. Safe allocations are released.
```

Mapping:

```text
RESIZE
```

Scale-down later maps to:

```text
RELEASE
```

after domain-safe drain.

---

# 34. Rollout / Upgrade Controller

Workflow:

```text
1. New revision qualified.
2. Choose canary/surge step.
3. Request replacement/surge capacity.
4. Start new revision.
5. Wait for readiness.
6. Shift traffic.
7. Drain old revision.
8. Release old allocation.
```

Mapping:

```text
ACQUIRE
```

for surge, or:

```text
REPLACE
```

for one-for-one replacement, followed by `RELEASE`.

---

# 35. Training Job Controller

Workflow:

```text
1. Training workflow submitted.
2. Model/data/checkpoint configuration resolved.
3. Distributed execution shape selected.
4. Checkpoint/preemption capability known.
5. Scheduler called.
6. Allocation returned.
7. Training ranks launched.
8. Training runs.
9. On completion/checkpointed stop, resources released.
```

Fixed training:

```text
ACQUIRE
```

Elastic training:

```text
RESIZE
```

Completion:

```text
RELEASE
```

---

# 36. Batch / Evaluation Controller

Workflow:

```text
1. Workflow creates parallel work.
2. Parallelism/deadline determined.
3. Resource requirements selected.
4. Scheduler capacity requested.
5. Workers start as capacity arrives.
6. Work completes.
7. Allocations released.
```

Mapping:

```text
ACQUIRE
```

or elastic:

```text
RESIZE
```

Completion:

```text
RELEASE
```

Future-guaranteed capacity may use `RESERVE`.

---

# 37. Interactive / Notebook Controller

Workflow:

```text
1. User requests GPU session.
2. Policy resolves accelerator/lease.
3. Scheduler called.
4. Allocation returned.
5. Runtime starts.
6. Lease expires or user exits.
7. Resource released.
```

Mapping:

```text
ACQUIRE
```

then:

```text
RELEASE
```

---

# 38. Capacity Reservation Controller

Workflow:

```text
1. Future launch/training event planned.
2. Capacity shape/time determined.
3. Scheduler called ahead of execution.
4. Capacity protected.
5. Backfill may use safe reclaimable capacity beforehand.
6. Actual consumer acquires against reservation.
7. Reservation expires/releases.
```

Mapping:

```text
RESERVE
```

Later consumer uses:

```text
ACQUIRE / RESIZE
```

with `reservation_ref`.

---

# 39. Failure / Recovery Controller

Workflow:

```text
1. Lost/unhealthy allocation detected.
2. Desired state becomes under-satisfied.
3. Replacement capacity requested.
4. Scheduler places replacement.
5. Runtime starts/rejoins replacement.
6. Group returns to desired state.
```

Mapping:

```text
REPLACE
```

or desired-state:

```text
RESIZE
```

when only cardinality matters.

---

# 40. LLM Router Is Normally Not a Direct Scheduler Actor

The router emits demand signals:

```text
queue depth
TTFT
TPOT
outstanding tokens
capacity pressure
```

Path:

```text
LLM Router
    -> Serving Autoscaler
    -> RESIZE ResourceIntent
    -> GPU Fleet Scheduler
```

Router decisions happen in milliseconds; GPU capacity changes happen over seconds/minutes. Keeping the loops separate avoids instability.

---

# 41. Maintenance Controller Is Supply-Side

Maintenance changes supply:

```text
Drain node N42
Drain rack R7
Firmware maintenance at 18:00
```

It updates `OperationalState` through FleetControl, not `ResourceIntent`.

---

# 42. Actor-to-Action Mapping

| Actor | Workflow point where scheduler is called | Public scheduler action |
|---|---|---|
| Serving Deployment Controller | After ServingProfile/resource compilation, before worker startup | `ACQUIRE` or initial `RESIZE` |
| Serving Autoscaler | After desired replica count is computed | `RESIZE`; later `RELEASE` after drain |
| Rollout/Upgrade Controller | Before starting surge/replacement revision | `ACQUIRE` or `REPLACE`; later `RELEASE` |
| Training Job Controller | After distributed shape is chosen, before rank launch | `ACQUIRE`; elastic `RESIZE`; completion `RELEASE` |
| Batch/Evaluation Controller | After parallelism/deadline planning, before workers launch | `ACQUIRE` or `RESIZE`; completion `RELEASE` |
| Interactive/Notebook Controller | Before session startup | `ACQUIRE`; end/expiry `RELEASE` |
| Reservation Controller | Before future capacity window | `RESERVE`; consumer later `ACQUIRE/RESIZE` |
| Failure/Recovery Controller | After allocation loss/drain is detected | `REPLACE` or desired-state `RESIZE` |
| LLM Router | Normally never directly | Signal to Autoscaler |
| Maintenance/Fleet Ops | Supply-side | FleetControl / OperationalState mutation |

---

# 43. Result Contract

```text
ResourceIntentStatus {
    intent_id
    intent_version

    resource_group_id

    state:
        ACCEPTED
        PENDING
        PARTIALLY_SATISFIED
        SATISFIED
        REJECTED
        CANCELLED

    desired_instances?
    allocated_instances?
    pending_instances?

    allocations[]
    reservations[]

    release_candidates[]

    reason_code?
    retryability?

    observed_at
}
```

This lets upstream actors reconcile their workflow without the scheduler understanding the workflow.

---

# 44. Phase 2 State Ownership

## Actor owns

```text
why capacity is needed
domain workflow
model/training semantics
autoscaling calculation
rollout progression
checkpoint/drain behavior
```

## GPU Scheduler owns

```text
resource intent state
resource-group desired capacity
resource-shape interpretation
resource reservation
physical allocation ownership
release/replacement resource semantics
```

## Runtime / Reconciler owns

```text
starting processes/pods
loading models/checkpoints
distributed rendezvous
readiness
domain-safe termination
```

---

# 45. Important API Semantics

## Versioned

Newer desired-state versions supersede older ones.

## Idempotent

Retrying the same intent does not duplicate allocation.

## Property-Oriented

Prefer:

```text
8 H100
same NVSwitch domain
```

not exact GPU IDs.

## Validated Shapes Only

Scheduler chooses among certified shapes; it never invents model/training configurations.

## Hard vs Soft Constraints

Hard controls feasibility. Soft controls ranking.

---

# 46. Locked Phase 2 Invariants

1. GPU Fleet Scheduler is a resource broker, not a workflow engine.
2. Consumer workflows translate domain semantics into `ResourceIntent`.
3. Demand-plane requests are separate from supply-plane fleet mutations.
4. Public actions are `ACQUIRE`, `RESIZE`, `RELEASE`, `RESERVE`, and `REPLACE`.
5. Internal mechanisms such as `RECLAIM` and `PREEMPT` are not caller-controlled.
6. `ResourceGroup` is the stable ownership/desired-state scope.
7. `ResourceRequirements` is one platform-qualified way to create one usable resource instance.
8. `ResourceInstance` is logical capacity; `ResourceAllocation` is physical ownership.
9. Scheduler may choose among qualified ResourceInstance alternatives but never invent a new one.
10. Desired-state updates are versioned and idempotent.
11. Hard constraints determine feasibility; preferences affect ranking only.
12. Actor owns safe runtime transitions such as drain/checkpoint; scheduler owns resource ownership.
13. Temporary scarcity yields pending/queueing, not invalid intent.
14. Request routing normally influences fleet capacity through an autoscaler, not directly.
15. Phase 2 does not decide tenant entitlement; quota/fair-share begins in Phase 3.

---

# 47. Principal-Level Summary

> **I expose the GPU Fleet Scheduler as a resource broker with a small stable action vocabulary: Acquire, Resize, Release, Reserve, and Replace. Higher-level controllers—serving deployment, autoscaling, rollout, training, batch, notebooks, reservations, and recovery—decide why capacity is needed and translate that into a common versioned `ResourceIntent`. The scheduler operates on stable `ResourceGroup`s composed of platform-qualified `ResourceRequirements`s; one logical `ResourceInstance` becomes a strongly owned physical `ResourceAllocation`. The public action describes the desired resource outcome, while internal mechanisms such as Allocate, Backfill, Consolidate, Reclaim, or Preempt remain scheduler policy. That keeps the scheduler ignorant of workflow semantics while still giving it all resource, topology, disruption, priority, and temporal information required to schedule correctly.**
