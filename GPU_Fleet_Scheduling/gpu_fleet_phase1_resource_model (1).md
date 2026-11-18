# Phase 1 — GPU Fleet Resource Model

## 1. Why This Phase Exists

Before quotas, gangs, topology placement, preemption, or autoscaling, the scheduler needs a precise answer to:

> **What resources exist, how are they connected, what is schedulable now, and what makes two apparently free GPUs non-equivalent?**

A flat view such as:

```text
Cluster:
    8,000 GPUs
Free:
    600 GPUs
```

is insufficient.

A workload may require:

```text
8 × H100 80GB
TP = 8
same NVLink/NVSwitch domain
RDMA-capable networking
400 GB host RAM
healthy/non-draining nodes
```

The cluster may have 600 free GPUs but zero feasible placements of that exact shape.

The scheduler therefore manages **versioned, topology-aware, multi-resource capacity**, not a flat GPU counter.

---

## 2. Locked Design Principles

1. GPUs are not interchangeable counters.
2. Resource flavor and topology are separate dimensions.
3. Static capability is separate from dynamic allocation state.
4. Hardware capability is separate from empirical performance.
5. Physically free capacity is not necessarily schedulable capacity.
6. GPU, CPU, RAM, NIC, and storage form one resource vector.
7. Health, maintenance, drain state, and reservations reduce schedulable capacity.
8. Topology is hierarchical, not a global all-pairs preference ranking.
9. Warm model/image/cache state is an optimization hint, not hardware identity.
10. Scheduling reads a versioned snapshot; exclusive ownership requires an atomic commit.
11. Strong consistency is required for ownership, not for every telemetry signal.
12. Aggregate free GPU count is insufficient because fragmentation and topology matter.

---

## 3. Final Logical Resource Model

```text
GPUFleetResourceSnapshot {
    snapshot_version
    observed_at

    resource_flavors[]
    topology_domains[]

    nodes[]
    allocations[]
    reservations[]

    health_state
    operational_state
}
```

The model has five core layers:

```text
1. Resource Identity
2. Static Hardware Capability
3. Physical Topology
4. Dynamic Capacity / Ownership
5. Health + Operational State
```

Dynamic performance and warm-state information are intentionally separate views.

---

## 4. Resource Identity

Each physical GPU needs a stable identity.

```text
GPUResourceID {
    cluster_id
    node_id
    device_uuid
    local_ordinal
}
```

Human-readable form:

```text
cluster-a/node-0042/gpu-3
```

We do not globally number GPUs `GPU-1 ... GPU-8000` and maintain a permanent peer-preference list.

Why?

```text
Identity:
    Which physical resource is this?

Topology:
    Where is it relative to other resources?

Placement policy:
    Is that locality desirable for this workload?
```

These are different questions.

---

## 5. Resource Flavor

A `ResourceFlavor` describes what kind of accelerator resource a GPU belongs to.

```text
ResourceFlavor {
    flavor_id

    gpu_type
    gpu_architecture
    gpu_memory_bytes
    form_factor

    mig_capabilities?

    cost_class?
    ownership_class?
}
```

Examples:

```text
H100_SXM_80GB
H100_PCIE_80GB
H200_SXM
B200
A100_PCIE_80GB
```

Why?

Because:

```text
8 × A100
8 × H100
8 × H100 PCIe
8 × H100 SXM + NVSwitch
```

are not equivalent scheduling resources.

Later policy can express:

```text
required:
    H100_SXM_80GB

preferred:
    H200

fallback:
    H100
```

Queues/quotas may also eventually be flavor-specific.

---

## 6. Node Hardware Profile

A GPU workload consumes more than GPU count.

```text
NodeHardwareProfile {
    node_id

    gpu_resources[]

    cpu_count
    cpu_architecture

    host_memory_bytes

    nic_resources[]
    local_nvme[]

    rdma_capable
    gpudirect_rdma_capable?

    pcie_topology_ref
}
```

Example:

```text
Node N42
-----------------------------
8 × H100 80GB
2 CPU sockets
1.5 TB host RAM
2 × 400G NIC
RDMA capable
8 TB local NVMe
```

A workload can fit by GPU count but still be infeasible due to insufficient RAM/CPU, missing RDMA, missing storage, or the wrong accelerator class.

---

## 7. Multi-Resource Capacity Vector

```text
ResourceVector {
    gpu_by_flavor

    cpu_cores
    host_memory_bytes

    nic_capacity
    rdma_nics

    nvme_bytes

    mig_resources?
}
```

Keep these concepts distinct:

```text
PhysicalCapacity
AllocatableCapacity
AllocatedCapacity
ReservedCapacity
SchedulableCapacity
```

Example:

```text
Physical:
    8 GPUs

Platform-reserved:
    1

Allocated:
    4

Reserved for rollout:
    2

Schedulable for unrelated work:
    1
```

---

## 8. Physical Topology

Topology describes where resources sit relative to one another.

Useful hierarchy:

```text
Region
  |
Zone
  |
SuperPod / Fabric Block
  |
Rack
  |
Node
  |
NVLink / NVSwitch Domain
  |
NUMA Domain
  |
GPU
```

Network-specific metadata may also include:

```text
InfiniBand / RoCE fabric
network rail
NIC affinity
PCIe root
```

Conceptually:

```text
TopologyLocation {
    region
    zone

    fabric_block
    network_fabric_id

    rack
    node

    nvlink_domain
    numa_domain

    preferred_nic
    network_rail
}
```

Example:

```text
GPU node-42/gpu-3

Zone            = us-west-a
FabricBlock     = B7
Rack            = R12
Node            = N42
NVLinkDomain    = N42/NVSwitch0
NUMADomain      = NUMA1
NICAffinity     = NIC1
Rail            = Rail-B
```

---

## 9. Why Topology Is First-Class

Two placements can both satisfy:

```text
8 × H100
```

but differ drastically.

Placement A:

```text
8 GPUs in one NVSwitch domain
```

Placement B:

```text
8 GPUs scattered across four nodes
```

For TP8, A can be far better because collectives stay on the tight local fabric.

Topology also affects:

```text
PP activation movement
EP all-to-all
DP collectives
P->D KV transfer
GPU-to-NIC communication
failure isolation
```

So topology cannot be added after scheduling; it belongs in the resource model.

---

## 10. Topology Domains Instead of Pairwise GPU Ranking

For an 8,000-GPU cluster, do not maintain a primary 8,000 × 8,000 preference matrix.

Maintain hierarchical indexes:

```text
NVSwitchDomain -> GPUs
Node           -> GPUs
Rack           -> Nodes
FabricBlock    -> Racks
Zone           -> Blocks
```

Conceptual domain object:

```text
TopologyDomain {
    domain_id
    level

    parent_domain_id?

    member_resources
    resource_capacity
}
```

Communication locality can then be classified as:

```text
same NVLink/NVSwitch domain
same node
same rack
same fabric block
cross block
cross zone
```

---

## 11. Interconnect / Fabric Profile

```text
FabricProfile {
    fabric_id

    type:
        INFINIBAND
        ROCE
        ETHERNET

    bandwidth_class
    latency_class

    rdma_capable

    oversubscription_class?
    rail_id?
}
```

Vocabulary:

```text
NVLink
    high-bandwidth NVIDIA GPU interconnect

NVSwitch
    switched fabric connecting many NVLink endpoints

InfiniBand / IB
    high-performance inter-node network with native RDMA

RoCE
    RDMA over Ethernet

RDMA
    direct remote-memory transfer capability

GPUDirect RDMA
    direct GPU-memory <-> NIC data path
```

The resource model records physical capability/topology, not assumed runtime performance.

---

## 12. Static Capability vs Empirical Performance

A NIC advertising `400 Gb/s` is a capability statement, not guaranteed application throughput.

Actual effective bandwidth depends on:

```text
topology
message size
communication pattern
participant count
software stack
congestion
link health
```

Therefore keep three layers:

### Hardware / Topology Capability

```text
400G RoCE NIC
same rack
same fabric block
```

### Telemetry

```text
NIC utilization = 72%
ECN elevated
rail B degraded
```

### Empirical Performance Index

```text
cross-rack same-block
RoCE
point-to-point
16-32 GB transfer

effective BW P50 = 320 Gb/s
conservative BW = 270 Gb/s
```

The scheduler should use empirical performance for optimization rather than blindly plugging nominal hardware capacity into cost equations.

A useful model:

```text
PredictedCommunicationCost
=
EmpiricalBaseline(
    topology,
    transfer type,
    bytes
)
+
BoundedDynamicAdjustment(
    current congestion / health
)
```

---

## 13. CommunicationCost(A, B)

`A` and `B` are endpoints at the level relevant to the workload.

Individual GPUs:

```text
A = node-42/gpu-0
B = node-42/gpu-5
```

Whole serving groups:

```text
A = Prefill replica P3
B = Decode replica D7
```

Parallelism groups:

```text
A = TP group 0
B = TP group 1
```

Coarse domains during pruning:

```text
A = rack-12
B = rack-14
```

Conceptually:

```text
CommunicationCost(
    source_endpoint_or_group,
    destination_endpoint_or_group,
    communication_pattern,
    current_network_state
)
```

Approximation:

```text
T_comm
≈
setup_latency
+
bytes / effective_bandwidth
+
collective_overhead
```

Static topology supplies the baseline, telemetry a bounded correction, and the workload determines how much communication matters.

---

## 14. Allocation State

Allocation is fast-changing ownership state.

```text
Allocation {
    allocation_id
    workload_id
    tenant_id

    resources[]
    topology_domains[]

    allocation_epoch
    state
}
```

Example:

```text
N42/GPU0 -> Workload W17
N42/GPU1 -> Workload W17
N42/GPU2 -> FREE
N42/GPU3 -> Reserved W29
```

This state requires strong consistency. Two exclusive workloads must never successfully own the same GPU.

---

## 15. Reservation State

Reservations are first-class capacity commitments.

```text
Reservation {
    reservation_id
    owner

    resource_shape
    topology_constraints

    resources?
    domain?

    valid_from
    expires_at

    state
}
```

A GPU can be physically idle yet unavailable because it is being accumulated for a gang, a rollout, a future reservation, or quota recovery.

---

## 16. Health State

Kubernetes `Node Ready=True` is insufficient for GPU fleets.

```text
HealthState {
    node_health

    gpu_health[]
    nvlink_health
    nic_health
    fabric_health

    observed_at
}
```

Possible states:

```text
HEALTHY
DEGRADED
UNHEALTHY
UNKNOWN
```

Signals can include:

```text
ECC errors
XID errors
NVLink failure
NIC failure
fabric degradation
thermal issues
```

Health may be a hard filter or a soft penalty depending on severity.

---

## 17. Operational State

A resource can be healthy but unsuitable for new work.

```text
OperationalState {
    schedulable

    draining
    maintenance_pending

    maintenance_at?
    expected_return_at?
}
```

Examples:

```text
HEALTHY + SCHEDULABLE
HEALTHY + DRAINING
HEALTHY + MAINTENANCE_IN_30_MIN
UNHEALTHY
```

This later supports maintenance-aware placement, graceful drain, rollout, and backfill.

---

## 18. Dynamic Performance State Is Separate

Do not mix these into static inventory:

```text
GPU utilization
NIC utilization
fabric congestion
temperature
collective latency
```

Keep:

```text
FleetInventory
    physical capability

FleetAllocationState
    ownership

FleetTelemetryState
    raw dynamic observations

FleetPerformanceIndex
    derived empirical scheduling models
```

This mirrors the LLM-routing distinction between `ServingProfile` and `PerformanceState`.

---

## 19. Warm State Is Separate

Transient affinity hints include:

```text
model weights loaded
container image cached
compiled kernels present
CUDA graphs ready
```

These belong in something like:

```text
WarmStateIndex
```

not hardware inventory.

```text
"This node is H100"
    = capability

"This node already has Llama-70B warm"
    = transient optimization opportunity
```

---

## 20. Scheduler-Optimized Indexes

The conceptual model is graph-shaped, but implementation should use indexed read views.

Likely indexes:

```text
NodeStore

FlavorIndex
    H100 -> candidate nodes/domains

TopologyDomainIndex
    NVSwitchDomain -> GPUs
    Rack -> Nodes
    Fabric -> Racks

CapacityByDomainIndex

AllocationIndex
ReservationIndex
HealthIndex
OperationalStateIndex

FleetPerformanceIndex

optional WarmStateIndex
optional TelemetryIndex
```

This lets scheduling narrow a 100K-GPU fleet into a bounded candidate set instead of scanning every GPU.

---

## 21. Versioned Fleet Resource Snapshot

Scheduling operates on a versioned observation.

```text
FleetResourceSnapshot {
    snapshot_version

    inventory_version
    topology_version

    allocation_version
    reservation_version

    health_version
    operational_version

    observed_at
}
```

The snapshot is not ownership.

Concurrency boundary:

```text
Read snapshot
      |
      v
Compute placement
      |
      v
Atomic Reserve / Allocate
```

If someone else wins the GPUs first:

```text
commit fails
    ->
refresh / replan
```

This is the fleet-scheduling equivalent of `Planner observation != TryClaim` in the LLM Router.

---

## 22. Consistency Requirements

### Strong consistency

```text
allocation ownership
reservation ownership
exclusive GPU assignment
```

### Eventual / cached consistency is acceptable

```text
GPU utilization
network congestion
temperature
warm-state hints
empirical performance observations
```

Staleness can create a poorer placement, but must not create double ownership.

### Slow-changing configuration

```text
resource flavor
physical topology
failure domains
hardware capability
```

is discovered/configured through the control plane and changes infrequently.

---

## 23. Example — 1,000 Nodes / 8,000 GPUs

Suppose:

```text
1,000 nodes
8 × H100 per node
8,000 GPUs total

1 NVSwitch domain/node
20 nodes/rack
10 racks/fabric block
5 fabric blocks
```

The system maintains roughly:

```text
8,000 GPU identities
1,000 NVSwitch domains
1,000 nodes
50 racks
5 fabric blocks
```

Request:

```text
Llama-70B Decode replica

TP = 8

requires:
    8 × H100
    same NVSwitch domain
    RDMA-capable node

prefers:
    rack different from existing replicas
```

Later scheduling:

```text
H100 flavor lookup
    |
candidate NVSwitch domains
    |
filter >= 8 schedulable GPUs
    |
filter reservations / health / drain
    |
score:
    fragmentation
    failure spread
    warm state
    empirical communication performance
    |
choose domain
    |
atomic reservation
```

No global ranking of all 8,000 GPUs is required.

---

## 24. Multiple-Replica Example

Request:

```text
4 Decode replicas
each TP8
```

Constraints:

```text
within replica:
    PACK inside one NVSwitch domain

between replicas:
    SPREAD across >= 3 racks
```

At one topology level we want `PACK` for communication.

At another we want `SPREAD` for failure isolation.

That is why topology must be hierarchical.

---

## 25. Locked Phase 1 Whiteboard Model

```text
GPUFleetResourceModel
|
+-- ResourceFlavorRegistry
|
+-- NodeInventory
|      |
|      +-- GPUResource
|      +-- CPU
|      +-- Memory
|      +-- NIC
|      +-- NVMe
|
+-- TopologyModel
|      |
|      +-- Zone
|      +-- FabricBlock
|      +-- Rack
|      +-- Node
|      +-- NVLinkDomain
|      +-- NUMADomain
|      +-- NIC / Rail affinity
|
+-- AllocationState
|
+-- ReservationState
|
+-- HealthState
|
+-- OperationalState
|
+-- [separate] TelemetryState
|
+-- [separate] FleetPerformanceIndex
|
+-- [separate] WarmState / PlacementAffinity
```

Exposed through a versioned `FleetResourceSnapshot` and protected by an atomic `ResourceReservation / Allocation` boundary.

---

# 26. Briefly: Subsystems Needed to Realize the Model

The logical resource model comes first. From it we derive the implementation/control-plane pieces.

### Resource Discovery Agent
Discovers GPU identity/type/memory, CPU/RAM, NIC/RDMA, NVLink/NVSwitch, PCIe/NUMA, and local storage.

### Resource Inventory Store / Registry
Authoritative store for slow-changing nodes, flavors, and hardware capabilities.

### Topology Discovery / Registry
Builds and indexes zone → fabric → rack → node → NVLink/NUMA/NIC relationships.

### GPU / Fabric Health Monitor
Publishes GPU, NVLink, NIC, and fabric health relevant to schedulability.

### Allocation / Reservation Registry
Strongly consistent owner of exclusive allocations, reservations, epochs, and leases.

### Operational State Controller
Owns drain, maintenance, administrative exclusions, and expected-return state.

### Telemetry Pipeline
Collects high-volume GPU/network/fabric measurements.

### Empirical Performance Modeler + FleetPerformanceIndex
Turns raw observations into scheduler-ready empirical compute/communication performance distributions.

Key boundary:

```text
ResourceSnapshot
    determines feasibility

PerformanceIndex
    helps rank feasible placements
```

### WarmState / PlacementAffinity Index
Tracks transient model/image/kernel locality as a soft optimization signal.

### Fleet State Aggregator / Snapshot Cache
Builds scheduler-optimized versioned read views so scheduling does not fan out to all control-plane systems.

### Resource Allocation Service
Executes the atomic transition from planned placement to owned/reserved resources; stale plans fail and replan.

---

## Final Interview Summary

> **I model a GPU fleet as versioned, topology-aware, multi-resource capacity rather than a flat GPU count. Resource flavor describes what hardware a resource is; topology describes where it sits; allocation and reservation state describe ownership; health and operational state determine whether it is schedulable. Dynamic utilization and empirical bandwidth belong in separate telemetry/performance views because nominal hardware capability is not actual application performance. The scheduler reads indexed snapshots for fast planning, but exclusive GPU ownership is committed through a strongly consistent reservation/allocation boundary. From that logical model I derive the discovery, topology, health, allocation, telemetry/performance, snapshot, and commit subsystems needed to keep the model trustworthy.**
