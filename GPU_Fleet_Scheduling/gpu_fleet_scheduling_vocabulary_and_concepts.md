# GPU Fleet Scheduling — Vocabulary and Core Concepts
## Reference Primer Before Architecture Design

**Scope:** This document defines the vocabulary and first-principles concepts needed to reason about a large multi-tenant GPU fleet.

The eventual platform must support both:

```text
ONLINE
======
LLM / ML inference services
long-running replicas
latency and availability SLOs
autoscaling
careful disruption

OFFLINE
=======
training
fine-tuning
batch inference
evaluation
GPU workflows
job completion time / fairness
checkpointing and preemption
```

We will design **online inference first**, then reuse and extend these concepts for offline workloads.

The nine online fleet-scheduling phases are locked as:

```text
0. GPU Fleet Resource Model
1. Workload / Resource Intent
2. Tenant Admission / Quota / Fair Share
3. Workload Priority
4. Gang Formation
5. Topology-Aware Placement
6. Fragmentation / Bin Packing
7. Preemption / Reclaim / Consolidation
8. Autoscaling / Capacity Control
9. Lifecycle / Rollout / Drain
```

This document intentionally does **not** define our final scheduler components yet.

---

# 1. Request Routing vs Fleet Scheduling

The LLM Router answers:

```text
Which already-running model replica should serve request R42?
```

The GPU Fleet Scheduler answers:

```text
Which workloads are entitled to GPU capacity,
how many GPUs,
of what kind,
and where should those GPUs physically come from?
```

Example:

```text
Fleet Scheduler:
    Search Llama-70B Decode pool
        -> 12 replicas
        -> each TP4
        -> H100
        -> TP ranks packed inside NVLink domains
        -> independent replicas spread across racks

LLM Router:
    request R42
        -> Decode replica D7
```

The router cannot route to capacity that has not first been admitted, allocated, placed, started and made healthy.

---

# 2. Scheduling Unit / Workload

The **scheduling unit** is the smallest entity the scheduler reasons about atomically at a particular layer.

Examples:

```text
one Pod
one GPU process
one TP4 model replica
one distributed training job
one Prefill+Decode scaling group
one whole inference service
```

A critical design question is:

> **At which level should allocation be atomic?**

For a TP4 replica, four GPU ranks together may be the correct atomic unit. For eight independent TP4 replicas, all 32 GPUs usually should not be one hard gang.

This leads to hierarchical scheduling units, which Grove/KAI model using PodCliques, scaling groups, PodGroups and subgroups.

---

# 3. Queue

## What is a Queue?

A queue is not merely a FIFO list of pending jobs.

In a multi-tenant GPU platform, a queue is usually a **resource-governance boundary**.

It groups workloads that share:

```text
ownership
quota
priority policy
fairness rules
preemption rules
resource limits
```

Example:

```text
Company
|
+-- Search AI
|    |
|    +-- Production Inference
|    +-- Experimental Inference
|    +-- Offline Training
|
+-- Ads AI
|
+-- Messaging AI
```

Queues map organizational structure into resource policy.

---

# 4. Why Queues Are Needed

Imagine a 1,000-H100 fleet with no queues.

Every workload enters one global competition. A burst from one team can consume most of the fleet, causing starvation and making capacity ownership unpredictable.

Queues let the platform express:

> **Before individual workloads compete, how should the organization divide the fleet?**

That is why queue-level allocation and workload-level ordering should remain conceptually separate.

---

# 5. Hierarchical Queues

Large organizations are hierarchical.

Example:

```text
AI Platform
    |
    +-- Search
    |     |
    |     +-- Ranking
    |     +-- GenAI
    |
    +-- Ads
    |
    +-- Trust
```

A hierarchical queue lets resource policy recurse through the organization.

Example:

```text
AI organization: 800 H100 entitlement

Search: 400
Ads: 250
Trust: 150

Within Search:
    Ranking: 150
    GenAI: 250
```

Important property:

> Fairness should often be calculated among siblings, then recursively within each child.

---

# 6. Quota / Guaranteed Capacity

A quota answers:

> **What capacity is this tenant entitled to when it has demand?**

Example:

```text
Search guaranteed = 300 H100
Ads guaranteed = 300 H100
Messaging guaranteed = 200 H100
```

If Search uses only 100 today, its unused 200 need not sit idle. This immediately creates borrowing, lending and reclaim.

A quota is commonly a **guarantee**, not necessarily a hard partition.

---

# 7. Hard Limit / Maximum

Separate guaranteed quota from maximum allowed usage.

Example:

```text
Search:
    guaranteed = 300
    max = 500
```

Interpretation:

```text
Search can rely on 300 when it needs them.
Search may consume up to 500 if spare capacity exists.
```

This avoids both strict static partitioning and unbounded opportunistic consumption.

---

# 8. Borrowing

Borrowing means:

> **A tenant temporarily consumes unused quota owned by another tenant or shared cohort.**

Example:

```text
Search quota = 300
Ads quota = 300
Ads uses 100
Search demand = 450
```

Search may run:

```text
300 in-quota
+
150 borrowed
```

Total utilization improves without permanently changing ownership.

Kueue models ClusterQueues grouped into a Cohort, with explicit borrowing and lending limits.

---

# 9. Lending

Lending is the mirror image of borrowing.

It answers:

> **How much of my currently unused quota may other queues consume?**

Why limit lending? A tenant may want capacity to remain quickly reclaimable for production traffic, launches, or failure reserve.

A quota can therefore conceptually have:

```text
nominal quota
borrowing limit
lending limit
```

rather than being one scalar.

---

# 10. In-Quota vs Over-Quota

```text
IN-QUOTA
========
capacity covered by the tenant's guaranteed entitlement

OVER-QUOTA
==========
borrowed / surplus capacity beyond the guaranteed entitlement
```

This distinction matters because disruption policy can differ.

Example:

```text
in-quota production replica:
    strongly protected

over-quota experimental replica:
    reclaimable
```

---

# 11. Fair Share

Quota says:

```text
what is guaranteed?
```

Fair share asks:

> **How should surplus capacity be divided when multiple tenants want more than their guarantees?**

Example:

```text
100 spare GPUs
Search wants +100
Ads wants +100

equal weight:
    Search +50
    Ads +50

weights 2:1:
    Search ~67
    Ads ~33
```

Fair share prevents first requester wins everything and gives predictable organizational behavior.

---

# 12. Dominant Resource Fairness — DRF

When the cluster contains multiple scarce resources, simple GPU counts may not represent pressure.

A workload may request:

```text
8 GPUs
64 CPUs
1 TB RAM
high-speed NICs
```

Another may request:

```text
4 GPUs
256 CPUs
2 TB RAM
```

DRF computes each tenant's share of each resource and considers its largest, dominant share.

Simplified:

```text
GPU share    = tenant GPUs / fleet GPUs
CPU share    = tenant CPU / fleet CPU
memory share = tenant memory / fleet memory

DominantShare = max(all shares)
```

Fairness tries to balance dominant shares.

This prevents a tenant from appearing fair by GPU count while monopolizing another resource required to feed those GPUs.

---

# 13. Historical / Time-Based Fairness

Instantaneous fairness can still be unfair over time.

Example:

```text
Team A used 500 GPUs all night.
Team B used 0.
At 9 AM both request 500 GPUs.
```

An instantaneous 50/50 split ignores A's prior consumption.

Time-based fair share maintains decayed historical usage:

```text
HistoricalUsage(t)
=
decayed previous usage
+
recent resource consumption
```

A tenant that consumed less historically may receive preference.

This is especially important for offline workloads, but belongs in the common vocabulary.

---

# 14. Admission

Admission asks:

> **May this workload enter the set of workloads allowed to consume cluster capacity?**

This occurs before exact node placement.

Kueue makes this separation especially explicit:

```text
Quota Reservation
    !=
Pod-to-node scheduling
```

Why separate them? Tenant policy, quota, fairness, resource flavor and gang requirements can often be decided at a higher level than exact node selection.

---

# 15. Queueing

Queueing is the period:

```text
workload submitted
        ->
workload admitted / allocated
```

A workload waits because quota is unavailable, its gang cannot fit, required topology is unavailable, higher-priority work exists, or the required GPU flavor is unavailable.

Queueing policy determines:

```text
who waits
how long
who moves ahead
whether old work eventually gets priority
```

For online inference, long queueing of desired replicas indirectly becomes user-facing latency because capacity expansion cannot materialize. For offline jobs, queue wait is directly part of job completion time.

---

# 16. FIFO

FIFO means oldest submitted workload first.

Advantages:

```text
simple
predictable
easy to explain
```

Problems:

```text
huge unschedulable job can block smaller work
priority ignored
tenant fairness ignored
```

Variants commonly use priority first, then FIFO within a priority class.

---

# 17. Head-of-Line Blocking

Suppose queue order is:

```text
Job A needs 64 contiguous GPUs
Job B needs 1 GPU
Job C needs 1 GPU
```

Only eight fragmented GPUs are free. Strict FIFO leaves B and C waiting although they could run.

Schedulers address this with backfill, skip policies, reservations and multiple queues while preserving some fairness guarantee for A.

---

# 18. Backfill

Backfill means:

> **Use temporarily idle resources for work that will not prevent an important waiting workload from eventually starting.**

Example:

```text
Large gang A waits for 64 GPUs.
16 GPUs are currently free.
Run short/preemptible B/C jobs there instead of leaving them idle.
```

Backfill is especially valuable for offline work and spare online capacity.

---

# 19. Reservation

A reservation protects future capacity for a workload.

Example:

```text
TP64 rollout needs 64 GPUs in one block.
40 are free now.
24 will drain within 10 minutes.
```

Without a reservation, smaller workloads can continuously consume newly released GPUs and the 64-GPU shape never forms.

Reservation accumulates resources for the target workload. Backfill can still use them if it guarantees not to delay the target.

---

# 20. Priority

Priority answers:

> **Among workloads competing inside a governance scope, which work is more important?**

Example:

```text
production inference = 1000
canary inference     = 700
evaluation           = 400
training             = 200
```

Priority may affect queue ordering, admission or preemption.

Queue-level fairness and workload priority should remain distinct.

---

# 21. Preemptibility

Priority and preemptibility are related but should not be identical.

Two distinct questions are:

```text
Who should run first?
Who can safely be interrupted?
```

A latency-sensitive service may be non-preemptible even if another workload has higher organizational priority.

---

# 22. Preemption

Preemption means:

> **Evict already-running work so higher-value work can obtain its resources.**

Example:

```text
cluster full
high-priority inference needs 8 H100
lower-priority evaluation owns 8 H100
```

A mature victim cost should include:

```text
RestartCost
+
CheckpointLoss
+
ModelWarmupCost
+
KVStateLoss
+
TrafficImpact
+
TimeAlreadyRun
```

Online inference generally has a higher disruption cost than restartable offline jobs.

---

# 23. Reclaim

Reclaim is conceptually different from preemption.

Reclaim means:

> **Return borrowed/over-quota capacity to the tenant that is entitled to it.**

Example:

```text
Ads quota = 300
Ads currently uses 100
Search borrowed 150 Ads GPUs
Ads later requests its guaranteed capacity
```

The scheduler may reclaim Search's borrowed capacity.

KAI uses a particularly clean vocabulary: preemption is priority enforcement within a queue; reclaim recovers over-quota resources across queues.

---

# 24. Minimum Guaranteed Runtime

Even a preemptible workload should sometimes receive time to make useful progress.

Otherwise:

```text
start
-> evict after 5 sec
-> restart
-> evict
```

causes thrashing.

A minimum runtime protects a recently started workload from reclaim/preemption for a bounded interval.

---

# 25. Gang Scheduling

Gang scheduling means:

> **Allocate enough members of a coordinated workload together, or do not start it.**

Example: a TP8 model replica. Starting ranks 0–3 while 4–7 remain pending is useless and wastes four GPUs.

Gang scheduling provides all-or-nothing or minimum-viable-group semantics.

---

# 26. Why Gang Scheduling Is Required for AI

Distributed AI workloads often have barriers, collectives, rank dependencies, leader/worker startup and fixed TP/PP degrees.

Partial placement can create idle resources, deadlocks or startup failure.

Gang scheduling turns a set of processes into one schedulable resource shape.

---

# 27. Hierarchical Gang Scheduling

Modern inference services are more complex than one flat gang.

Example:

```text
Inference Service
|
+-- Router
+-- Prefill pool
|    +-- P replica 1 = TP4 gang
|    +-- P replica 2 = TP4 gang
+-- Decode pool
     +-- D replica 1 = TP8 gang
     +-- D replica 2 = TP8 gang
```

We may require each TP replica to schedule atomically while allowing pool replicas to scale independently.

Grove models this with PodClique, PodCliqueScalingGroup, PodCliqueSet and PodGang. KAI supports hierarchical PodGroups/SubGroups and nested gang thresholds.

---

# 28. Elastic Gang

Some workloads have a minimum and maximum useful size.

Example:

```text
training:
    min 8 GPUs
    max 32 GPUs

Prefill pool:
    min 2 replicas
    max 20 replicas
```

An elastic workload can run at several resource sizes.

Gang semantics become:

```text
schedule at least MIN
possibly grow toward MAX
```

instead of one fixed all-or-nothing count.

---

# 29. Resource Flavor / Accelerator Class

GPU is not one homogeneous resource.

Examples:

```text
H100 80GB
H200
B200
GB200
A100 40GB
A100 80GB
L40S
MIG slice
```

A Resource Flavor captures a schedulable resource class and the nodes that provide it.

A workload might say:

```text
requires H100-class
prefers H200
can fall back to A100 if policy allows
```

---

# 30. Heterogeneous Accelerator Scheduling

A job's performance varies by GPU type, and the ratio is workload-dependent.

Example:

```text
Model A:
    H100 = 3x A100

Model B:
    H100 = 1.4x A100
```

Therefore equal GPU count is not necessarily equal progress or fair value.

Gavel's key abstraction is **effective throughput**: compare allocations by useful progress rather than raw accelerator count.

---

# 31. Multi-Resource Scheduling

A GPU workload also consumes CPU, host memory, NIC bandwidth, storage bandwidth, local NVMe and power.

Synergy showed that treating CPU/memory as fixed quantities proportional to GPU count can materially hurt DNN throughput because workloads have different auxiliary-resource sensitivities.

A mature resource request is a vector, not a GPU count.

---

# 32. Feasibility

A node/domain is feasible if every hard requirement can be satisfied.

Example requirement:

```text
4 H100
400 GB host RAM
RDMA NIC
same NVLink domain
```

A node with four free H100s but only 128 GB RAM is not feasible.

Scheduling normally proceeds:

```text
filter infeasible
then score feasible candidates
```

---

# 33. Hard Constraint vs Soft Preference

Hard constraints filter:

```text
must be H100
must fit model memory
must be in compliance zone
TP ranks must share required domain
```

Soft preferences rank:

```text
prefer same rack
prefer low fragmentation
prefer warm image/model
prefer cheaper zone
```

Never let a soft score override a hard constraint.

---

# 34. Placement Group / Resource Bundles

A placement group is a reservation composed of multiple resource bundles scheduled according to a placement strategy.

Ray exposes bundles with strategies such as:

```text
PACK
STRICT_PACK
SPREAD
STRICT_SPREAD
```

This is a useful application-facing abstraction for gang and multi-actor placement.

---

# 35. Topology

A GPU cluster is not flat.

Useful hierarchy:

```text
region
  |
zone
  |
block / superpod
  |
rack / network-fabric domain
  |
node / NVLink or NVSwitch domain
  |
NUMA
  |
GPU
```

Communication cost changes dramatically across topology boundaries.

---

# 36. Topology-Aware Scheduling

Topology-aware scheduling asks:

> **Which feasible placement minimizes communication cost while respecting capacity and availability?**

The recurring inference pattern is:

```text
within one distributed replica:
    PACK

between independent replicas:
    SPREAD
```

---

# 37. Topology Domain

A topology domain is one value at a hierarchy level, e.g. rack=r17 or gpu-domain=nvlink-23.

A workload should often express:

```text
all ranks inside one rack
```

without hard-coding which rack, so the scheduler retains placement freedom.

---

# 38. PACK vs SPREAD

PACK places related components close together for communication and consolidation.

SPREAD distributes components for availability and failure isolation.

A sophisticated workload can require both at different hierarchy levels:

```text
TP8 ranks:
    PACK within NVLink domain

four TP8 replicas:
    SPREAD across racks
```

---

# 39. Bin Packing

Bin packing places new work into already-used capacity so other nodes/domains remain empty.

Example:

```text
Node A: 6/8 GPUs used
Node B: 2/8 used
new workload: 2 GPUs
```

Choose A -> 8/8 to preserve the six-GPU shape on B.

KAI's topology scheduler currently uses bin-packing at topology-domain selection.

---

# 40. Fragmentation

Fragmentation means:

> **Enough aggregate capacity exists, but not in the shapes/topologies workloads require.**

Example:

```text
8 free GPUs total
2 on node A
2 on B
2 on C
2 on D

TP8 requires one NVLink/NVSwitch domain.
```

Reported free capacity is 8 GPUs; usable capacity for that request is 0.

---

# 41. Internal vs External Fragmentation

External fragmentation: free GPUs exist but are scattered into unusable shapes.

Internal fragmentation: allocated GPUs are underused, e.g. a workload reserves an entire 80GB GPU while using 25GB and little compute.

GPU sharing, MIG or MPS can reduce internal fragmentation but introduce interference/isolation complexity.

---

# 42. Consolidation / Defragmentation

Consolidation means:

> **Move or restart workloads to create cleaner large resource blocks.**

Before:

```text
A: XX..XXXX
B: XXXX..XX
C: XX.XXXXX
```

After moving flexible jobs:

```text
A: XXXXXXXX
B: XXXXXXXX
C: ........
```

Now a large gang can fit.

KAI explicitly includes Consolidate before more disruptive reclaim/preempt actions.

---

# 43. GPU Sharing / Fractional GPU

Sharing mechanisms include memory-based fractional allocation, MIG, MPS, time slicing and application-level space sharing.

Potential benefit: high utilization for small models and development work.

Risks:

```text
memory interference
compute interference
weak isolation
unpredictable latency
```

Whole-GPU isolation may remain preferable for latency-critical production inference.

---

# 44. Colocation / Space Sharing

Colocation places multiple workloads on one accelerator simultaneously.

Throughput is not necessarily additive and interference may be asymmetric.

A scheduler needs profiling, performance models or strong isolation to make intelligent colocation decisions.

Gavel explicitly incorporates space sharing into heterogeneity-aware policy.

---

# 45. Time Sharing

Time sharing multiplexes a GPU over time.

Gandiva exploited repetitive DNN iterations to time-slice and migrate training jobs, improving early progress and resource fit.

For online inference, scheduler-level time slicing is generally less attractive than application batching/partitioning because latency is sensitive; for offline workloads it can be useful.

---

# 46. Workload Introspection

Generic schedulers know resource requests.

ML-aware schedulers can also observe:

```text
iteration time
scaling efficiency
GPU utilization
CPU stalls
remaining work
checkpoint cost
training progress
```

Gandiva uses repetitive iteration behavior for introspection and migration. Pollux observes training behavior to estimate goodput under different allocations.

---

# 47. Throughput

System throughput measures examples/sec, tokens/sec, steps/sec or requests/sec.

But raw throughput does not always equal useful ML progress.

That motivates goodput.

---

# 48. Goodput

Pollux combines:

```text
system throughput
×
statistical efficiency
```

into **goodput**.

A larger batch may increase samples/sec while requiring more samples to reach the same model quality. So more raw throughput can produce less useful progress per second.

This is primarily an offline-training concept.

---

# 49. Scaling Efficiency

Suppose:

```text
1 GPU  = 100 samples/s
2 GPUs = 190
4 GPUs = 330
8 GPUs = 500
```

Speedup is sublinear. Elastic schedulers need a workload-specific resource/performance curve rather than assuming twice the GPUs means twice the progress.

---

# 50. Effective Throughput

Gavel uses effective throughput to normalize progress across heterogeneous accelerator types.

If job J gets twice the throughput on H100 as A100, one H100-second is roughly twice as much progress for J. Another job may have a different ratio.

Accelerator equivalence is workload dependent.

---

# 51. Job Completion Time — JCT

Offline objective:

```text
JCT
=
queue wait
+
runtime
+
preemption/restart overhead
```

Schedulers such as Tiresias target average JCT.

For online services, rollout/scale-up latency is analogous but services do not normally have a finite JCT.

---

# 52. Makespan

Makespan is the time until an entire set of jobs completes.

Useful for batch campaigns, evaluation suites and overall fleet throughput.

---

# 53. Least Attained Service — LAS

LAS prioritizes workloads that have received the least service so far.

Without knowing job duration, new/short jobs initially receive priority while long jobs gradually accumulate service.

Tiresias uses a two-dimensional LAS-inspired policy tailored to distributed DL jobs.

---

# 54. Gittins Index

Gittins-style scheduling minimizes mean response/JCT under uncertain job lengths using partial statistical information.

Tiresias develops a discretized two-dimensional version for distributed DL jobs.

Interview takeaway:

> **Exact job duration is not required to exploit probabilistic information about remaining work.**

---

# 55. Finish-Time Fairness

Traditional fair sharing asks whether current resource shares are fair.

Themis asks:

> **How much later does contention cause my job to finish compared with running alone?**

This creates finish-time fairness, which can differ substantially from instantaneous GPU fairness.

---

# 56. Future Planning

Most schedulers react to current state. Shockwave studies dynamically adapting ML jobs and uses future planning to jointly reason about fairness and efficiency.

Broader lesson:

> **When workload performance curves evolve, purely myopic allocation can repeatedly make locally sensible but globally poor decisions.**

---

# 57. Autoscaling

Autoscaling answers:

> **How much capacity should a workload have?**

Placement answers where that requested capacity can actually fit.

Example:

```text
Autoscaler:
    Decode pool 8 -> 12 replicas

Fleet Scheduler:
    place the four new replica gangs
```

Ray Serve makes this separation concrete: Serve derives desired replicas from request load, while Ray Core places the resource bundles.

---

# 58. Horizontal Scaling

Horizontal scaling changes the number of replicas.

Example:

```text
D pool: 8 -> 16 TP4 replicas
```

This is the dominant scaling mechanism for online serving.

---

# 59. Vertical / Shape Scaling

Vertical scaling changes resources per replica, e.g. TP4 -> TP8 or A100 -> H100.

It often requires restart, re-sharding or a different ServingProfile, so it is more disruptive than horizontal scale-out.

---

# 60. Desired State vs Actual State

Control planes separate:

```text
desired replicas = 20
```

from:

```text
healthy running = 16
pending = 4
```

The LLM router must route to healthy actual capacity, while the autoscaler/fleet controller works to converge actual toward desired.

---

# 61. Control Loop

A control loop repeatedly:

```text
observe
-> compare actual vs desired
-> act
-> wait
-> observe again
```

Example:

```text
request queue grows
-> autoscaler increases replicas
-> fleet scheduler places GPUs
-> replicas load model
-> queue falls
```

Poorly tuned loops oscillate.

---

# 62. Hysteresis / Cooldown

Without hysteresis, small demand fluctuations cause repeated scale-up/scale-down churn.

Use scale thresholds, downscale delay, minimum runtime, EWMA and cooldown windows.

Ray Serve explicitly provides conservative downscale delay/factor controls.

---

# 63. Backpressure

Backpressure prevents unlimited work accumulation when demand exceeds service capacity.

It can exist at the request router, deployment queue, fleet admission queue or tenant quota.

Online: uncontrolled queue growth becomes latency collapse.

Offline: queuing is acceptable but must remain fair and observable.

---

# 64. Scheduler Action

Some schedulers structure a scheduling cycle as ordered actions.

KAI's current default cycle includes:

```text
Allocate
Consolidate
Reclaim
Preempt
StaleGangEviction
```

Volcano exposes actions such as:

```text
enqueue
allocate
backfill
preempt
reclaim
```

This architecture separates:

> **What operation are we trying to perform?**

from:

> **Which policy/plugin ranks or filters candidates for that operation?**

---

# 65. Allocate

Allocate is the non-disruptive path:

```text
pending workload
+
currently free resources
-> bind workload
```

Prefer allocation before disruptive reclaim/preemption when feasible.

---

# 66. Enqueue / In-Queue Admission

Some schedulers distinguish submitted from eligible/in-queue.

Volcano's enqueue stage checks whether a workload meets queue/gang conditions before allocation.

This is useful when admission itself has policy.

---

# 67. Reconciliation

A scheduler decision can fail after it is made: node disappears, pod fails, resource claim fails, GPU becomes unhealthy.

A control plane therefore repeatedly reconciles desired allocation with observed state and repairs drift.

---

# 68. Scheduler Sharding

At very large scale, one global scheduling loop may become too slow or have too large a blast radius.

KAI supports scheduling shards/node pools with independent scheduler instances, queues and quotas.

Tradeoff:

```text
better scalability/isolation
vs
less global optimization
```

---

# 69. Scheduler Throughput vs Scheduling Quality

A theoretically optimal placement taking 30 seconds may be worse than a near-optimal one taking 100 ms.

Scheduling itself has an SLO.

Relevant dimensions:

```text
jobs considered per cycle
nodes evaluated
candidate pruning
incremental caches
parallelism
sharding
```

This mirrors the routing-latency-budget principle from LLM Routing.

---

# 70. Placement Locality vs Failure Isolation

Communication says pack; availability says spread.

Principal-level schedulers reason at multiple hierarchy levels:

```text
TP ranks:
    pack within node/NVLink domain

replicas:
    spread across racks

service copies:
    spread across zones
```

This is not one PACK/SPREAD switch.

---

# 71. Startup Ordering

Gang scheduling answers who must receive resources together.

Startup ordering answers who must become ready first.

Example:

```text
workers start
-> leader initializes distributed runtime
```

Grove explicitly supports startup dependencies for multi-component inference workloads.

---

# 72. Rollout

Updating an online service temporarily requires old and new replicas simultaneously.

If steady-state quota equals normal usage exactly, rolling updates can deadlock.

Useful concepts:

```text
temporary over-quota
surge capacity
reserved rollout capacity
drain old only after new is ready
```

Online fleet scheduling must understand lifecycle, not only steady state.

---

# 73. Drain

Drain means:

> **Stop sending new work to an instance, allow in-flight work to finish, then remove it.**

For inference:

```text
remove replica from router eligibility
wait for in-flight requests
optionally preserve/export execution state
terminate
```

Graceful drain is much cheaper than abrupt preemption.

---

# 74. Warmup Cost

A GPU is not useful the instant a container starts.

Inference startup may include image pull, model download, weight load, CUDA initialization, kernel compilation, graph capture, distributed rendezvous and warmup.

Therefore free GPU does not equal immediately available serving capacity.

Warmup cost should affect preemption, autoscaling and rollout.

---

# 75. Migration Cost

Moving work may require checkpointing, network copy, restart, model reload, KV loss or distributed rendezvous.

Consolidation and preemption should compare the placement benefit against migration cost.

---

# 76. Checkpointing

Checkpointing converts running state into recoverable state.

Offline training typically checkpoints model weights, optimizer state, RNG state and data-loader progress.

Preemption becomes much cheaper when jobs checkpoint frequently.

This will be central to the offline scheduler.

---

# 77. Cost of Preemption

Victim selection should consider more than priority:

```text
priority
quota status
preemptibility
time since start
minimum runtime
checkpoint age
restart/warmup cost
remaining work
state loss
topology scarcity
```

---

# 78. Scarcity-Aware Resource Value

Eight H100s in one NVLink island can be more valuable than eight scattered H100s.

A rare GB200 rack-scale domain should not be consumed by a job that performs equally well on common resources.

The scheduler should reason about opportunity cost of consuming scarce resource shapes.

---

# 79. Resource Efficiency vs Utilization

High utilization is not automatically good.

A GPU can be 99% busy while the workload is on a poor accelerator type, network-bound, statistically inefficient or missing its SLO.

Better objective:

```text
useful work per dollar / scarce resource / unit time
```

Gavel, Pollux and Synergy each broaden efficiency beyond GPU busy percentage.

---

# 80. Online vs Offline Objectives

## Online

Primary goals:

```text
availability
latency SLO
capacity headroom
fast scale-up
low disruption
cost
```

Workloads are long-lived, continuously serving, and often hard to preempt.

## Offline

Primary goals:

```text
fairness
queue wait
JCT
makespan
goodput
utilization
cost
```

Workloads are finite, often checkpointable, often preemptible, and sometimes elastic.

A unified fleet scheduler can share mechanisms while applying different policies.

---

# 81. Reference Systems — What to Remember

## KAI Scheduler

Remember:

```text
hierarchical queues
deserved quota
over-quota fair share
reclaim vs preemption
gang / hierarchical PodGroups
topology-aware placement
bin packing
consolidation
GPU sharing
scheduler sharding
```

Especially useful vocabulary:

```text
Allocate
Consolidate
Reclaim
Preempt
```

## Grove

Remember:

```text
inference workload composition
PodClique
PodCliqueScalingGroup
PodCliqueSet
PodGang
hierarchical gangs
multi-level autoscaling
topology constraints
startup ordering
rollout semantics
```

Grove describes workload structure; KAI performs underlying allocation/placement.

## Kueue

Remember:

```text
admission != pod placement
LocalQueue
ClusterQueue
Cohort
nominal quota
borrowing/lending
ResourceFlavor
fair sharing
Topology-Aware Scheduling
AdmissionCheck
```

Kueue gives a particularly clean quota/admission abstraction.

## Volcano

Remember:

```text
gang scheduling
enqueue
allocate
backfill
preempt
reclaim
DRF
proportion
binpack
topology-aware placement
plugin/action framework
```

Volcano becomes especially relevant for offline distributed training.

## Ray Serve / Ray Core

Remember:

```text
serving demand -> desired replicas
target ongoing requests
max ongoing requests
backpressure
placement groups
resource bundles
PACK / STRICT_PACK / SPREAD
accelerator types
graceful shutdown
application-level elasticity
```

Important separation:

```text
Serve autoscaler decides HOW MANY replicas.
Ray Core placement decides WHERE resources go.
```

---

# 82. Research Lineage — What Each Paper Adds

## Gandiva — OSDI 2018

Key insight: DNN iterations are repetitive enough to introspect, time-share and migrate jobs.

Concepts:

```text
time sharing
migration
job introspection
early feedback for hyperparameter searches
dynamic resource fit
```

## Tiresias — NSDI 2019

Key problem: unknown training duration plus distributed all-or-nothing jobs.

Concepts:

```text
JCT optimization
2D LAS
Gittins-style scheduling under uncertainty
consolidated placement
```

## Themis — NSDI 2020

Key insight: instantaneous GPU fairness is not necessarily finish-time fairness.

Concepts:

```text
finish-time fairness
placement-sensitive jobs
auction/two-level allocation
efficiency vs fairness
```

## Gavel — OSDI 2020

Key insight: hardware heterogeneity belongs inside the scheduling policy.

Concept:

```text
effective throughput
```

It generalizes fairness, FIFO, makespan, cost/SLO and finish-time-fairness objectives to heterogeneous accelerators.

## Pollux — OSDI 2021

Key insight: scheduler allocation and the training job's own configuration should adapt together.

Concept:

```text
goodput = system throughput × statistical efficiency
```

Also: elastic allocation, co-adaptation, dynamic performance models.

## Synergy — OSDI 2022

Key insight: GPUs are not the only resource determining DNN throughput.

Concepts:

```text
CPU sensitivity
memory sensitivity
multi-resource allocation
optimistic profiling
```

## Shockwave — NSDI 2023

Key insight: dynamically adapting ML jobs need future-aware fairness and efficiency.

Concepts:

```text
future planning
dynamic performance
market/fairness reasoning
stochastic adaptation
```

---

# 83. Concepts We Will Use Heavily in the Online Design

Know these deeply before component design:

```text
Queue
Hierarchical Queue

Quota / Guaranteed Capacity
Limit / Maximum
In-Quota / Over-Quota
Borrowing
Lending
Fair Share
DRF

Admission
Priority
Preemptibility
Preemption
Reclaim

Gang Scheduling
Hierarchical Gang
Elastic Gang

Resource Flavor
Feasibility
Hard vs Soft constraints

Topology
PACK / SPREAD
Bin Packing
Fragmentation
Consolidation

Reservation
Backfill

Autoscaling
Desired vs Actual capacity
Control loops
Hysteresis
Backpressure

Rollout
Drain
Warmup cost

Scheduler Actions
Scheduler Sharding
```

---

# 84. Concepts We Will Emphasize Later for Offline / Training

Know the vocabulary now; deep dive later:

```text
JCT
Makespan
LAS
Gittins Index
Finish-Time Fairness
Effective Throughput
Goodput
Statistical Efficiency
Scaling Efficiency
Elastic Training
Time Sharing
Checkpointing
Migration
Historical Fairness
Multi-resource sensitivity
Future Planning
```

---

# 85. The Most Important Separations to Memorize

```text
QUEUE
    organizational resource-governance boundary

QUOTA
    what capacity am I entitled to?

FAIR SHARE
    how is surplus divided?

BORROWING
    temporarily use another queue's unused entitlement

RECLAIM
    recover borrowed capacity across queues

PRIORITY
    which workload is more important?

PREEMPTION
    interrupt lower-value running work

GANG
    what resources must become available together?

TOPOLOGY
    where should those resources physically be placed?

BIN PACKING
    how do I preserve useful future resource shapes?

CONSOLIDATION
    can I move existing work to repair fragmentation?

AUTOSCALING
    how much capacity should the workload request?

FLEET SCHEDULING
    can it have that capacity, and where?

REQUEST ROUTING
    which already-running replica serves this request?
```

---

# 86. Principal-Level Mental Model

A GPU fleet scheduler is not one algorithm.

It is several policy layers composed together:

```text
ORGANIZATIONAL POLICY
    queues / quotas / fair share
             |
             v
WORKLOAD POLICY
    priority / preemptibility / admission
             |
             v
ATOMICITY
    gang / elastic gang
             |
             v
FEASIBILITY
    GPU type / memory / CPU / NIC / topology
             |
             v
PLACEMENT OPTIMIZATION
    locality / fragmentation / cost / availability
             |
             v
DISRUPTION POLICY
    reclaim / preempt / consolidate / reserve
             |
             v
CONTROL LOOP
    autoscale / rollout / drain / reconcile
```

The Principal Staff question is rarely:

> Which scheduling algorithm do you use?

The stronger answer is:

> **Which policies are hard guarantees, which are optimization objectives, at what organizational scope do they apply, how do they interact under overload, and how do you keep those control loops stable?**

That is the vocabulary foundation for the architecture we design next.

---

# Primary Study Sources

The concepts above were synthesized from the current designs and documentation of:

- NVIDIA KAI Scheduler and Grove.
- Kubernetes Kueue.
- Volcano Scheduler.
- Ray Core / Ray Serve.
- Gandiva (OSDI 2018).
- Tiresias (NSDI 2019).
- Themis (NSDI 2020).
- Gavel (OSDI 2020).
- Pollux (OSDI 2021).
- Synergy (OSDI 2022).
- Shockwave (NSDI 2023).

---

# GPU Interconnect & Network Topology Primer

These terms describe **different communication layers** in a GPU cluster. They should not be treated as interchangeable.

A useful mental hierarchy is:

```text
same GPU
   |
   v
NVLink
   |
   v
NVSwitch fabric
   |
   v
PCIe / host boundary
   |
   v
GPU NIC
   |
   +--> InfiniBand
   |
   +--> RoCE / Ethernet
   |
   v
other server / rack / fabric domain
```

The exact hardware topology varies by GPU generation and system design, but this hierarchy is a useful scheduler mental model.

## NVLink

**NVLink is NVIDIA's high-bandwidth, low-latency point-to-point interconnect technology for connecting accelerators and, in some systems, CPUs and GPUs.**

For fleet scheduling, the important use is:

```text
GPU <---- NVLink ----> GPU
```

instead of forcing GPU-to-GPU traffic through a slower host-oriented path.

Why it matters: distributed model parallelism constantly exchanges tensors.

Examples:

```text
Tensor Parallelism:
    all-reduce / all-gather every layer

Expert Parallelism:
    token dispatch between experts

Pipeline Parallelism:
    activation transfer between stages
```

For communication-heavy TP, placing ranks on GPUs connected by NVLink can materially outperform placement across ordinary PCIe or network boundaries.

Scheduler implication:

```text
TP ranks should often be placed
inside the same high-bandwidth NVLink domain.
```

Important distinction:

> **NVLink is a link/interconnect technology, not a cluster-wide packet network.**

## NVSwitch

NVLink gives high-speed links between devices. As GPU count grows, directly wiring every GPU to every other GPU becomes difficult.

**NVSwitch is NVIDIA's switching fabric for connecting many NVLink endpoints.**

Mental model:

```text
GPU0 ---\
GPU1 ----\
GPU2 ----- NVSwitch Fabric ----- GPU4
GPU3 ----/                     -- GPU5
                               -- GPU6
                               -- GPU7
```

Instead of reasoning only about individual physical links, GPUs connected through the same NVSwitch fabric can participate in a high-bandwidth GPU communication domain.

Classic server example:

```text
8 GPUs
     |
NVSwitch fabric
     |
near-uniform high-bandwidth GPU-to-GPU connectivity
```

Newer systems can extend NVLink/NVSwitch fabrics beyond one conventional server chassis, so a scheduler should model the actual **NVLink/NVSwitch domain** supplied by the platform rather than assuming `NVSwitch == exactly one host`.

The conceptual difference is:

```text
NVLink
    = high-speed connection technology

NVSwitch
    = switching fabric connecting
      many NVLink endpoints
```

A rough analogy is:

```text
Ethernet link   <-> NVLink
Ethernet switch <-> NVSwitch
```

The analogy is imperfect, but useful.

## RDMA

To understand InfiniBand and RoCE, first understand **RDMA**.

RDMA means:

> **Remote Direct Memory Access**

It allows one machine to transfer data directly to/from registered memory associated with another machine with minimal remote CPU involvement.

A traditional network path is roughly:

```text
NIC
 -> kernel / CPU
 -> copy
 -> application memory
```

An RDMA path can be closer to:

```text
NIC
 -> registered application memory
```

For GPU clusters, GPUDirect RDMA can make the path:

```text
GPU memory
   |
NIC
   |
network
   |
NIC
   |
GPU memory
```

without staging the bulk data through host CPU memory.

Important separation:

```text
RDMA
    = memory-access / transport capability

InfiniBand
    = network architecture that natively supports RDMA

RoCE
    = RDMA transported over Ethernet
```

## InfiniBand — IB

**InfiniBand is a dedicated high-performance switched network architecture widely used in HPC and large AI clusters.**

It is designed for:

```text
very high bandwidth
low latency
RDMA
large-scale collective communication
```

A cluster path may look like:

```text
GPU Server A
   |
InfiniBand HCA
   |
InfiniBand Switch
   |
InfiniBand Fabric
   |
InfiniBand Switch
   |
InfiniBand HCA
   |
GPU Server B
```

`HCA` means **Host Channel Adapter** — the InfiniBand network adapter.

InfiniBand is not NVIDIA-specific, although NVIDIA/Mellanox is a major vendor in this ecosystem.

### Why AI clusters use it

Distributed training and large-model inference frequently perform collectives such as:

```text
all-reduce
all-gather
reduce-scatter
all-to-all
```

If communication is slow:

```text
GPU computes
   -> waits for collective
GPU computes
   -> waits again
```

The expensive GPUs become communication-stalled.

Scheduler implication: not all racks or nodes necessarily have equivalent InfiniBand connectivity. The scheduler may need to know:

```text
same IB fabric
same leaf switch
same rail
same network block
cross-fabric path
```

and pack communication-heavy workers accordingly.

## RoCE

**RoCE means RDMA over Converged Ethernet.**

It provides RDMA semantics over an Ethernet network instead of requiring a native InfiniBand fabric.

Mental model:

```text
GPU Server A
   |
RoCE-capable NIC
   |
Ethernet Switch Fabric
   |
RoCE-capable NIC
   |
GPU Server B
```

RoCEv2 is commonly used in modern data-center fabrics because it transports RDMA traffic over routable IP/UDP-based Ethernet infrastructure.

The attraction is:

```text
use Ethernet ecosystem
+
obtain RDMA-like low-overhead data movement
```

rather than operating a separate InfiniBand network.

### Why RoCE requires careful network engineering

RDMA traffic is sensitive to:

```text
packet loss
congestion
incast
queue buildup
```

Large RoCE deployments therefore often use carefully engineered congestion-control and traffic-management mechanisms such as:

```text
ECN
PFC in some deployments
QoS
traffic-class separation
buffer tuning
```

The details vary by deployment.

The scheduling lesson is:

> **Two nodes with `400 Gb Ethernet` are not automatically equivalent to two nodes connected through a well-engineered low-latency RoCE fabric.**

Fabric class and topology should be explicit scheduler metadata.

## InfiniBand vs RoCE

Both can support high-performance RDMA communication.

High-level distinction:

```text
InfiniBand
==========
Dedicated HPC/AI network architecture
InfiniBand switches + HCAs
RDMA is native to the fabric

RoCE
====
RDMA transported over Ethernet
Ethernet switching/IP ecosystem
requires careful congestion engineering
```

Do **not** memorize:

```text
IB is always faster than RoCE
```

as a universal rule.

Actual performance depends on:

```text
NIC generation
link speed
switch design
oversubscription
routing
congestion
collective library
topology
deployment tuning
```

For the scheduler, the useful abstraction is closer to:

```text
FabricProfile {
    fabric_type = INFINIBAND | ROCE | ETHERNET
    bandwidth_class
    latency_class
    topology_domain
    oversubscription_class
    rdma_capable
}
```

rather than hard-coding one technology as universally superior.

## GPUDirect RDMA

Ordinary RDMA may move data directly between NICs and registered host memory.

**GPUDirect RDMA** allows compatible NICs and GPUs to move data directly between GPU memory and the network path.

Conceptually:

```text
Without GPUDirect:

GPU
 -> CPU/host memory
 -> NIC
 -> network
 -> NIC
 -> CPU/host memory
 -> GPU


With GPUDirect RDMA:

GPU
 -> NIC
 -> network
 -> NIC
 -> GPU
```

This reduces:

```text
host-memory copies
CPU involvement
latency
```

and can increase effective transfer efficiency for tensor/KV movement.

This is directly related to the NIXL/KV-transfer path studied in the LLM-routing design.

## NIC

`NIC` means **Network Interface Card / Controller**.

For AI servers, relevant NIC characteristics include:

```text
link speed
RDMA support
InfiniBand vs RoCE
number of NICs
PCIe placement
NUMA locality
GPU affinity
fabric/rail membership
```

A node with eight GPUs and one congested NIC can behave very differently from a node with several high-bandwidth NICs carefully mapped to GPU groups.

## Multi-Rail Networking

Large AI systems may attach a server to multiple independent network paths or **rails**.

Example:

```text
GPU0-3 -> NIC0 -> Fabric Rail A
GPU4-7 -> NIC1 -> Fabric Rail B
```

This can improve:

```text
aggregate bandwidth
fault tolerance
parallel communication capacity
```

The scheduler/runtime may need to preserve GPU-to-NIC affinity so ranks use the correct local NIC/rail.

## NUMA

NUMA means **Non-Uniform Memory Access**.

A large CPU server may contain multiple CPU sockets. Resources attached near one socket are cheaper to access from that socket than from another.

Example:

```text
CPU Socket 0
  |
  +-- GPU0-3
  +-- NIC0

CPU Socket 1
  |
  +-- GPU4-7
  +-- NIC1
```

If a process controlling GPU0 uses CPU cores and a NIC attached to Socket 1, data may cross the CPU inter-socket fabric.

That can increase latency and reduce throughput.

Scheduler implication:

```text
GPU + CPU + NIC
```

sometimes need **joint NUMA-aware placement**.

## PCIe

PCI Express is the general-purpose high-speed I/O interconnect used to attach devices such as:

```text
GPUs
NICs
NVMe drives
```

to CPUs and other PCIe devices.

Even when GPUs use NVLink for GPU-to-GPU collectives, PCIe topology can matter for:

```text
GPU <-> NIC
GPU <-> CPU
GPU <-> NVMe
GPUDirect Storage
GPUDirect RDMA
```

Two devices that appear on the same server may have very different communication paths depending on their PCIe root complex / switch topology.

## Putting the Hierarchy Together

For a scheduler, an approximate communication hierarchy might be:

```text
FASTEST / TIGHTEST

same GPU

same NVLink/NVSwitch domain

same host through PCIe / local fabric

same rack / same low-level IB or RoCE switch domain

same larger network fabric / block

cross-block

cross-zone

SLOWER / MORE FAILURE BOUNDARIES
```

Do not treat this as a universal latency table. The exact performance comes from the hardware and fabric profile.

The scheduler uses the hierarchy primarily to reason about:

```text
communication locality
bandwidth scarcity
failure domains
fragmentation
```

## Example — TP8 Placement

Suppose a model uses:

```text
TP = 8
```

and performs frequent tensor-parallel collectives.

Candidate A:

```text
8 GPUs
same NVSwitch domain
```

Candidate B:

```text
4 GPUs on Node 1
+
4 GPUs on Node 2
connected by RoCE
```

Both satisfy:

```text
8 x H100
```

but they are not equivalent.

Candidate A is generally the much stronger placement for communication-heavy TP.

The workload should therefore express something like:

```text
TP8:
    require/prefer same NVSwitch domain
```

instead of merely:

```text
request 8 GPUs
```

## Example — Data Parallel Training

Suppose:

```text
DP = 64
```

The topology requirement may be hierarchical:

```text
each local TP8 group
    packed inside one NVSwitch domain

8 TP groups
    distributed across nodes

all nodes
    preferably inside the same high-bandwidth IB/RoCE fabric
```

So different parallelism dimensions can require different topology constraints.

## Example — P/D Disaggregated Inference

Suppose:

```text
Prefill P3
Decode D7
```

must transfer many GB of KV state.

If:

```text
P3 and D7
    same high-bandwidth RoCE/IB domain
```

transfer may be relatively cheap.

If they are:

```text
across congested fabric blocks
```

transfer may eliminate the benefit of P/D disaggregation.

This connects GPU Fleet Scheduling directly to the LLM Router:

```text
Fleet Scheduler
    determines physical P/D topology

Router
    later observes those topology relationships
    when evaluating P->D execution paths
```

## Topology Vocabulary to Memorize

```text
NVLink
    High-bandwidth NVIDIA device interconnect.

NVSwitch
    Switching fabric connecting many NVLink endpoints.

NIC
    Network interface connecting a server to a fabric.

RDMA
    Direct remote-memory data movement with low CPU involvement.

GPUDirect RDMA
    RDMA directly between compatible NICs and GPU memory.

InfiniBand / IB
    Dedicated high-performance RDMA network architecture.

RoCE
    RDMA over Ethernet.

PCIe
    General-purpose local device interconnect.

NUMA
    CPU/memory/device locality inside a multi-socket host.

Rail
    One independent high-bandwidth network path/fabric.

Topology Domain
    A set of resources sharing a locality level:
    NVLink domain, node, rack, fabric, zone, etc.
```

The scheduler should convert these into:

```text
hard placement constraints
soft locality preferences
bandwidth/latency classes
failure-domain boundaries
scarcity information
```

rather than treating them merely as hardware names.


---

# Topology Cost, GPU Identity, and Runtime Placement

## Why We Do Not Rank 8,000 GPUs From 1 to 8,000

Suppose the cluster has:

```text
1,000 nodes
8 GPUs per node
8,000 GPUs total
```

The scheduler should not assign each GPU a permanent global preference rank such as:

```text
GPU-1
GPU-2
...
GPU-8000
```

and then say:

```text
GPU-17 prefers GPU-18, then GPU-19, ...
```

That loses the structure that actually matters.

A better model gives every physical GPU a **stable resource identity** plus independent topology membership.

Conceptually:

```text
GPUResourceID {
    cluster_id
    node_id
    device_uuid
}
```

Human-readable form:

```text
cluster-a/node-0042/gpu-3
```

Canonical form may use a device UUID:

```text
cluster-a/node-0042/GPU-<uuid>
```

The local ordinal `gpu-0 ... gpu-7` is useful for humans and local topology, but the canonical identity should not depend only on a global sequential number.

Each GPU also belongs to topology domains:

```text
GPU = cluster-a/node-0042/gpu-3

zone            = zone-a
fabric-block    = block-7
rack            = rack-12
node            = node-0042
nvlink-domain   = node-0042/nvswitch-domain-0
numa-domain     = numa-1
nic-affinity    = nic-1 / rail-b
```

Identity answers:

> Which physical resource is this?

Topology metadata answers:

> Where is it relative to other resources?

These are separate concepts.

---

## A and B in `CommunicationCost(A, B)`

`A` and `B` are not always individual GPUs.

They are **communication endpoints at the level relevant to the scheduling decision**.

### Case 1 — Individual GPU-to-GPU path

```text
A = node-42/gpu-0
B = node-42/gpu-5
```

Useful when examining NVLink connectivity, PCIe locality, or GPU-to-NIC affinity.

### Case 2 — Two distributed model replicas

For disaggregated inference:

```text
A = Prefill replica P3
    = {node-10/gpu0..gpu3}

B = Decode replica D7
    = {node-20/gpu0..gpu7}
```

Now `CommunicationCost(A,B)` means the estimated cost of the required P->D communication pattern, not one arbitrary pair of GPUs.

For KV transfer it may depend on:

```text
which TP shard on P maps to which TP shard on D
bytes per shard
number of concurrent flows
NIC/rail affinity
network path
```

### Case 3 — Parallelism groups

For distributed training:

```text
A = TP group 0
B = TP group 1
```

or:

```text
A = one DP replica
B = another DP replica
```

Again, the cost is group-to-group communication.

### Case 4 — Topology domains

During an early pruning step:

```text
A = rack-12
B = rack-14
```

The scheduler may only need a coarse communication class such as same-rack, same-fabric-block, or cross-block before choosing exact nodes/GPUs.

So a better abstraction is:

```text
CommunicationCost(
    source_endpoint_or_group,
    destination_endpoint_or_group,
    communication_pattern,
    current_network_state
)
```

---

## Communication Cost Is Not Purely Static

Topology gives a **static or slow-changing base cost**, but actual communication cost has three layers.

### 1. Static / Slow-Changing Physical Cost

Derived from:

```text
same NVLink/NVSwitch domain?
same PCIe root / host?
same rack?
same IB/RoCE fabric?
cross-block?
cross-zone?

link bandwidth class
nominal latency class
oversubscription design
GPU-NIC affinity
```

This information can be precomputed/indexed.

Example topology classes:

```text
same NVLink/NVSwitch domain:
    base class = L0

same host but weaker local path:
    base class = L1

same rack / same fabric:
    base class = L2

cross-rack same block:
    base class = L3

cross-block:
    base class = L4
```

These are topology classes, not exact runtime latency predictions.

### 2. Dynamic Network State

Changes over time:

```text
link utilization
fabric congestion
failed/degraded rail
ECN congestion
NIC contention
collective traffic from other jobs
temporary routing changes
```

So two physically identical paths can have different current effective bandwidth.

### 3. Workload-Specific Communication Pattern

A 1 KB control message and a 20 GB KV transfer do not care about topology in the same way.

Likewise:

```text
point-to-point transfer
all-reduce
all-gather
reduce-scatter
all-to-all
```

have different communication behavior.

A useful model is:

```text
CommunicationCost(A, B, W, t)
=
StaticTopologyCost(A, B)
+
DynamicCongestionAdjustment(A, B, t)
+
WorkloadCommunicationModel(W, placement)
```

Or approximately:

```text
T_comm
≈
setup_latency(topology, pattern)
+
bytes / effective_bandwidth(topology, congestion, pattern)
+
collective_overhead(pattern, participants)
```

Production schedulers usually use a coarse model rather than simulating the network exactly.

---

## Example — 1,000 Nodes × 8 GPUs

Suppose:

```text
1,000 nodes
8 H100 GPUs/node
8,000 GPUs total

Every node:
    one 8-GPU NVSwitch domain

Racks:
    20 nodes/rack

Fabric blocks:
    10 racks/block
```

The scheduler can maintain roughly:

```text
8,000 GPU resource identities
1,000 NVSwitch domains
50 racks
5 fabric blocks
```

It does **not** need an 8,000 × 8,000 pairwise preference matrix.

### Workload A — TP8 replica

Request:

```text
8 × H100
TP = 8
required:
    same NVSwitch domain
```

Runtime scheduling:

```text
1. Find NVSwitch domains of flavor H100.
2. Filter domains with >= 8 currently allocatable GPUs.
3. Filter unhealthy/draining/reserved domains.
4. Score remaining domains for:
       fragmentation
       warm model state
       rack/failure-domain policy
       current pressure
5. Choose one domain.
6. Atomically reserve its 8 GPUs.
```

The scheduler never needs to ask:

```text
"For GPU #3271, what are its seven favorite peers?"
```

It asks:

```text
"Which 8-GPU NVSwitch domains currently satisfy this workload?"
```

### Workload B — 4 TP8 replicas

Request:

```text
4 replicas
each TP8

within each replica:
    PACK into one NVSwitch domain

between replicas:
    SPREAD across racks
```

Runtime scheduling:

```text
1. Find candidate 8-GPU NVSwitch domains.
2. Group them by rack.
3. Choose four domains satisfying rack-spread policy.
4. Prefer placements that preserve large clean capacity elsewhere.
```

The preferred peers for one GPU are therefore **workload-dependent**.

For TP8:

```text
same NVSwitch domain is extremely important.
```

For four independent replicas:

```text
different racks may be preferred.
```

A single permanent preference list cannot express both.

---

## Precomputed Topology Index vs Runtime Calculation

Real schedulers usually combine both.

### Precomputed / Maintained Continuously

Keep indexes such as:

```text
FlavorIndex:
    H100 -> nodes/domains

TopologyDomainIndex:
    NVSwitchDomain -> GPUs
    Rack -> Nodes
    FabricBlock -> Racks

CapacityByDomain:
    domain -> currently allocatable resource vector

HealthIndex:
    domain/node/GPU -> health

Optional LocalConnectivityMatrix:
    per node, GPU-to-GPU / GPU-to-NIC locality
```

These indexes are updated incrementally as nodes join/leave, allocations change, health changes, or maintenance begins.

For an 8-GPU node, maintaining a small local connectivity matrix is cheap and useful when internal topology is non-uniform.

### Computed at Scheduling Time

Given a workload:

```text
TP8 H100
same NVSwitch required
spread replicas across racks
```

runtime evaluation combines:

```text
eligible topology domains
+
current free capacity
+
workload-specific hard constraints
+
soft placement scores
```

The scheduler should not scan and rank all 8,000 GPUs individually unless the candidate set has already been narrowed dramatically.

---

## Why a Permanent GPU Preference List Is Usually Wrong

Suppose GPU A has peers B,C,D with excellent local connectivity.

A static list might say:

```text
B > C > D > ...
```

But compare workloads.

### Workload 1 — TP4

```text
A,B,C,D together
```

may be ideal.

### Workload 2 — independent inference replica

If B,C,D already host replicas in the same failure domain, availability policy may prefer another rack.

### Workload 3 — P/D handoff

Decode placement may prefer a GPU group whose NIC/rail is closest to the chosen Prefill group.

### Workload 4 — fragmented cluster

Using B,C,D may destroy the only clean TP8 domain, while a slightly worse placement preserves future capacity.

Thus:

> **Static topology defines locality classes; workload-time scheduling decides how much each locality class matters.**

---

## Staff+ Abstraction — Workload Graph -> Physical Resource Graph

For sophisticated scheduling, think of two graphs.

### Workload Communication Graph

Vertices:

```text
TP ranks
PP stages
DP replicas
Prefill/Decode groups
```

Edges carry communication demand:

```text
expected bytes
collective type
frequency
latency sensitivity
```

### Physical Resource Graph

Vertices:

```text
GPUs
NICs
nodes
switch/fabric domains
```

Edges encode:

```text
bandwidth class
latency class
failure domain
dynamic congestion
```

Scheduling becomes approximately:

> **Embed the workload communication graph onto the physical resource graph while satisfying resource constraints and minimizing communication + fragmentation + disruption cost.**

Production schedulers simplify this aggressively using hierarchical topology domains and candidate pruning rather than solving a full graph-embedding optimization for every workload.

---

## Practical Rule

```text
STATIC / PRE-INDEXED
====================
Who is connected to whom?
At what topology level?
What nominal bandwidth/latency class?
Which GPU is near which NIC?
Which failure domain?

DYNAMIC / CONTINUOUSLY UPDATED
==============================
What is free?
What is reserved?
What is healthy?
What is congested?
What is draining?

WORKLOAD-TIME COMPUTED
======================
Which topology level is required?
Which level is merely preferred?
How many resources must be co-located?
What communication pattern matters?
How should locality trade off against
fragmentation, availability and cost?
```

That is more scalable and expressive than a permanent per-GPU preference ranking.
