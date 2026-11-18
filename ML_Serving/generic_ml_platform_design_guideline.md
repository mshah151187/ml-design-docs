# Generic ML Platform Design Guideline
## State Machines, Domain Controllers, Durable State, and Platform Composition

## 1. Why This Abstraction Matters

Large ML platforms often look very different on the surface:

```text
GPU Scheduling
Model Optimization
Model Serving
Training
Evaluation
Feature Platforms
```

But many of them share the same control-plane structure:

```text
Intent
  |
  v
Durable State
  |
  v
Domain Controller
  |
  +--> observes current state
  +--> evaluates policy
  +--> issues idempotent actions
  |
  v
Workers / External Systems
  |
  v
Observed State
```

The important design insight is:

> **An ML platform is usually not one giant service or one giant controller. It is a composition of multiple domain controllers, each owning one coherent durable state machine, coordinated through explicit contracts.**

This gives us a reusable way to design Serving, Optimization, Training, and Scheduling systems.

---

# 2. State Machine

## 2.1 Definition

A **state machine** describes the valid lifecycle of a durable domain object.

It answers:

```text
What state is this object currently in?

What transitions are legal?

What event/condition causes each transition?

What side effects are required?

What happens if execution fails halfway?
```

Example:

```text
REQUESTED
   |
   v
ALLOCATING
   |
   v
STARTING
   |
   v
READY
   |
   v
SERVING
   |
   v
DRAINING
   |
   v
RELEASED
```

This is not merely a visualization. It defines the lifecycle semantics of the object.

---

## 2.2 State Machine vs Workflow Script

A workflow script says:

```text
do A
then B
then C
```

A state machine says:

```text
The durable object is currently in state B.

If the process crashes now,
another controller instance can resume from B.
```

That is a major production difference.

Bad:

```text
function deploy():
    allocate_gpu()
    start_container()
    load_model()
    publish_endpoint()
```

If the process dies after `load_model()`, the system may not know:

```text
Was the GPU allocated?
Did the container start?
Was the model loaded?
Was traffic published?
Should we retry?
```

State-machine design instead records durable progress:

```text
ALLOCATED
STARTING
MODEL_LOADING
READY
PUBLISHED
```

and the controller continuously reconciles from that state.

---

## 2.3 State Machines Represent Domain Semantics

States should represent meaningful domain facts.

Good:

```text
MODEL_LOADING
READY
DRAINING
```

Weak:

```text
STEP_3
STEP_4
STEP_5
```

The first form tells operators and other controllers what is true. The second merely exposes implementation ordering.

---

## 2.4 State Transition Invariants

Every state transition should answer:

```text
Precondition
    What must already be true?

Action
    What side effect must happen?

Success condition
    What observation proves completion?

Failure behavior
    Can we retry? Roll back? Mark failed?

Durability
    What state must be written before/after side effects?
```

Example:

```text
ALLOCATING -> ALLOCATED

Precondition:
    valid ResourceIntent exists

Action:
    ask GPU Scheduler for resources

Success:
    ResourceAllocation exists

Failure:
    retry / timeout / fail deployment

Durable evidence:
    allocation_id
```

---

# 3. Durable Storage

## 3.1 Definition

**Durable Storage** is the persistent source of control-plane truth that survives:

```text
process restart
controller failover
machine failure
network interruption
leader election
rolling deployment
```

A controller should never depend only on in-memory state for important lifecycle decisions.

Typical durable objects include:

```text
ServingDeployment
ServingInstance
OptimizationJob
OptimizationExperiment
ResourceIntent
ResourceAllocation
TrainingJob
Checkpoint
Rollout
```

---

## 3.2 Desired State and Observed State

A useful control-plane model is:

```text
Desired State
    what should exist

Observed State
    what currently exists
```

Example:

```text
ServingDeployment

desired:
    replicas = 20
    model_bundle = v42
    profile = P0

observed:
    allocated = 20
    runtime_started = 19
    ready = 18
    serving = 17
```

The controller's job is to reduce the difference:

```text
Desired - Observed
```

through safe, idempotent actions.

---

## 3.3 Durable State Is Not Necessarily One Database

"Durable Storage" is a logical ownership concept.

It might physically be:

```text
PostgreSQL
Kubernetes API objects / etcd
distributed KV
workflow database
metadata service
object store
```

The important question is:

> **Which store is authoritative for this domain fact?**

For example:

```text
ServingDeploymentStore
    authoritative for deployment lifecycle

GPU Scheduler AllocationStore
    authoritative for resource ownership

Model Registry
    authoritative for bundle aliases/version discovery
```

Avoid two systems both believing they are authoritative for the same fact.

---

## 3.4 Durable State Enables Recovery

Suppose:

```text
Controller A
    claims Deployment D
    starts runtime
    crashes
```

A new controller can read:

```text
Deployment D state = STARTING
Runtime workload exists
Resource allocation exists
```

and continue reconciliation.

Without durable state, controller failover becomes guesswork.

---

# 4. Domain Controller

## 4.1 Definition

A **Domain Controller** owns the lifecycle of one coherent domain object or state machine.

It continuously asks:

```text
What is desired?

What is observed?

Are they converged?

If not, what is the smallest safe action I can take?
```

Conceptually:

```text
while true:
    object = read_durable_state()

    observed = inspect_external_state(object)

    action = reconcile(object.desired, observed)

    if action:
        execute_idempotent(action)

    update_status()
```

The implementation may be event-driven rather than a polling loop, but the semantics are the same.

---

## 4.2 Controller Is Not the Worker

A controller decides:

```text
WHAT should happen next?
```

A worker/executor performs:

```text
the expensive or operational side effect
```

Example:

```text
OptimizationController
    decides:
        candidate C17 needs benchmark

BenchmarkWorker
    executes:
        load model
        generate traffic
        collect TTFT/TPOT
```

Similarly:

```text
ServingDeploymentController
    decides:
        instance I3 must start

Runtime system
    executes:
        container launch
        model load
        runtime initialization
```

---

## 4.3 Controller Is Not the Analyzer

An analyzer answers:

```text
What is happening?
```

A controller answers:

```text
What lifecycle transition should happen because of it?
```

Example:

```text
Traffic/SLO Analyzer:
    prompt distribution shifted
    TTFT headroom = 3%
    current cost/token increased 20%

Reoptimization Policy:
    current profile qualification is invalid

ServingOptimizationController:
    create OptimizationJob
```

Do not put statistical aggregation, telemetry query logic, and lifecycle reconciliation into one giant controller.

---

# 5. Observer / Analyzer / Policy / Controller / Worker

This is a useful generic decomposition.

```text
Observer / Analyzer
    derives facts

Policy
    interprets those facts

Domain Controller
    owns durable lifecycle transition

Worker / Executor
    performs side effects
```

Example:

```text
Metrics Platform
      |
      v
Traffic/SLO Analyzer
      |
      | observed facts
      v
Reoptimization Policy
      |
      | decision
      v
ServingOptimizationController
      |
      | create OptimizationJob
      v
OptimizationController
      |
      | launch experiment
      v
Benchmark / Quantization / Compilation Workers
```

This separation keeps each component understandable.

---

# 6. When Does Something Deserve Its Own Domain Controller?

Do not create a controller merely because there is a new event type.

Create one when the responsibility has a distinct durable lifecycle.

A useful test:

```text
Does this responsibility have:

1. its own durable desired/observed state?
2. asynchronous progress?
3. retries or recovery?
4. independent failure modes?
5. a meaningful state machine?
6. potentially independent scaling/ownership?
```

If most answers are yes, it probably deserves a separate controller.

---

## 6.1 Example: Reoptimization

A new trigger does NOT necessarily imply a new Optimization controller.

These:

```text
INITIAL_OPTIMIZATION
TRAFFIC_DRIFT
NEW_GPU_GENERATION
NEW_RUNTIME_VERSION
COST_REOPTIMIZATION
```

can all become:

```text
OptimizationJob
```

owned by the same:

```text
OptimizationController
```

because once the request exists, the durable lifecycle is the same.

---

## 6.2 Example: Serving Deployment vs Serving Optimization

These do have different durable lifecycles.

### Serving Deployment

```text
REQUESTED
  ->
ALLOCATING
  ->
STARTING
  ->
MODEL_LOADING
  ->
WARMING
  ->
READY
  ->
PUBLISHED
  ->
SERVING
  ->
DRAINING
  ->
RELEASED
```

### Serving Optimization Feedback Loop

```text
MONITORING
  ->
EVALUATING_PROFILE_VALIDITY
  ->
REOPTIMIZATION_REQUESTED
  ->
WAITING_FOR_PROFILE
  ->
PROFILE_AVAILABLE
  ->
ROLLOUT_REQUESTED
  ->
MONITORING
```

Those are two different domain state machines.

Therefore two controllers are justified.

---

# 7. One Platform with Multiple Domain Controllers
# Example: ML Serving Platform

An ML Serving Platform is not simply:

```text
ServingController
```

It may contain several independently understandable controllers.

For our current design, consider two.

---

## 7.1 State Machine A — Serving Deployment Lifecycle

Domain object:

```text
ServingDeployment
```

State machine:

```text
REQUESTED -> ALLOCATING -> ALLOCATED -> STARTING
                                      |
                                      v
                                MODEL_LOADING
                                      |
                                      v
                                   WARMING
                                      |
                                      v
                                    READY
                                      |
                                      v
                                  PUBLISHED
                                      |
                                      v
                                   SERVING
                                      |
                                      v
                                  DRAINING
                                      |
                                      v
                                  WITHDRAWN
                                      |
                                      v
                                  STOPPING
                                      |
                                      v
                                  RELEASED
```

Controller:

```text
ServingDeploymentController
```

Responsibilities:

```text
resolve immutable model/profile
request resources
create runtime workload
observe startup
load model
coordinate warmup/readiness
publish/withdraw routing membership
drain
release resources
```

---

## 7.2 State Machine B — Serving Optimization Lifecycle

Domain object:

```text
ServingOptimizationState
```

or a durable optimization-tracking object associated with the deployment.

State machine:

```text
                 +------------------------------+
                 |                              |
                 v                              |
             MONITORING                         |
                 |                              |
                 v                              |
      EVALUATING_PROFILE_VALIDITY               |
            /              \                    |
           /                \                   |
       VALID              INVALIDATED           |
         |                    |                 |
         +--------------------|-----------------+
                              |
                              v
                  REOPTIMIZATION_REQUESTED
                              |
                              v
                   WAITING_FOR_PROFILE
                              |
                              v
                     PROFILE_AVAILABLE
                              |
                              v
                    ROLLOUT_REQUESTED
                              |
                              +------------------+
```

Controller:

```text
ServingOptimizationController
```

Responsibilities:

```text
consume observed traffic/SLO facts
evaluate reoptimization policy
deduplicate/cool down triggers
create OptimizationJob
track resulting profile publication
request rollout of newly selected profile
```

---

## 7.3 Both Controllers Inside One Serving Platform

```text
                       ML SERVING PLATFORM
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
  ServingDeploymentController       ServingOptimizationController
              |                                 |
              |                                 |
              v                                 v
   ServingDeploymentStore          OptimizationFeedbackState
              |                                 |
              |                                 |
              v                                 v
  Runtime / Router / Scheduler      ReoptimizationPolicy
              |                                 ^
              |                                 |
              v                                 |
        Serving Traffic                       |
              |                                 |
              +----------> Metrics ------------+
                            Platform
                               |
                               v
                       Traffic/SLO Analyzer
```

The controllers cooperate, but they do not own the same state machine.

---

# 8. Cross-Controller Communication

Controllers should communicate through durable domain objects or explicit APIs/events, not by reaching into each other's private state.

Good:

```text
ServingOptimizationController
    ->
Create OptimizationJob
    ->
OptimizationController
```

Good:

```text
OptimizationController
    ->
Publish ModelBundleVersion / ServingProfiles
    ->
ServingOptimizationController observes completion
```

Weak:

```text
ServingOptimizationController
    directly calls OptimizationController.internal_generate_candidates()
```

The durable object is the contract:

```text
OptimizationJob
```

not one controller's implementation.

---

# 9. Platform Composition

A generic ML platform often has this structure:

```text
                         API / User Intent
                                |
                                v
                         Durable Domain State
                                |
                +---------------+---------------+
                |                               |
                v                               v
        Domain Controller A             Domain Controller B
                |                               |
                +---------------+---------------+
                                |
                     Policies / Analyzers
                                |
                +---------------+---------------+
                |                               |
                v                               v
           Worker Pool                    External Platform
                |                               |
                +---------------+---------------+
                                |
                                v
                         Observed State
```

A platform therefore typically contains:

```text
APIs
Domain objects
Durable stores
Domain controllers
Policy engines
Observers/analyzers
Workers/executors
Data-plane systems
```

Controllers are the durable decision-making backbone, but not the entire platform.

---

# 10. Same Abstractions Across ML Platforms

The exact domain changes, but the underlying design repeats.

---

# 11. ML Serving Platform

## Durable Domain Objects

```text
ServingDeployment
ServingInstance
ServingOptimizationState
Rollout
RoutingMembership
```

## Controllers

```text
ServingDeploymentController
ServingOptimizationController
RolloutController?       # if rollout lifecycle becomes independently complex
```

## Analyzer / Policy

```text
Traffic/SLO Analyzer
ReoptimizationPolicy
ReadinessPolicy
RolloutPolicy
```

## Workers / External Executors

```text
runtime launcher
model loader
warmup executor
router
GPU Fleet Scheduler
```

## Diagram

```text
                     ML SERVING PLATFORM
                            |
        +-------------------+-------------------+
        |                                       |
        v                                       v
ServingDeploymentController          ServingOptimizationController
        |                                       |
        v                                       v
ServingDeploymentStore               Feedback/Optimization State
        |                                       |
        v                                       v
 Runtime / Router / Fleet             ReoptimizationPolicy
        ^                                       ^
        |                                       |
        +---------------- Serving Metrics ------+
                            |
                            v
                    Traffic/SLO Analyzer
```

---

# 12. Model Optimization Platform

## Durable Domain Objects

```text
OptimizationJob
ServingProfileCandidate
OptimizationExperiment
OptimizationResult
```

## Primary State Machine

```text
PENDING
  ->
GENERATING_CANDIDATES
  ->
RUNNING_EXPERIMENTS
  ->
QUALIFYING
  ->
RANKING
  ->
PUBLISHING
  ->
COMPLETED
```

## Controller

```text
OptimizationController
```

The trigger can be:

```text
initial optimization
traffic drift
new hardware
new runtime
cost reoptimization
```

but they all create the same durable `OptimizationJob`.

## Workers

```text
quantization worker
conversion worker
compilation worker
benchmark worker
quality evaluation worker
```

## External Platforms

```text
GPU Fleet Scheduler
Artifact Store
Model Registry
Metrics/Benchmark Store
```

## Diagram

```text
                  MODEL OPTIMIZATION PLATFORM
                            |
                    OptimizationJobStore
                            |
                            v
                   OptimizationController
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
         Quantization   Compilation   Benchmark /
            Worker         Worker     Evaluation
              |             |             |
              +-------------+-------------+
                            |
                            v
                     ExperimentStore
                            |
                            v
                       Profile Ranker
                            |
                            v
                      Bundle Publisher
```

---

# 13. GPU Scheduling Platform

## Durable Domain Objects

```text
ResourceIntent
QueueEntry
ResourceAllocation
Node/Topology State
Reservation
```

## State Machine Example

```text
PENDING
  ->
SCHEDULING
  ->
ALLOCATED
```

or terminal:

```text
FAILED
CANCELLED
RELEASED
LOST
```

## Controller / Decision Components

Depending on implementation, scheduling may contain multiple logical controllers:

```text
Queue / Admission Controller
Scheduling Controller
Allocation / Reconciliation Controller
Capacity / Topology Controller
```

The important boundary is:

```text
Scheduler:
    decides resource ownership

Workload/Domain Controller:
    decides workload lifecycle
```

The scheduler should not decide whether a model should restart or whether an optimization job should retry its benchmark.

It decides:

```text
which resource intent gets which physical resources
```

## Diagram

```text
                    GPU SCHEDULING PLATFORM
                             |
                       ResourceIntent
                             |
                             v
                     Durable Queue State
                             |
                             v
                     Scheduling Controller
                             |
             +---------------+----------------+
             |                                |
             v                                v
       Policy / Quota                     Topology /
       Fair-share State                 Capacity State
             |                                |
             +---------------+----------------+
                             |
                             v
                    ResourceAllocation
                             |
                             v
                       GPU / Node Fleet
```

---

# 14. Same Control-Plane Pattern Across All Three

```text
+----------------------+----------------------+----------------------+
|      ML SERVING      |   MODEL OPTIMIZATION |    GPU SCHEDULING    |
+----------------------+----------------------+----------------------+
| ServingDeployment    | OptimizationJob      | ResourceIntent       |
|                      |                      |                      |
| durable lifecycle    | durable lifecycle    | durable lifecycle    |
+----------+-----------+----------+-----------+----------+-----------+
           |                      |                      |
           v                      v                      v
 ServingDeployment        Optimization          Scheduling
 Controller               Controller            Controller
           |                      |                      |
           v                      v                      v
 Runtime actions          Experiment actions     Allocation actions
           |                      |                      |
           v                      v                      v
 Observed runtime         Experiment results     Fleet state
```

The nouns change. The control-plane abstraction does not.

---

# 15. Unified Generic Pattern

```text
              DOMAIN INTENT
                   |
                   v
          +-------------------+
          | Durable Domain    |
          | Object / State    |
          +---------+---------+
                    |
                    v
          +-------------------+
          | Domain Controller |
          +---------+---------+
                    |
       +------------+-------------+
       |                          |
       v                          v
  Policy / Analyzer         Worker / Executor
       |                          |
       +------------+-------------+
                    |
                    v
             External State
                    |
                    v
              Observation
                    |
                    +-------> controller reconciles again
```

Examples:

```text
ServingDeployment
    -> ServingDeploymentController
    -> Runtime

OptimizationJob
    -> OptimizationController
    -> Benchmark/Compiler Workers

ResourceIntent
    -> SchedulingController
    -> GPU Fleet
```

---

# 16. Domain Controller Design Rules

## Rule 1 — One coherent lifecycle per controller

Prefer:

```text
ServingDeploymentController
ServingOptimizationController
```

over:

```text
MegaServingController
```

that owns deployment, traffic statistics, autoscaling, optimization, rollout, routing, and resource scheduling.

---

## Rule 2 — Persist before assuming progress exists

If a lifecycle transition matters after a crash, write enough durable state to recover it.

---

## Rule 3 — Actions must be idempotent

A reconcile loop may execute an action more than once.

Therefore:

```text
ensure_runtime_exists(instance_id)
```

is safer than:

```text
create_runtime()
```

with an anonymous new identity every retry.

---

## Rule 4 — Deterministic child identity

If a controller repeatedly ensures child resources:

```text
ServingDeployment D
    ->
ServingInstance D/instance-17
```

then retry should discover the same child, not create duplicates.

---

## Rule 5 — Controller owns decisions, executor owns work

```text
Controller:
    benchmark candidate C17

Worker:
    actually run benchmark
```

---

## Rule 6 — Analyzer owns derived facts

```text
Analyzer:
    traffic distribution drift = material

Controller:
    transition to REOPTIMIZATION_REQUESTED
```

---

## Rule 7 — Policy is explicit

Do not bury rules such as:

```text
when TTFT increases 20%, re-optimize
```

inside arbitrary controller branches.

Represent:

```text
ReoptimizationPolicy
QuotaPolicy
RolloutPolicy
ReadinessPolicy
```

as explicit versioned contracts where practical.

---

## Rule 8 — Cross-platform contracts are durable objects

Use:

```text
ResourceIntent
OptimizationJob
ServingDeployment
ModelBundle
```

as platform boundaries.

Do not couple platforms by internal function calls.

---

## Rule 9 — Ownership must be singular

For every important fact ask:

```text
Who is authoritative?
```

Example:

```text
ResourceAllocation ownership
    -> GPU Scheduler

ServingInstance lifecycle
    -> Serving Platform

OptimizationExperiment state
    -> Model Optimization Platform

Model Bundle version
    -> Model Registry / Bundle Publisher
```

---

## Rule 10 — Status is projection, not necessarily truth

A cached status field:

```text
ready_replicas = 18
```

may be a projection of runtime state, routing state, and resource allocation state.

The controller must know which underlying systems are authoritative.

---

# 17. Controller Coordination Example

Consider production traffic invalidating a ServingProfile.

```text
Metrics Platform
      |
      v
Traffic/SLO Analyzer
      |
      v
ObservedTrafficDistribution O7
      |
      v
ReoptimizationPolicy
      |
      v
ServingOptimizationController
      |
      | writes
      v
OptimizationJob J42
      |
      v
OptimizationController
      |
      +--> Transformation Workers
      +--> Benchmark Workers
      +--> Quality Evaluators
      |
      v
OptimizationResult
      |
      v
New ModelBundleVersion
      |
      v
ServingOptimizationController
      |
      | requests rollout
      v
ServingDeploymentController
      |
      v
New ServingInstances
```

No controller needs to know the private implementation of another controller.

They coordinate through durable contracts.

---

# 18. Why This Architecture Scales Organizationally

The abstraction is not only technical.

It supports ownership boundaries.

For example:

```text
Serving Team
    owns:
        ServingDeploymentController
        ServingOptimizationController

Optimization Team
    owns:
        OptimizationController
        experiment workers

Fleet Team
    owns:
        SchedulingController
        allocation state

Metrics Team
    owns:
        telemetry/analyzers
```

Each team can evolve its internals as long as the domain contracts remain stable.

---

# 19. Failure Isolation

Separate controllers reduce blast radius.

Example:

```text
Traffic Analyzer fails
```

should not prevent:

```text
ServingDeploymentController
```

from restarting failed serving instances.

Similarly:

```text
OptimizationController outage
```

should not stop currently healthy serving deployments.

This is another reason not to merge unrelated lifecycles.

---

# 20. Principal-Level Mental Model

When designing an ML platform, ask in this order:

```text
1. What are the durable domain objects?

2. What lifecycle/state machine does each object have?

3. Which component is authoritative for each state?

4. Which Domain Controller owns each lifecycle?

5. What external state must each controller observe?

6. Which policies affect its decisions?

7. Which workers/executors perform side effects?

8. Which other platforms are called through explicit durable contracts?

9. How does the controller recover after restart/failover?

10. Are two responsibilities actually separate state machines
    that deserve separate controllers?
```

---

# 21. Interview Example — ML Serving

> **“I model an ML platform as a collection of domain control loops rather than one large orchestration service. Each important domain object has durable desired/observed state and an explicit lifecycle state machine. A Domain Controller owns that lifecycle and continuously reconciles durable intent against external observations using idempotent actions. Analyzers derive facts, policy interprets those facts, and workers perform expensive side effects. For example, my Serving Platform has a ServingDeploymentController that owns allocate→start→load→warm→publish→drain, and a separate ServingOptimizationController that owns monitor→detect qualification invalidation→request reoptimization→wait for a new profile→request rollout. Both are part of the same Serving Platform, but they own different durable state machines. The same abstraction repeats in Model Optimization through OptimizationJob/OptimizationController and in GPU Scheduling through ResourceIntent/SchedulingController.”**

---

# 22. Final Abstraction

The reusable hierarchy is:

```text
PLATFORM
   |
   +-- Domain Object A
   |      |
   |      +-- Durable State
   |      +-- State Machine
   |      +-- Domain Controller
   |
   +-- Domain Object B
   |      |
   |      +-- Durable State
   |      +-- State Machine
   |      +-- Domain Controller
   |
   +-- Policies
   |
   +-- Observers / Analyzers
   |
   +-- Workers / Executors
   |
   +-- APIs
   |
   +-- Data Plane / External Platforms
```

And the most important principle is:

> **Split controllers by durable lifecycle ownership, not by arbitrary code/module boundaries. A platform is the composition of those independently recoverable control loops around a shared domain.**
