# Model Optimization Traffic Distribution
## Expected Traffic, Observed Traffic, Qualification Drift, Model Adaptation, and Closed-Loop Serving

## Purpose

This document defines the closed-loop relationship between:

```text
Product / Application Team
ML Serving Platform
Metrics / Observability Platform
Model Optimization Platform
Model Evaluation Platform
Training Platform
GPU Scheduling Platform
Model Registry / Bundle Publication
```

when real production traffic differs from the workload assumptions used to qualify a model and its ServingProfiles.

The central idea is:

> **Model Optimization is iterative, not a one-time post-training step. Initial ServingProfiles are qualified against Product-supplied expected traffic. Canary and production produce observed traffic. Model Optimization defines the ObservationPolicy that says when enough production evidence has been collected. Serving then determines whether the deployed qualification is invalid. A ModelAdaptationController decides whether the remediation is serving-profile reoptimization, model evaluation/retraining, or both.**

---

# Zoom Level 1 — Detailed Principal / Staff+ Design

# 1. Two Traffic Distributions

We lock two distinct concepts.

## 1.1 `expected_traffic_distribution`

Supplied initially by the Product/Application team.

It represents:

> **What traffic do we believe this model will receive?**

Typical dimensions:

```text
prompt/input length distribution
output length distribution
request arrival-rate distribution
concurrency distribution
streaming percentage
tool-call percentage
prefix-sharing characteristics
request class mix
language/domain mix
sampling/generation characteristics
```

Example:

```text
expected_chat_v3

prompt length:
    p50 = 2K
    p95 = 8K
    p99 = 16K

output length:
    p50 = 300
    p95 = 1K

concurrency:
    p50 = 20
    p99 = 64

streaming:
    95%

tool calls:
    18%

language mix:
    English = 80%
    Spanish = 10%
    Japanese = 10%
```

It is available before the model has production traffic.

Therefore the first Model Optimization run uses:

```text
CanonicalModel
+
ServingRequirements
+
expected_traffic_distribution
```

---

## 1.2 `observed_traffic_distribution`

Derived from actual serving telemetry.

It represents:

> **What traffic is the deployment really receiving?**

It may be computed from:

```text
Canary traffic
Production traffic
Regional traffic
Tenant / product segment traffic
```

The dimensions should be compatible with the expected distribution so that the two can be compared:

```text
prompt length histogram
output length histogram
arrival rate
concurrency
streaming mix
tool-call mix
prefix-sharing behavior
language/domain mix
request class mix
```

Observed traffic is not known during initial Model Optimization.

It appears only after serving begins.

---

# 2. Expected and Observed Traffic Have Different Owners

The ownership is intentionally different.

```text
Product Team
    owns:
        expected_traffic_distribution
        product SLO / quality intent

Metrics / Serving Telemetry
    owns:
        live observed measurements
        observed_traffic_distribution snapshots

Model Optimization Platform
    owns:
        ObservationPolicy
        the evidence requirements under which its qualification
        may be reassessed

Serving Platform
    owns:
        deployed qualification lifecycle

Model Adaptation
    owns:
        remediation selection when qualification is invalid
```

The Product Team should not define statistical sampling rules.

The Metrics Platform should not decide whether to reoptimize or retrain.

The Serving Platform should not decide how many samples constitute statistically meaningful evidence.

The OptimizationController should not silently become a TrainingController.

---

# 3. ObservationPolicy

This is a critical production contract.

`ObservationPolicy` answers:

> **When has enough real production evidence been collected to compare observed traffic and deployed behavior against the qualification assumptions?**

It is owned by the **Model Optimization Platform** because that platform generated the original empirical qualification and knows what evidence is required to reassess it.

Conceptually:

```text
ObservationPolicy {
    policy_id
    policy_version

    minimum_observation_window
    minimum_request_count

    required_dimensions[]
    minimum_samples_per_dimension_or_segment

    required_regions_or_traffic_classes[]

    quality_sampling_policy?

    confidence_requirements?

    snapshot_frequency

    expiration / refresh policy
}
```

Example:

```text
policy:
    opt-observation-v7

minimum observation:
    6 hours
    AND 100,000 requests

required traffic dimensions:
    prompt length
    output length
    concurrency
    tool-call mix
    language mix

minimum samples:
    5,000 per high-priority traffic segment
```

The Metrics / Observability Platform **executes** this policy.

It does not invent it.

---

# 4. ObservationPolicy Flow

```text
Model Optimization Platform
      |
      | publishes / associates
      v
ObservationPolicy OP7
      |
      v
Metrics / Traffic Analyzer
      |
      | collect until OP7 is satisfied
      v
ObservedTrafficDistribution O17
      +
ObservedServingMetrics S17
      +
ObservationEvidence metadata
```

The policy should be versioned so operators can answer:

```text
Why did the system consider this production snapshot trustworthy?
```

---

# 5. Initial Optimization Flow

Before deployment:

```text
Product Team
      |
      | expected traffic + SLO + quality intent
      v
ServingRequirements
      |
      v
Model Optimization Platform
      |
      +-- candidate generation
      +-- quantization / compilation
      +-- performance benchmarking
      +-- quality evaluation
      +-- qualification
      +-- ranking
      |
      +-- emits ObservationPolicy
      |
      v
Primary + Alternative ServingProfiles
```

The initial empirical evidence means:

> **This ServingProfile was qualified under expected traffic E1.**

It does not yet prove that production traffic will exactly match E1.

---

# 6. Model Bundle Representation

For every selected ServingProfile:

```text
ServingProfile
    |
    +-- QualificationEvidence
            |
            +-- traffic_distribution
            |      source = EXPECTED or OBSERVED
            |
            +-- benchmark_workload
            |
            +-- PerformanceEnvelope
            |
            +-- QualityValidation
            |
            +-- observation_policy_ref
```

For initial optimization:

```text
traffic_distribution.source = EXPECTED
```

For later production-driven adaptation:

```text
traffic_distribution.source = OBSERVED
```

The live observed traffic stream is **not** mutable Model Bundle state.

Only the immutable snapshot used to qualify a published profile enters that profile's `QualificationEvidence`.

The `ObservationPolicy` may be embedded or referenced by immutable version; the important semantic is that Serving/Metrics can discover the exact evidence policy associated with the selected profile.

---

# 7. Why Traffic Distribution Matters

A serving-performance number without its workload is incomplete.

Example:

```text
sustainable_qps = 18.4
```

does not mean much unless we know:

```text
prompt distribution
output distribution
concurrency
arrival process
streaming mix
prefix-hit characteristics
```

Likewise, a model-quality score can become misleading if production traffic moves into a different semantic region.

Example:

```text
Canonical quality:
    tool-call success = 92%

Production traffic:
    tool calls become much more complex
    and now dominate request volume
```

Even if TTFT and TPOT remain healthy, the deployed model may no longer meet product quality requirements.

Therefore:

> **Traffic distribution is part of both serving-performance qualification and model-quality validity.**

---

# 8. Production Creates a Closed Loop

After profiles are published:

```text
Model Bundle
    |
    v
Canary
    |
    v
Production
    |
    v
Observed Traffic
+
Observed Serving Metrics
+
Observed Quality Evidence
```

The loop becomes:

```text
EXPECT
   |
   v
OPTIMIZE
   |
   v
DEPLOY
   |
   v
OBSERVE
   |
   v
VALIDATE DEPLOYED QUALIFICATION
   |
   +-- valid ----------------------> continue
   |
   +-- invalid --------------------> ADAPT
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
              Serving Reoptimization              Model Adaptation
                     |                                   |
                     v                                   v
             new ServingProfiles                  Evaluation / Training
                                                         |
                                                         v
                                                  new CanonicalModel
                                                         |
                                                         v
                                                  Model Optimization
```

---

# 9. Canary Is Not the Only Observation Point

Canary is useful because it is the first place where real workload evidence can be collected.

But traffic and quality observation continue after full production rollout.

```text
Canary
    |
    +--> observed traffic
    +--> serving metrics
    +--> sampled quality evidence

Production
    |
    +--> continuously updated observed traffic
    +--> serving metrics
    +--> quality evidence
```

Traffic may evolve because of:

```text
new product features
new clients
seasonality
regional growth
prompt changes
tool adoption
long-context adoption
new routing policies
new languages/domains
product launches
```

Therefore production observation is a lifecycle responsibility, not a one-time canary gate.

---

# 10. Raw Traffic Drift Is Not Automatically an Adaptation Trigger

Avoid:

```text
distribution changed
    ->
rerun expensive optimization / retraining
```

Example:

```text
Expected prompt p95:
    8K

Observed prompt p95:
    12K
```

but:

```text
TTFT p99:
    280 ms

TTFT SLO:
    500 ms

TPOT p99:
    31 ms

TPOT SLO:
    50 ms

quality:
    healthy

GPU efficiency:
    healthy
```

The old deployment may still be completely acceptable.

So the real question is:

> **Has traffic drift materially invalidated the current model + ServingProfile qualification?**

---

# 11. Qualification Envelope

A deployed qualification includes several dimensions.

## Serving realization envelope

```text
traffic distribution
TTFT
TPOT
throughput
sustainable QPS
memory usage
KV-cache behavior
error/OOM rate
startup characteristics
cost / capacity behavior
```

## Model-quality envelope

```text
task success
tool-call success
structured output correctness
language/domain quality
safety behavior
business/task quality metrics
```

Observed production behavior is compared against these assumptions.

The system asks:

```text
Is the selected ServingProfile still valid?
Is the CanonicalModel still good enough for the observed workload?
Are both still valid?
```

---

# 12. Four Signal Classes

Adaptation logic should consume four categories.

## 12.1 Traffic Drift

Examples:

```text
prompt length shift
output length shift
concurrency shift
arrival-rate shift
streaming mix shift
tool-call mix shift
prefix-sharing shift
language/domain shift
request-class shift
```

## 12.2 Serving Behavior

Examples:

```text
TTFT headroom
TPOT headroom
queue wait
throughput
OOM/error rate
HBM pressure
KV-cache pressure
runtime instability
```

## 12.3 Economic / Capacity Behavior

Examples:

```text
cost per token
GPU utilization
overprovisioning
underprovisioning
capacity efficiency
```

## 12.4 Model Quality Behavior

Examples:

```text
task success degradation
tool-call correctness regression
structured-output failures
language/domain quality regression
safety regression
human/user feedback degradation
business KPI degradation
```

This fourth category is what makes the system a **model adaptation loop**, not merely a serving-performance feedback loop.

---

# 13. How Production Quality Is Measured

Traffic drift tells us:

```text
inputs changed
```

It does not automatically prove:

```text
model quality degraded
```

Quality evidence may come from:

```text
online labels
delayed ground truth
user feedback
task completion
tool-call correctness
business KPI
sampled offline replay
LLM-as-judge / rubric evaluation
human evaluation
safety evaluation
```

Therefore a **Model Evaluation Platform** may participate in the loop.

The Evaluation Platform owns:

```text
EvaluationJob
EvaluationController
quality measurement
quality report
```

It answers:

> **Does the current CanonicalModel still satisfy the required quality envelope on representative observed traffic?**

---

# 14. Reoptimization Policy vs Adaptation Decision

We should not overload one policy.

Use two levels.

## 14.1 ObservationPolicy

Owned by Model Optimization.

Answers:

```text
Do we have enough evidence to make a decision?
```

## 14.2 Adaptation / Qualification Policy

Consumes trustworthy evidence and answers:

```text
Is the deployed qualification still valid?
```

It may use:

```text
traffic drift
serving SLO headroom
cost efficiency
quality evidence
```

The Serving Platform owns the **deployed qualification lifecycle**.

But it should not decide the specific remediation implementation.

---

# 15. Generic Invalidation Condition

Illustrative logic:

```text
if ObservationPolicy is satisfied
    AND (
        serving_profile_qualification_invalid
        OR model_quality_qualification_invalid
    ):
        create ModelAdaptationRequest
```

ServingProfile invalidation might mean:

```text
material traffic shift
AND
(
    SLO headroom too small
    OR profile outside qualified workload envelope
    OR cost efficiency materially degraded
)
```

Model-quality invalidation might mean:

```text
observed traffic moved into a different semantic mix
AND
quality evaluation / production quality evidence violates policy
```

The point is not the exact boolean expression.

The point is to distinguish:

```text
EVIDENCE SUFFICIENCY
QUALIFICATION INVALIDATION
REMEDIATION SELECTION
```

---

# 16. Why ServingOptimizationController Should Not Decide Retraining

`ServingOptimizationController` knows:

```text
what is deployed
what qualification evidence applies
what production evidence is being observed
whether deployment qualification is still valid
```

It should not own:

```text
fine-tuning strategy
training dataset creation
training topology
training lifecycle
model evaluation methodology
```

Therefore when qualification is invalid:

```text
ServingOptimizationController
    ->
ModelAdaptationRequest
```

not:

```text
ServingOptimizationController
    ->
TrainingJob directly
```

This keeps lifecycle ownership clean.

---

# 17. ModelAdaptationRequest

Conceptually:

```text
ModelAdaptationRequest {
    adaptation_request_id

    model_bundle_ref
    canonical_model_ref
    current_serving_profile_id

    serving_requirements_ref

    expected_traffic_distribution_ref
    observed_traffic_distribution_ref

    observed_serving_metrics_ref
    observed_quality_evidence_ref

    observation_policy_id
    qualification_policy_id

    trigger_reasons[]

    created_at
}
```

It says:

> **The assumptions under which this deployment was qualified are no longer valid. Diagnose and choose remediation.**

---

# 18. ModelAdaptationController

A separate `ModelAdaptationController` owns this durable lifecycle.

It may live organizationally inside the broader Model Optimization / Model Lifecycle platform, but it is **not the same controller as `OptimizationController`**.

Why separate them?

Because:

```text
OptimizationController
    owns:
        ServingProfile search lifecycle

ModelAdaptationController
    owns:
        diagnosis / remediation routing lifecycle
```

The latter may route work to:

```text
Model Optimization
Model Evaluation
Training
or a sequence of them
```

---

# 19. Three Adaptation Outcomes

## 19.1 Case A — Serving realization problem

Example:

```text
model quality:
    PASS

TTFT:
    FAIL / low headroom

TPOT:
    marginal

cost:
    poor
```

Then:

```text
ModelAdaptationRequest
      |
      v
SERVING_REOPTIMIZATION
      |
      v
OptimizationJob
      |
      v
new ServingProfiles
```

Same canonical model.

---

## 19.2 Case B — Model-quality problem

Example:

```text
TTFT:
    PASS

TPOT:
    PASS

quality on observed workload:
    FAIL
```

No hardware/runtime sweep can fix a canonical model that no longer satisfies the semantic workload.

Then:

```text
ModelAdaptationRequest
      |
      v
MODEL_QUALITY_INVALIDATED
      |
      v
EvaluationJob
      |
      v
Training / Fine-tuning required
      |
      v
new CanonicalModel
      |
      v
Model Optimization
      |
      v
new ServingProfiles
```

---

## 19.3 Case C — Both model and serving realization changed

Example:

```text
traffic becomes:
    much longer
    more concurrent
    more coding/tool heavy

serving SLO:
    failing

model quality:
    failing
```

Then the likely sequence is:

```text
Model Evaluation
    |
    v
Training / Model Adaptation
    |
    v
new CanonicalModel
    |
    v
Model Optimization
    |
    v
new ServingProfiles
```

It is usually wasteful to deeply optimize the old model first if a replacement model is already required.

---

# 20. Platforms and Domain Responsibilities

## 20.1 Product / Application Team

Owns:

```text
ServingRequirements
expected_traffic_distribution
product SLO
quality constraints
business requirements
```

Does not own:

```text
ObservationPolicy
drift statistics
optimization search
training lifecycle
```

---

## 20.2 Metrics / Observability Platform

Logical components:

```text
Metrics ingestion
Traffic/SLO Analyzer
TrafficDistributionStore
ServingMetricStore
Quality-signal ingestion
```

Executes:

```text
ObservationPolicy
```

Produces:

```text
ObservedTrafficDistribution
ObservedServingMetrics
TrafficDriftFacts
SloHeadroomFacts
CostEfficiencyFacts
```

The analyzer answers:

> **What is happening?**

It does not decide remediation.

---

## 20.3 ML Serving Platform

Important Domain Controllers:

```text
ServingDeploymentController
ServingOptimizationController
```

### ServingDeploymentController

Owns:

```text
allocate
start
load
warm
publish
serve
drain
replace
rollout
```

### ServingOptimizationController

Owns:

```text
track selected profile qualification

consume ObservationPolicy satisfaction

consume observed traffic/serving/quality facts

evaluate deployed qualification validity

deduplicate / cool down invalidation

create ModelAdaptationRequest

wait for resolution

request rollout when resolution produces new bundle/profile
```

It does not choose:

```text
retrain vs reoptimize
```

---

## 20.4 Model Optimization / Adaptation Platform

Contains at least two distinct Domain Controllers.

### ModelAdaptationController

Owns:

```text
ModelAdaptationRequest
diagnosis
remediation selection
coordination with Evaluation / Training / Optimization
```

### OptimizationController

Owns:

```text
OptimizationJob
candidate generation
experiment execution
qualification
ranking
ServingProfile publication
```

The Platform also owns:

```text
ObservationPolicy generation/versioning
```

because it knows what evidence is required to reassess its qualification.

---

## 20.5 Model Evaluation Platform

Important Domain Controller:

```text
EvaluationController
```

Owns:

```text
EvaluationJob
representative observed-traffic sample
quality measurement
quality regression analysis
quality report
```

It does not retrain or reconfigure serving.

---

## 20.6 Training Platform

Important Domain Controller:

```text
TrainingController
```

Owns:

```text
TrainingJob
fine-tuning/retraining lifecycle
checkpointing
canonical model output
```

When training produces a new CanonicalModel:

```text
new CanonicalModel
    ->
Model Optimization
```

because new canonical model truth requires new serving qualification.

---

## 20.7 GPU Scheduling Platform

Important Domain Controller:

```text
SchedulingController
```

Owns:

```text
ResourceIntent
ResourceAllocation
physical GPU ownership
```

It schedules temporary resources for:

```text
optimization experiments
evaluation jobs
training jobs
```

and production resources for:

```text
ServingInstances
```

It does not decide model/profile validity.

---

## 20.8 Model Registry / Bundle Publication

Owns:

```text
immutable ModelBundleVersion
manifest digest
aliases
profile publication
canonical-model publication
```

Traffic-driven serving reoptimization may produce:

```text
same CanonicalModel
new ServingProfiles
new ModelBundleVersion
```

Retraining produces:

```text
new CanonicalModel
new ServingProfiles
new ModelBundleVersion
```

---

# 21. Domain State Machines

## 21.1 Serving Optimization State Machine

```text
                    +-------------------------------------+
                    |                                     |
                    v                                     |
                MONITORING                                |
                    |                                     |
                    v                                     |
         COLLECTING_OBSERVATION                           |
                    |                                     |
        ObservationPolicy satisfied                      |
                    |                                     |
                    v                                     |
       EVALUATING_QUALIFICATION                           |
            /                       \                     |
           /                         \                    |
       VALID                      INVALIDATED             |
         |                            |                   |
         +----------------------------|-------------------+
                                      |
                                      v
                           ADAPTATION_REQUESTED
                                      |
                                      v
                         WAITING_FOR_RESOLUTION
                                      |
                                      v
                           RESOLUTION_AVAILABLE
                                      |
                         /            |            \
                        /             |             \
                  NO_CHANGE      NEW_PROFILE     NEW_MODEL
                        |             |             |
                        |             +------+------+
                        |                    |
                        |                    v
                        |            ROLLOUT_REQUESTED
                        |                    |
                        +--------------------+
                                      |
                                      v
                                  MONITORING
```

Owned by:

```text
ServingOptimizationController
```

Notice that this state machine does **not** contain:

```text
RETRAIN
REQUANTIZE
CHANGE TP
```

Those are remediation decisions outside its domain.

---

## 21.2 Model Adaptation State Machine

```text
PENDING
   |
   v
DIAGNOSING
   |
   +-------------------+----------------------+------------------+
   |                   |                      |                  |
   v                   v                      v                  v
NO_CHANGE      SERVING_REOPTIMIZE      EVALUATE_MODEL       BOTH_SUSPECT
                       |                      |                  |
                       v                      v                  v
                OPTIMIZATION_JOB        EVALUATION_JOB      EVALUATION_JOB
                                              |                  |
                                              v                  v
                                       MODEL_QUALITY_OK?   MODEL_QUALITY_OK?
                                         /         \         /         \
                                       yes         no      yes         no
                                        |           |       |           |
                                        v           v       v           v
                                   COMPLETE      TRAINING  OPTIMIZE   TRAINING
                                                   |                    |
                                                   v                    v
                                             NEW_MODEL             NEW_MODEL
                                                   |                    |
                                                   +---------+----------+
                                                             |
                                                             v
                                                       OPTIMIZATION
                                                             |
                                                             v
                                                         COMPLETE
```

Owned by:

```text
ModelAdaptationController
```

---

## 21.3 Optimization Job State Machine

```text
PENDING
    |
    v
GENERATING_CANDIDATES
    |
    v
RUNNING_EXPERIMENTS
    |
    v
QUALIFYING
    |
    v
RANKING
    |
    v
PUBLISHING
    |
    v
COMPLETED
```

Owned by:

```text
OptimizationController
```

---

## 21.4 Evaluation State Machine

Simplified:

```text
PENDING
    |
    v
PREPARING_EVAL_SET
    |
    v
RUNNING_EVALUATION
    |
    v
AGGREGATING
    |
    v
QUALIFYING
    |
    v
COMPLETED
```

Owned by:

```text
EvaluationController
```

---

## 21.5 Training State Machine

Simplified for this document:

```text
PENDING
    |
    v
PREPARING
    |
    v
TRAINING
    |
    v
VALIDATING
    |
    v
EXPORTING_CANONICAL_MODEL
    |
    v
COMPLETED
```

Owned by:

```text
TrainingController
```

---

## 21.6 Serving Deployment State Machine

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
```

For replacement:

```text
SERVING
    ->
DRAINING
    ->
WITHDRAWN
    ->
STOPPING
    ->
RELEASED
```

Owned by:

```text
ServingDeploymentController
```

---

## 21.7 Scheduling State Machine

Simplified:

```text
PENDING
    |
    v
SCHEDULING
    |
    v
ALLOCATED
```

Then eventually:

```text
RELEASED
LOST
CANCELLED
```

Owned by:

```text
SchedulingController
```

---

# 22. Cross-Platform Durable Contracts

Controllers communicate through durable contracts, not internal function calls.

Important contracts:

```text
ServingRequirements
ExpectedTrafficDistribution
ObservationPolicy

ObservedTrafficDistribution
ObservedServingMetrics
ObservedQualityEvidence

ModelAdaptationRequest

OptimizationJob
EvaluationJob
TrainingJob

ResourceIntent
ResourceAllocation

ModelBundleVersion
CanonicalModel
ServingProfile

ServingDeployment
```

Good:

```text
ServingOptimizationController
    ->
creates ModelAdaptationRequest

ModelAdaptationController
    ->
creates OptimizationJob / EvaluationJob / TrainingJob
```

Weak:

```text
ServingOptimizationController
    ->
calls TrainingController.retrain() internally
```

The durable domain object is the platform boundary.

---

# 23. Initial Optimization End to End

```text
Product Team
    |
    | ServingRequirements
    | expected_traffic_distribution E1
    v
Model Optimization Platform
    |
    | OptimizationJob O1
    v
OptimizationController
    |
    | ResourceIntent(s)
    v
GPU Scheduling Platform
    |
    v
temporary optimization GPUs
    |
    v
Optimization Workers
    |
    v
qualified P0 / P1 / P2
    |
    +-- ObservationPolicy OP7
    |
    v
Bundle Publisher / Model Registry
    |
    v
ModelBundleVersion B42
    |
    v
ML Serving Platform
```

Each published profile records:

```text
traffic source = EXPECTED
traffic profile = E1
observation policy = OP7
```

---

# 24. Production Adaptation End to End

Suppose:

```text
P0
```

was qualified against:

```text
expected traffic E1
```

and production later produces:

```text
observed traffic O7
```

Flow:

```text
Serving Traffic
    |
    v
Metrics Platform
    |
    | execute ObservationPolicy OP7
    v
Traffic/SLO Analyzer
    |
    +-- O7
    +-- serving drift facts
    +-- SLO headroom
    +-- cost/efficiency facts
    |
    +--------------------------+
                               |
                               v
                    Quality Evaluation signals
                               |
                               v
                    ServingOptimizationController
                               |
                               | deployed qualification invalid?
                               |
                    +----------+----------+
                    |                     |
                   NO                    YES
                    |                     |
                    v                     v
                continue          ModelAdaptationRequest A2
                                          |
                                          v
                               ModelAdaptationController
                                  /        |        \
                                 /         |         \
                                v          v          v
                         REOPTIMIZE    EVALUATE    BOTH
                                |          |          |
                                v          v          v
                        Optimization    Evaluation   Evaluation
                            Job            Job         Job
                                                       |
                                           +-----------+-----------+
                                           |                       |
                                      quality OK               quality bad
                                           |                       |
                                           v                       v
                                       Optimize                  Train
                                                                   |
                                                                   v
                                                            new CanonicalModel
                                                                   |
                                                                   v
                                                            Model Optimization
                                                                   |
                                                                   v
                                                         new ServingProfiles
                                                                   |
                                                                   v
                                                        new ModelBundleVersion
                                                                   |
                                                                   v
                                                   ServingOptimizationController
                                                                   |
                                                                   v
                                                    ServingDeploymentController
                                                                   |
                                                                   v
                                                         new ServingInstances
```

---

# 25. Same Canonical Model vs New Canonical Model

## Serving-only adaptation

```text
Canonical Model M42
        |
        +-- Optimization Run O1
        |      traffic = expected E1
        |      -> P0 P1 P2
        |
        +-- Optimization Run O2
               traffic = observed O7
               -> P3 P4 P5
```

Possible changes:

```text
TP4 -> TP8
vLLM -> TensorRT-LLM
max_num_batched_tokens 8192 -> 32768
different CUDA Graph strategy
different GPU SKU
different quantization
different P/D topology
```

Canonical model does not change.

---

## Model-quality adaptation

```text
Canonical Model M42
        |
        | observed quality no longer acceptable
        v
Training / Fine-tuning
        |
        v
Canonical Model M43
        |
        v
Model Optimization
        |
        +-- P6
        +-- P7
        +-- P8
```

Now both model truth and serving realization change.

---

# 26. Why a New Immutable Bundle Version Is Published

Serving-only case:

```text
B42:
    Canonical M42
    P0 primary
    P1/P2 alternatives

B43:
    Canonical M42
    P3 primary
    P4/P5 alternatives
```

Model-retraining case:

```text
B44:
    Canonical M43
    P6 primary
    P7/P8 alternatives
```

Do not mutate prior bundle versions.

This preserves:

```text
auditability
rollback
reproducibility
deployment pinning
```

---

# 27. Control-Loop Stability

A feedback system needs damping.

Potential protections:

```text
ObservationPolicy minimum window

minimum request count

minimum samples per important segment

adaptation cooldown

one active ModelAdaptationRequest per deployment/model

deduplicate equivalent observed snapshots

require meaningful qualification impact

rate-limit automatic optimization/training requests

human approval for expensive or sensitive retraining
```

Without safeguards:

```text
small traffic change
    ->
adaptation
    ->
small traffic change
    ->
adaptation
```

can create an unstable control loop.

---

# 28. Policy Versioning

Persist:

```text
observation_policy_id
observation_policy_version

qualification_policy_id

traffic_snapshot_id
current_profile_id
canonical_model_id

trigger_reason

observed serving metrics
observed quality evidence

adaptation decision

decision timestamp
```

Then operators can answer:

```text
Why did the platform reoptimize this model yesterday?

Why did it decide to retrain rather than only change the ServingProfile?
```

---

# 29. Failure Handling

## Metrics Platform unavailable

Healthy serving continues.

Observation becomes stale; deployment does not fail simply because traffic analysis is unavailable.

---

## ObservationPolicy not yet satisfied

Do nothing.

Avoid making decisions from insufficient evidence.

---

## Quality evidence unavailable

The adaptation controller may:

```text
request EvaluationJob
```

rather than guessing that traffic drift implies quality regression.

---

## Serving reoptimization fails

Continue using current profile if operationally safe.

Escalate if production SLO cannot be sustained.

---

## Evaluation shows model quality is healthy

Do not retrain.

If serving performance is the issue, proceed with ServingProfile reoptimization.

---

## Training fails

Existing production deployment remains isolated from Training failure.

Do not have TrainingController own serving rollback.

---

## New profile/model publication succeeds but rollout fails

Existing deployment remains available.

`ServingDeploymentController` owns rollout/rollback recovery.

---

## GPU capacity unavailable

`OptimizationJob`, `EvaluationJob`, or `TrainingJob` remains pending/partial according to its platform policy.

Serving continues independently where safe.

---

# 30. Controller Ownership Summary

```text
Model Optimization Platform / ObservationPolicy
    HOW MUCH EVIDENCE IS ENOUGH?

Metrics / Traffic-SLO Analyzer
    WHAT IS HAPPENING?

Model Evaluation / EvaluationController
    IS MODEL QUALITY STILL GOOD ON OBSERVED WORKLOAD?

ServingOptimizationController
    IS THE CURRENT DEPLOYED QUALIFICATION STILL VALID?

ModelAdaptationController
    WHAT KIND OF REMEDIATION IS REQUIRED?

OptimizationController
    HOW DO WE PRODUCE BETTER SERVING REALIZATIONS?

TrainingController
    HOW DO WE PRODUCE A NEW CANONICAL MODEL?

SchedulingController
    WHICH GPU RESOURCES DOES EACH JOB OWN?

ServingDeploymentController
    HOW DO WE ROLL OUT THE RESULT?
```

That separation is the core architecture.

---

# Zoom Level 2 — Subsystem Interview Design

# 31. Complete Cross-Platform Boundary Diagram

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ PRODUCT / APPLICATION TEAM                                                   │
│                                                                              │
│  ServingRequirements                                                         │
│    ├── TTFT / TPOT SLO                                                       │
│    ├── model-quality constraints                                             │
│    ├── cost / availability intent                                            │
│    └── expected_traffic_distribution E1                                     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                │ initial OptimizationJob
                                v
┌──────────────────────────────────────────────────────────────────────────────┐
│ MODEL OPTIMIZATION / ADAPTATION PLATFORM                                     │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ OptimizationController                                                 │  │
│  │                                                                        │  │
│  │ generate -> transform -> benchmark -> qualify -> rank -> publish       │  │
│  └───────────────┬─────────────────────────────────────────┬──────────────┘  │
│                  │                                         │                 │
│                  │ ResourceIntent                          │ P0/P1/P2        │
│                  v                                         v                 │
│          experiment workers                         Bundle Publisher          │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ ObservationPolicy                                                      │  │
│  │ min window / min requests / required dimensions / sample confidence   │  │
│  └─────────────────────────────────────┬──────────────────────────────────┘  │
│                                        │                                     │
│  ┌─────────────────────────────────────┴──────────────────────────────────┐  │
│  │ ModelAdaptationController                                               │  │
│  │                                                                        │  │
│  │ diagnose -> REOPTIMIZE / EVALUATE / TRAIN / BOTH                       │  │
│  └───────────────▲────────────────────────────────────────────────────────┘  │
└──────────────────┼───────────────────────────────────────────────────────────┘
                   │
                   │ ModelAdaptationRequest
                   │
┌──────────────────┼───────────────────────────────────────────────────────────┐
│ ML SERVING PLATFORM                                                         │
│                  │                                                          │
│  ┌───────────────┴───────────────────────────────────────────────────────┐  │
│  │ ServingOptimizationController                                         │  │
│  │                                                                       │  │
│  │ monitor -> wait for observation policy -> validate qualification      │  │
│  │        -> adaptation request -> wait -> request rollout               │  │
│  └──────────────────────────────▲───────────────────────────┬────────────┘  │
│                                 │                           │               │
│                      observed facts                         │ rollout       │
│                                 │                           v               │
│  ┌──────────────────────────────┼────────────────────────────────────────┐ │
│  │ ServingDeploymentController  │                                        │ │
│  │ allocate -> start -> load -> warm -> publish -> drain/replace         │ │
│  └──────────────────────────────┼───────────────────────┬────────────────┘ │
│                                 │                       │                  │
│                                 │ telemetry             │ ResourceIntent   │
│                                 v                       v                  │
│                           Serving Instances                                │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  v
┌──────────────────────────────────────────────────────────────────────────────┐
│ METRICS / OBSERVABILITY PLATFORM                                             │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ Traffic / SLO Analyzer                                                 │  │
│  │                                                                        │  │
│  │ executes ObservationPolicy                                             │  │
│  │ builds observed_traffic_distribution O7                                │  │
│  │ computes TTFT/TPOT headroom, drift, cost/capacity facts                │  │
│  └───────────────────────────────┬────────────────────────────────────────┘  │
│                                  │                                           │
│                     observed traffic + serving facts                         │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
                                   │
                     +-------------+--------------+
                     |                            |
                     v                            v
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│ MODEL EVALUATION PLATFORM            │  │ GPU SCHEDULING PLATFORM           │
│                                      │  │                                   │
│ ┌──────────────────────────────────┐ │  │ ┌───────────────────────────────┐ │
│ │ EvaluationController             │ │  │ │ SchedulingController          │ │
│ │                                  │ │  │ │                               │ │
│ │ build observed eval set          │ │  │ │ ResourceIntent -> Allocation  │ │
│ │ measure quality                  │ │  │ │ physical GPU ownership        │ │
│ │ publish quality evidence         │ │  │ └───────────────────────────────┘ │
│ └───────────────┬──────────────────┘ │  └───────────────────────────────────┘
└─────────────────┼────────────────────┘
                  │ quality invalid?
                  v
┌──────────────────────────────────────────────────────────────────────────────┐
│ TRAINING PLATFORM                                                            │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ TrainingController                                                     │  │
│  │                                                                        │  │
│  │ prepare -> train/fine-tune -> validate -> export CanonicalModel        │  │
│  └──────────────────────────────────────┬─────────────────────────────────┘  │
└─────────────────────────────────────────┼────────────────────────────────────┘
                                          │ new CanonicalModel
                                          v
                              MODEL OPTIMIZATION PLATFORM
                                          │
                                          v
                                  new ServingProfiles
                                          │
                                          v
┌──────────────────────────────────────────────────────────────────────────────┐
│ MODEL REGISTRY / BUNDLE PUBLICATION                                          │
│                                                                              │
│  ModelBundleVersion                                                          │
│    ├── CanonicalModel                                                        │
│    ├── primary ServingProfile                                                │
│    ├── alternative ServingProfiles                                           │
│    └── qualification + traffic + observation-policy references               │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          v
                               ML SERVING PLATFORM rollout
```

---

# 32. Who Talks to Whom

The interview-day contract summary:

```text
Product Team
    ->
OptimizationController
    ServingRequirements + expected_traffic_distribution

OptimizationController
    ->
SchedulingController
    ResourceIntent for optimization experiments

OptimizationController
    ->
Model Registry / Bundle Publisher
    selected ServingProfiles + QualificationEvidence + ObservationPolicy ref

Model Optimization Platform
    ->
Metrics Platform
    ObservationPolicy

Serving Instances / Router
    ->
Metrics Platform
    production telemetry

Metrics Traffic/SLO Analyzer
    ->
ServingOptimizationController
    observed traffic + serving facts + ObservationPolicy satisfaction

ServingOptimizationController
    ->
ModelAdaptationController
    ModelAdaptationRequest

ModelAdaptationController
    ->
OptimizationController
    OptimizationJob
    [when serving realization is the problem]

ModelAdaptationController
    ->
EvaluationController
    EvaluationJob
    [when model quality is uncertain/suspect]

EvaluationController
    ->
ModelAdaptationController
    ObservedQualityEvidence

ModelAdaptationController
    ->
TrainingController
    TrainingJob
    [when canonical model quality is invalidated]

TrainingController
    ->
OptimizationController
    new CanonicalModel requiring new ServingProfile qualification

Optimization / Evaluation / Training Controllers
    ->
SchedulingController
    ResourceIntent for temporary GPU workloads

Model Registry
    ->
ServingOptimizationController
    new immutable ModelBundleVersion / resolution available

ServingOptimizationController
    ->
ServingDeploymentController
    rollout request

ServingDeploymentController
    ->
SchedulingController
    production ServingInstance ResourceIntent
```

---

# 33. Important Non-Calls

Avoid these couplings:

```text
Metrics Platform
    X-> directly create TrainingJob

Metrics Platform
    X-> directly trigger OptimizationJob

ServingOptimizationController
    X-> decide fine-tuning strategy

ServingOptimizationController
    X-> perform quality evaluation itself

OptimizationController
    X-> own TrainingJob lifecycle

TrainingController
    X-> directly replace production replicas

SchedulingController
    X-> decide model/profile validity

ServingDeploymentController
    X-> calculate traffic-distribution statistics

EvaluationController
    X-> choose production rollout
```

Each would cross a durable domain-controller boundary.

---

# 34. State-Machine Interaction — Serving Problem

```text
ServingOptimizationController       ModelAdaptationController       OptimizationController
-----------------------------       -------------------------       ----------------------

MONITORING
    |
    v
COLLECTING_OBSERVATION
    |
    | ObservationPolicy satisfied
    v
EVALUATING_QUALIFICATION
    |
    | serving profile invalid
    v
ADAPTATION_REQUESTED
    |
    | ModelAdaptationRequest ------> PENDING
    |                                   |
    v                                   v
WAITING_FOR_RESOLUTION             DIAGNOSING
                                        |
                                        | SERVING_REOPTIMIZE
                                        v
                                   REOPTIMIZE
                                        |
                                        | OptimizationJob ----------> PENDING
                                        |                                |
                                        |                                v
                                        |                        GENERATING_CANDIDATES
                                        |                                |
                                        |                                v
                                        |                        RUNNING_EXPERIMENTS
                                        |                                |
                                        |                                v
                                        |                            QUALIFYING
                                        |                                |
                                        |                                v
                                        |                             RANKING
                                        |                                |
                                        |                                v
                                        |                           PUBLISHING
                                        |                                |
                                        |                                v
                                        |                            COMPLETED
                                        |                                |
                                        <------- new profile/bundle -----+
                                        |
                                   COMPLETE
                                        |
    <---------- resolution -------------+
    |
    v
RESOLUTION_AVAILABLE
    |
    v
ROLLOUT_REQUESTED
    |
    v
MONITORING
```

---

# 35. State-Machine Interaction — Model Quality Problem

```text
ServingOptimization   ModelAdaptation      Evaluation         Training        Optimization
Controller             Controller           Controller         Controller      Controller
-------------------   ------------------   ----------------   -------------   ------------

EVALUATING
    |
    | qualification invalid
    v
ADAPTATION_REQUESTED
    |
    | AdaptationRequest ---> PENDING
    |                           |
    v                           v
WAITING                    DIAGNOSING
                                |
                                | quality uncertain
                                v
                            EVALUATE_MODEL
                                |
                                | EvaluationJob ----> PENDING
                                |                       |
                                |                       v
                                |                 PREPARING_EVAL_SET
                                |                       |
                                |                       v
                                |                 RUNNING_EVALUATION
                                |                       |
                                |                       v
                                |                    COMPLETED
                                |                       |
                                <----- quality bad -----+
                                |
                                v
                            TRAIN_REQUIRED
                                |
                                | TrainingJob ----------------> PENDING
                                |                                  |
                                |                                  v
                                |                              TRAINING
                                |                                  |
                                |                                  v
                                |                        EXPORT_CANONICAL_MODEL
                                |                                  |
                                <------ CanonicalModel M43 --------+
                                |
                                | OptimizationJob --------------------------------> PENDING
                                |                                                      |
                                |                                                      v
                                |                                             RUNNING_EXPERIMENTS
                                |                                                      |
                                |                                                      v
                                |                                                  PUBLISHING
                                |                                                      |
                                <------------ new bundle/profile ----------------------+
                                |
                            COMPLETE
                                |
    <------- resolution -------+
    |
    v
ROLLOUT_REQUESTED
```

This is the critical interview distinction:

```text
ServingOptimizationController
    detects invalid qualification

ModelAdaptationController
    chooses remediation

EvaluationController
    proves/denies model-quality regression

TrainingController
    changes CanonicalModel

OptimizationController
    changes ServingProfile realization
```

---

# 36. Zoom Level 2 Concrete Scenario A — Serving Reoptimization

Initial assumption:

```text
Expected E1:
    prompt p95 = 8K
    concurrency p99 = 64
```

Initial profile:

```text
P0:
    H200 / vLLM / FP8 / TP4
    max_num_batched_tokens = 8192

Qualification:
    traffic = EXPECTED E1
    TTFT p99 = 285 ms
    TPOT p99 = 31 ms

ObservationPolicy OP7:
    minimum 6h
    minimum 100K requests
```

Production after OP7 is satisfied:

```text
Observed O7:
    prompt p95 = 24K
    concurrency p99 = 140

Current:
    TTFT p99 = 475 ms
    TTFT SLO = 500 ms

quality:
    still healthy
```

Diagnosis:

```text
ServingProfile qualification invalidated
CanonicalModel quality still valid
```

Decision:

```text
SERVING_REOPTIMIZE
```

Possible new result:

```text
P3:
    H200 / vLLM / FP8 / TP8
    max_num_batched_tokens = 32768
```

No training required.

---

# 37. Zoom Level 2 Concrete Scenario B — Retraining

Initial workload:

```text
general English chat
light tool use
```

Observed production workload:

```text
large increase in:
    coding requests
    complex tool orchestration
    Japanese-language requests
```

Serving metrics:

```text
TTFT:
    healthy

TPOT:
    healthy

GPU efficiency:
    healthy
```

Observed quality:

```text
tool-call success:
    below product threshold

Japanese task success:
    below product threshold
```

Diagnosis:

```text
ServingProfile is healthy
CanonicalModel quality envelope is invalid
```

Decision:

```text
EVALUATE_MODEL
    ->
TRAIN / FINE-TUNE
    ->
new CanonicalModel
    ->
Model Optimization
    ->
new ServingProfiles
```

No amount of TP/CUDA Graph/runtime tuning fixes the semantic quality problem.

---

# 38. Zoom Level 2 Interview Explanation

> **“I treat production traffic feedback as a model-adaptation control loop, not just a serving-performance loop. Product supplies expected traffic and SLO/quality intent. Model Optimization qualifies the initial profiles and also defines a versioned ObservationPolicy—minimum time window, request count, required dimensions, and sampling confidence—because it knows what evidence is required to reassess its own qualification. Metrics executes that policy and produces observed traffic and serving facts. Once enough evidence exists, the ServingOptimizationController decides only whether the currently deployed model/profile qualification is still valid. If invalid, it creates a durable ModelAdaptationRequest. A separate ModelAdaptationController diagnoses whether the issue is the serving realization, the canonical model quality, or both. Serving problems create an OptimizationJob; quality uncertainty creates an EvaluationJob; confirmed model-quality regression can create a TrainingJob. A new canonical model always flows through Model Optimization again before deployment. GPU Scheduling allocates resources for all these jobs, while ServingDeploymentController owns the final rollout.”**

---

# Zoom Level 3 — 30–60 Second Full-System Box

# 39. One-Box Model

```text
Expected Traffic
    |
    v
OPTIMIZE + DEFINE OBSERVATION POLICY
    |
    v
DEPLOY
    |
    v
OBSERVE REAL TRAFFIC / SLO / QUALITY
    |
    v
VALIDATE QUALIFICATION
    |
    +-- valid ----------------------------> continue
    |
    +-- invalid
            |
            v
         DIAGNOSE
        /        \
       v          v
SERVING ISSUE   MODEL ISSUE
    |               |
    v               v
REOPTIMIZE       EVALUATE / TRAIN
    |               |
    +-------+-------+
            |
            v
       OPTIMIZE RESULT
            |
            v
          ROLLOUT
```

---

# 40. Six Concepts to Remember

```text
EXPECT
    Product supplies expected traffic.

OBSERVE POLICY
    Model Optimization defines how much production evidence is enough.

OBSERVE
    Metrics executes the policy and derives real traffic/SLO facts.

VALIDATE
    ServingOptimizationController determines whether deployed qualification remains valid.

DIAGNOSE
    ModelAdaptationController selects reoptimization, evaluation/training, or both.

ADAPT
    Optimization / Evaluation / Training controllers execute their own lifecycles.
```

---

# 41. Final Ownership Invariants

1. **Expected traffic is Product-supplied intent.**

2. **Observed traffic is telemetry-derived production reality.**

3. **Initial Model Optimization only knows expected traffic.**

4. **Model Optimization Platform owns the versioned `ObservationPolicy` that specifies when enough production evidence exists to reassess a qualification.**

5. **ObservationPolicy may include minimum observation time, minimum request count, required traffic dimensions, minimum samples per important segment, and confidence/snapshot requirements.**

6. **Metrics / Observability executes ObservationPolicy; it does not invent the policy or choose remediation.**

7. **Every published ServingProfile records the exact expected or observed traffic snapshot used for its qualification and references the applicable ObservationPolicy.**

8. **Live observed traffic is not mutable Model Bundle state.**

9. **Canary is the first observation point, not the only observation point.**

10. **Production traffic and production quality must continue to be observed for the deployment lifecycle.**

11. **Raw traffic drift alone does not automatically trigger optimization or training.**

12. **Traffic shift can invalidate serving-performance assumptions, canonical-model quality assumptions, or both.**

13. **ServingPerformance invalidation is based on profile qualification envelope, SLO headroom, stability, and/or economics—not distribution divergence alone.**

14. **Model-quality invalidation requires empirical quality evidence; traffic drift alone does not prove semantic quality regression.**

15. **Model Evaluation Platform owns quality measurement against representative observed workload when additional evidence is required.**

16. **ServingOptimizationController owns the deployed qualification lifecycle and decides whether the current qualification is valid or invalid.**

17. **ServingOptimizationController does not decide whether to retrain, quantize, change TP, or change runtime.**

18. **An invalid qualification produces a durable `ModelAdaptationRequest`.**

19. **ModelAdaptationController owns remediation diagnosis: no change, ServingProfile reoptimization, model evaluation/retraining, or both.**

20. **OptimizationController owns serving-realization search and publishes qualified ServingProfiles.**

21. **TrainingController owns retraining/fine-tuning and production of a new CanonicalModel.**

22. **Any new CanonicalModel must go through Model Optimization before it becomes a production ServingProfile.**

23. **GPU Scheduling owns physical resource allocation for Serving, Optimization, Evaluation, and Training jobs; it does not decide model/profile validity.**

24. **ServingDeploymentController owns rollout and rollback of the resulting ModelBundle/ServingProfile.**

25. **Traffic-driven ServingProfile reoptimization can produce a new ModelBundleVersion without changing the CanonicalModel.**

26. **Model-quality adaptation can produce a new CanonicalModel and therefore a new ModelBundleVersion plus newly optimized ServingProfiles.**

27. **Cooldown, evidence thresholds, deduplication, and one-active-adaptation guards prevent control-loop oscillation.**

28. **Cross-platform coordination uses durable contracts—`ModelAdaptationRequest`, `OptimizationJob`, `EvaluationJob`, `TrainingJob`, `ResourceIntent`, and `ModelBundleVersion`—not private controller implementation calls.**

---

# 42. Principal-Level Summary

> **Traffic-aware serving is a closed-loop model-adaptation system spanning multiple ML platforms. Product supplies expected traffic plus SLO and quality intent. Model Optimization qualifies the initial ServingProfiles and defines the ObservationPolicy that specifies how much production evidence is required to reassess that qualification. Metrics executes the observation policy and derives observed traffic and serving facts; Model Evaluation supplies quality evidence when needed. The ServingOptimizationController owns only the deployed qualification lifecycle. If qualification becomes invalid, it emits a durable ModelAdaptationRequest. A separate ModelAdaptationController determines whether the issue is the serving realization, the canonical model, or both, and creates OptimizationJob, EvaluationJob, or TrainingJob accordingly. Training may produce a new CanonicalModel, which must then pass through Model Optimization again. GPU Scheduling owns resource allocation for each workload, and ServingDeploymentController owns the final rollout.**
