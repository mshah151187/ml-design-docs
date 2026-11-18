# Horizontal Concerns in the Model Optimization Platform

> **Status:** Working architecture baseline  
> **Purpose:** Define the shared horizontal capabilities used by all objective-specific search strategies.

---

# 1. Core Principle

Optimization Objectives own **what experiment should be tried next**.

The horizontal platform owns **how experiments are made possible, executed, measured, recovered, and qualified**.

```text
ObjectiveSearchStrategy
        |
        | CandidateTrialRequest
        v
+------------------------------------------------------------+
|                 Horizontal Platform Areas                  |
|                                                            |
|  1. Input / Context Resolution                             |
|  2. Feasibility & Constraint Evaluation                    |
|  3. Artifact Materialization                               |
|  4. Resource & Environment Provisioning                    |
|  5. Workload Generation / Replay                           |
|  6. Experiment Execution                                   |
|  7. Measurement & Evidence Collection                      |
|  8. Quality Evaluation                                     |
|  9. Multi-Fidelity Trial Management                        |
| 10. Experiment History / Knowledge Store                   |
| 11. Reliability / Idempotency / Recovery                   |
| 12. Qualification & Publication                            |
+------------------------------------------------------------+
        |
        | ExperimentResult
        v
ObjectiveSearchStrategy
```

These concerns are horizontal because their core semantics do not fundamentally change when the optimization objective changes.

---

# 2. Input / Context Resolution

Every objective needs the same canonical, immutable inputs:

```text
CanonicalModel / ModelBundle
ServingRequirements
ExpectedTrafficDistribution
OptimizationPolicy
OptimizationObjective
Priority
HardwareCapabilityCatalog
RuntimeCapabilityCatalog
TransformationCapabilityCatalog
```

The horizontal layer resolves immutable versions, registry references, schema versions, checksums, capability-catalog versions, and experiment provenance.

The objective strategy may interpret this context differently, but it should not independently implement registry resolution or version pinning.

---

# 3. Feasibility & Constraint Evaluation

This area answers:

> Is this candidate legal and possible before we spend expensive accelerator resources on it?

Examples:

```text
Do model weights fit in HBM?
Is the requested TP degree valid for the model/runtime?
Does the runtime support this model architecture?
Does the hardware support the requested precision?
Is the requested context length supported?
Are required functional features available?
Is this artifact/runtime/topology combination valid?
```

Feasibility is objective-independent.

There are two forms of constraint evaluation:

```text
Static Feasibility
    evaluated before an experiment

Empirical Constraint Validation
    evaluated from experiment evidence
```

Examples:

```text
Static:
    model fit
    topology legality
    runtime compatibility

Empirical:
    p99 TTFT <= requirement
    error rate <= threshold
    quality regression <= threshold
```

---

# 4. Artifact Materialization

Objective strategies may request serving realizations such as:

```text
BF16 model
FP8 model
INT4 model
TensorRT-LLM engine
runtime-specific packaged artifacts
```

The objective strategy decides **which realization is worth testing**.

The horizontal artifact system owns:

```text
quantization execution
format conversion
compilation
engine building
artifact caching
checksums
lineage
immutability
reuse
```

A key invariant is reuse.

Example:

```text
Candidate 17:
    FP8 + vLLM + TP4
    max_num_seqs = 64

Candidate 18:
    FP8 + vLLM + TP4
    max_num_seqs = 128
```

These candidates should reuse the same FP8 artifact.

---

# 5. Resource & Environment Provisioning

Every experiment requires a reproducible execution environment.

Important dimensions include:

```text
GPU type
GPU count
node topology
interconnect
CUDA version
driver version
runtime container
runtime version
CPU / memory allocation
```

The horizontal platform owns the interaction with the external scheduling layer:

```text
CandidateTrialRequest
        |
        v
ResourceIntent
        |
        v
GPU Scheduling Platform
        |
        v
ResourceAllocation
```

The exact allocated environment must be recorded as experiment provenance.

---

# 6. Workload Generation / Replay

Every objective needs a consistent way to evaluate candidates against the expected workload.

The workload layer derives experiments from:

```text
ExpectedTrafficDistribution
```

and may support:

```text
segment sampling
input-token distributions
output-token distributions
arrival-rate generation
client concurrency
streaming behavior
prefix-reuse behavior
representative trace replay
```

The objective strategy may decide which workload regions deserve more experiment budget.

The horizontal workload system owns how those workloads are generated and reproduced.

---

# 7. Experiment Execution

The Experiment Execution layer takes a complete candidate request and runs it.

```text
CandidateTrialRequest
        |
        v
prepare artifact
        |
allocate resources
        |
start runtime
        |
load model
        |
warm up
        |
execute benchmark
        |
collect outputs and metrics
        |
cleanup
        |
        v
ExperimentResult
```

The shared contract is:

```text
CandidateTrialRequest
    ->
ExperimentResult
```

The objective does not need to know whether execution internally uses Kubernetes Jobs, Ray, Flyte, or custom workers.

---

# 8. Measurement & Evidence Collection

Each experiment must produce a standardized evidence record.

## Performance evidence

```text
TTFT p50 / p95 / p99
TPOT p50 / p95 / p99
input tokens / second
output tokens / second
request throughput
queueing latency
end-to-end latency
```

## Resource evidence

```text
GPU-seconds
peak HBM
KV-cache usage
GPU utilization
CPU utilization
communication statistics
```

## Economic evidence

```text
hardware cost
trial cost
cost per input token
cost per output token
normalized cost per token
```

## Stability evidence

```text
OOM events
timeouts
runtime failures
error rate
restarts
```

## Provenance

```text
artifact checksum
candidate configuration
runtime version
CUDA version
driver version
hardware type
workload snapshot
benchmark version
```

The same evidence model should be consumable by every objective strategy.

---

# 9. Quality Evaluation

Quality remains a horizontal concern even though quality optimization is not part of the first supported objective set.

Model transformations such as quantization, lower precision, compiler transformations, or speculative decoding may change outputs.

The common quality-evaluation system uses:

```text
evaluation_suite_ref
baseline_result_ref
candidate outputs
```

to produce:

```text
quality metrics
quality deltas
quality-gate pass/fail evidence
```

Quality is therefore a shared acceptance constraint.

---

# 10. Multi-Fidelity Trial Management

Not every candidate should receive a full qualification run.

The horizontal platform supports multiple experiment fidelity levels:

```text
F0
    static feasibility checks

F1
    startup / OOM / smoke test

F2
    short performance benchmark

F3
    representative workload benchmark

F4
    full qualification
```

Ownership boundary:

```text
Horizontal Platform:
    knows HOW to run each fidelity level

Objective Strategy:
    decides WHICH candidate should be promoted
```

---

# 11. Experiment History / Knowledge Store

All objective strategies reuse the same durable experiment evidence.

Historical records should retain:

```text
model features
workload features
candidate configuration
hardware
runtime
artifact provenance
performance metrics
cost metrics
quality results
failure reasons
qualification outcome
```

The Experiment History Store is horizontal.

Historical seed selection and ranking are objective-specific.

---

# 12. Reliability / Idempotency / Recovery

Experiment campaigns are expensive, distributed, and long-running.

The horizontal platform must provide:

```text
idempotent trial creation
deduplication
retry policy
lease / ownership
heartbeats
timeouts
cancellation
cleanup
durable experiment state
controller recovery
```

Objective strategies should not independently implement distributed-systems correctness.

---

# 13. Qualification & Publication

The Objective Search Strategy produces finalists.

Qualification proves that those candidates satisfy the full serving contract.

Typical final qualification includes:

```text
full workload replay
longer stability window
hard latency validation
quality gates
context-length extremes
capacity validation
error-rate thresholds
failure-behavior validation
```

Outputs include:

```text
QualificationEvidence
ServingProfile
```

Qualification is horizontal because the definition of "usable" comes primarily from `ServingRequirements`, not from whichever objective found the candidate.

---

# 14. Four Horizontal Subsystems

For architecture presentation, the twelve concerns can be grouped into four major subsystems.

```text
+---------------------------------------------------------+
|  A. SEARCH FOUNDATION                                   |
|                                                         |
|  Input / Context Resolution                             |
|  Feasibility & Constraint Evaluation                    |
|  Experiment History / Knowledge Store                   |
+---------------------------------------------------------+

+---------------------------------------------------------+
|  B. EXPERIMENT MATERIALIZATION & EXECUTION              |
|                                                         |
|  Artifact Materialization                               |
|  Resource & Environment Provisioning                    |
|  Experiment Execution                                   |
+---------------------------------------------------------+

+---------------------------------------------------------+
|  C. EXPERIMENT EVIDENCE                                 |
|                                                         |
|  Workload Generation / Replay                           |
|  Measurement & Evidence Collection                      |
|  Quality Evaluation                                     |
+---------------------------------------------------------+

+---------------------------------------------------------+
|  D. EXPERIMENT LIFECYCLE                                |
|                                                         |
|  Multi-Fidelity Trial Management                        |
|  Reliability / Idempotency / Recovery                   |
|  Qualification & Publication                            |
+---------------------------------------------------------+
```

---

# 15. Relationship to Objective-Specific Search

```text
                         ObjectiveSearchStrategy
                                  |
                        "what should I try next?"
                                  |
               +------------------+------------------+
               |                  |                  |
               v                  v                  v
        Search Foundation    Experiment Exec    Evidence / Lifecycle
               |                  |                  |
               +------------------+------------------+
                                  |
                           ExperimentResult
                                  |
                                  v
                         ObjectiveSearchStrategy
```

The strategy chooses experiments.

The horizontal systems make those experiments reproducible, safe, measurable, and qualifiable.

---

# 16. Most Important Boundary Contracts

The most important contracts between objective-specific search and the horizontal platform are:

```text
CandidateTrialRequest

ExperimentResult
```

These contracts will be defined separately in detail.

Conceptually:

```text
ObjectiveSearchStrategy
        |
        | CandidateTrialRequest
        v
Horizontal Experiment Platform
        |
        | ExperimentResult
        v
ObjectiveSearchStrategy
```

---

# 17. Locked Principle

> **Objective-specific strategies own experiment selection. Horizontal concerns own experiment feasibility, materialization, execution, evidence, lifecycle, reliability, and qualification.**

This is the baseline horizontal architecture for the Model Optimization Platform.
