# Optimization Policy and Optimization Objectives

> **Status:** Locked interview vocabulary for the first Model Optimization Platform release  
> **Purpose:** Define the client-facing optimization policies and the first-class optimization objectives whose end-to-end search processes we are prepared to defend in detail.

---

# 1. Design Principle

The first release intentionally supports a **small number of optimization policies and objectives deeply**.

We do not try to expose every possible optimization goal.

> We started with a small set of production objectives whose search semantics we understood end to end. Each objective is a first-class strategy in the platform. Shared infrastructure handles feasibility, experiments, metrics, retries, and multi-fidelity execution; the objective owns the actual search logic.

This avoids building one overly generic search process that becomes difficult to reason about.

---

# 2. First-Class Concepts

## Optimization Policy

`OptimizationPolicy` is the **client-supplied high-level preference** for what kind of qualified serving realization the client wants.

It answers:

> Among all realizations that satisfy my hard `ServingRequirements`, what type of solution do I prefer?

Example:

```text
ServingRequirements:
    max context >= 128K
    quality regression <= 0.5%
    p99 TTFT <= 800 ms
    p99 TPOT <= 40 ms

OptimizationPolicy:
    COST_OPTIMIZED
```

The policy is supplied when the AI/product team creates an `OptimizationJob`.

```python
optimize_model(
    model_registry_path="registry://feed-ai/feed-gen/versions/42",
    priority=HIGH,
    optimization_policy=COST_OPTIMIZED,
)
```

The client supplies the intent. The Model Optimization Platform owns the implementation.

Clients do not specify low-level optimizer mechanics such as:

```text
TPE hyperparameters
exact trial counts
exact TP candidate sets
runtime knob ranges
quantization backend internals
search-model parameters
```

---

## Optimization Objective

`OptimizationObjective` is a **first-class platform entity** that owns the search semantics for one measurable production goal.

It answers:

> Which outcome are we trying to improve, and what search process should be used to find better realizations for that outcome?

An objective is not only a scalar metric used at the end.

It owns objective-specific logic such as:

```text
which optimization dimensions are active
which analytical seeds are generated
which historical experiments are useful
which runtime knobs are important
which ranges / priors are used
how trial results are scored
which candidates are promoted to higher fidelity
when the search has converged
```

Conceptually:

```text
OptimizationPolicy
        |
        v
OptimizationObjective
        |
        v
ObjectiveSearchStrategy
```

---

# 3. Shared Platform vs Objective-Specific Search

The platform keeps horizontal experiment machinery common:

```text
resolve immutable inputs
        |
hard feasibility pruning
        |
artifact materialization
        |
GPU/resource acquisition
        |
experiment execution
        |
metrics collection
        |
quality evaluation
        |
retry / cleanup
        |
multi-fidelity execution
        |
qualification
        |
publication
```

But the actual search process belongs to the objective:

```text
active dimensions
candidate-generation heuristics
historical seed ranking
search priors
runtime-knob ranges
candidate scoring
promotion decisions
convergence criteria
```

Therefore:

```text
same experiment infrastructure
different objective-specific search strategies
```

---

# 4. Supported Optimization Policies

The first supported policy vocabulary is deliberately limited to four policies:

```text
COST_OPTIMIZED
INTERACTIVE_LATENCY_OPTIMIZED
THROUGHPUT_OPTIMIZED
CAPACITY_OPTIMIZED
```

Policies such as quality-optimized or general Pareto-balanced optimization are intentionally outside the first interview scope until their complete end-to-end search processes are designed.

---

## `COST_OPTIMIZED`

### What the client means

> Optimize the model artifact and serving environment so that production inference costs as little as possible while still satisfying every hard serving and quality requirement.

### Primary objective

```text
MINIMIZE_COST_PER_TOKEN
```

### Why

The direct product intent is economic efficiency.

The search minimizes normalized infrastructure cost per unit of useful inference work, not simply GPU count or hourly hardware price.

A more expensive GPU or larger topology may still be cheaper per token if it produces sufficiently more useful work.

### Example

```text
Candidate A:
    4 GPUs
    1800 output tok/s
    normalized cost/token = 1.00

Candidate B:
    8 GPUs
    5000 output tok/s
    normalized cost/token = 0.72
```

Candidate B uses more GPUs but is preferable because its normalized cost is lower.

---

## `INTERACTIVE_LATENCY_OPTIMIZED`

### What the client means

> Make interactive generation feel as responsive as possible while still satisfying quality, capacity, and all other hard serving requirements.

### Primary objective

```text
MINIMIZE_TTFT
```

### Why

For streaming interactive generation, TTFT is usually the first latency experienced by the user.

It captures:

```text
queueing
prefill execution
scheduler delay
initial execution overhead
```

The first release therefore treats TTFT as the dominant interactive-latency objective.

### Secondary measured signals

```text
TPOT
end-to-end latency
cost/token
```

These remain important qualification and comparison metrics, but the objective-specific search strategy is centered on minimizing TTFT.

### Example

```text
Candidate A:
    TTFT = 430 ms
    TPOT = 34 ms

Candidate B:
    TTFT = 620 ms
    TPOT = 28 ms
```

If both satisfy all hard requirements, the policy prefers Candidate A because first-token responsiveness is the primary goal.

---

## `THROUGHPUT_OPTIMIZED`

### What the client means

> Maximize the amount of useful inference work completed per unit time.

This policy resolves to one of two concrete objectives because prefill and decode are different computational regimes.

### Objective A

```text
MAXIMIZE_INPUT_TOKEN_THROUGHPUT
```

Use when the workload is primarily:

```text
prefill-heavy
large-prompt
offline input-processing dominated
```

### Why

Input-token throughput measures how much prompt/prefill work the system can complete per second.

### Example

```text
Feed offline workload:
    very large input datasets
    relatively short generated outputs

Optimization objective:
    MAXIMIZE_INPUT_TOKEN_THROUGHPUT
```

### Objective B

```text
MAXIMIZE_OUTPUT_TOKEN_THROUGHPUT
```

Use when the workload is primarily:

```text
decode-heavy
generation-heavy
high-concurrency generation
long-output workloads
```

### Why

Output-token throughput measures how many generated tokens the serving realization can sustain per second.

### Example

```text
Generation workload:
    moderate prompt sizes
    large generated outputs

Optimization objective:
    MAXIMIZE_OUTPUT_TOKEN_THROUGHPUT
```

### Why throughput is split

LLM inference has two distinct phases:

```text
Prefill
    -> input-token processing
    -> commonly compute-heavy

Decode
    -> output-token generation
    -> commonly memory-bandwidth / KV sensitive
```

A configuration can improve one while hurting the other.

Therefore the platform does not use one ambiguous `MAXIMIZE_THROUGHPUT` objective.

---

## `CAPACITY_OPTIMIZED`

### What the client means

> Maximize how much useful concurrent serving demand the available GPU memory/resources can hold and sustain.

### Primary objective

```text
MAXIMIZE_CONCURRENT_SEQUENCE_CAPACITY
```

### Why

The product-visible capacity question is:

> How many useful active/resident sequences can this serving realization sustain under the target context/workload envelope?

This is more useful than exposing internal proxies such as free HBM or KV-cache occupancy.

### Example

```text
Candidate A:
    24 active long-context sequences / replica

Candidate B:
    56 active long-context sequences / replica
```

If both satisfy latency and throughput requirements, the capacity objective prefers Candidate B.

---

# 5. Final Supported Optimization Objectives

The first release supports exactly these five first-class objectives:

```text
MINIMIZE_COST_PER_TOKEN
MINIMIZE_TTFT
MAXIMIZE_INPUT_TOKEN_THROUGHPUT
MAXIMIZE_OUTPUT_TOKEN_THROUGHPUT
MAXIMIZE_CONCURRENT_SEQUENCE_CAPACITY
```

We intentionally do not claim support for additional objectives until their complete search processes are designed and validated.

---

# 6. Objective Definitions and Search Ownership

## `MINIMIZE_COST_PER_TOKEN`

### Meaning

Minimize normalized infrastructure cost required to process the expected production workload.

Conceptually:

```text
cost_per_token
    =
resource_cost_over_measurement_window
/
useful_tokens_processed
```

### Objective-owned search concerns

```text
quantization economics
hardware economics
minimum useful parallelism
runtime efficiency
batching efficiency
prefix-reuse opportunities
useful throughput per GPU-dollar
```

---

## `MINIMIZE_TTFT`

### Meaning

Minimize Time To First Token for streaming generation.

### Objective-owned search concerns

```text
prefill critical path
queueing delay
parallelism
chunked prefill
runtime scheduler behavior
CUDA graph / launch overhead
batching limits
```

---

## `MAXIMIZE_INPUT_TOKEN_THROUGHPUT`

### Meaning

Maximize useful input/prompt tokens processed per second.

### Objective-owned search concerns

```text
large-token batching
prefill parallelism
runtime/kernel efficiency
quantization
chunked prefill behavior
GPU compute efficiency
```

---

## `MAXIMIZE_OUTPUT_TOKEN_THROUGHPUT`

### Meaning

Maximize useful generated tokens produced per second.

### Objective-owned search concerns

```text
continuous batching
decode concurrency
KV-cache efficiency
memory bandwidth
parallelism
speculative decoding where supported
runtime scheduling
```

---

## `MAXIMIZE_CONCURRENT_SEQUENCE_CAPACITY`

### Meaning

Maximize the number of useful active/resident sequences one serving realization can support under the required workload envelope.

### Objective-owned search concerns

```text
weight footprint
KV-cache footprint
KV precision
parallelism
memory allocation
paged KV behavior
context distribution
resident-batch limits
```

---

# 7. Policy-to-Objective Mapping

| Optimization Policy | Client intent | Primary Optimization Objective |
|---|---|---|
| `COST_OPTIMIZED` | Reduce normalized production inference cost while satisfying all hard constraints. | `MINIMIZE_COST_PER_TOKEN` |
| `INTERACTIVE_LATENCY_OPTIMIZED` | Make interactive generation respond as quickly as possible. | `MINIMIZE_TTFT` |
| `THROUGHPUT_OPTIMIZED` | Complete the maximum useful inference work per second. | `MAXIMIZE_INPUT_TOKEN_THROUGHPUT` **or** `MAXIMIZE_OUTPUT_TOKEN_THROUGHPUT`, selected from workload characteristics |
| `CAPACITY_OPTIMIZED` | Maximize useful concurrent serving capacity within the resource envelope. | `MAXIMIZE_CONCURRENT_SEQUENCE_CAPACITY` |

---

# 8. How `THROUGHPUT_OPTIMIZED` Chooses Its Objective

`THROUGHPUT_OPTIMIZED` is the only first-release policy that resolves to one of two first-class objectives.

```text
ExpectedTrafficDistribution
        |
        +--> input-token / prefill dominated
        |        |
        |        v
        |   MAXIMIZE_INPUT_TOKEN_THROUGHPUT
        |
        +--> output-token / decode dominated
                 |
                 v
            MAXIMIZE_OUTPUT_TOKEN_THROUGHPUT
```

The exact workload classifier is defined as part of the throughput objective search design.

---

# 9. Objective as a Strategy

Each objective owns its own search implementation.

Conceptually:

```text
OptimizationObjectiveDefinition {

    activate_dimensions(context)

    generate_analytical_seeds(context)

    retrieve_historical_seeds(context, history)

    build_objective_search_space(context)

    score_experiment_result(result)

    propose_next_candidates(search_state)

    decide_trial_promotion(result, fidelity)

    check_convergence(search_state)
}
```

Concrete strategies:

```text
CostPerTokenObjective
TTFTObjective
InputTokenThroughputObjective
OutputTokenThroughputObjective
ConcurrentSequenceCapacityObjective
```

---

# 10. Common Search Lifecycle

The platform preserves a common high-level lifecycle:

```text
1. Resolve OptimizationObjective

2. Build objective-specific search plan

3. Apply shared hard-feasibility pruning

4. Generate initial candidates
       analytical seeds
       historical warm-starts

5. Execute trials

6. Feed measurements back to objective strategy

7. Objective proposes next candidates

8. Apply multi-fidelity promotion / elimination

9. Stop on objective-specific convergence

10. Run final qualification
```

The principle is:

```text
same lifecycle
different objective-specific decision logic
```

---

# 11. Shared Horizontal Mechanisms

These remain common across objectives:

```text
hard feasibility rules
artifact materialization
GPU/resource allocation
experiment execution
metrics collection
quality evaluation
failure handling
multi-fidelity trial infrastructure
historical experiment store
final qualification
profile publication
```

---

# 12. Metrics That Are Evidence, Not Objectives

These are useful measurements but are not first-class client-facing objectives:

```text
GPU utilization
SM occupancy
HBM utilization
free memory
KV-cache utilization
kernel occupancy
NVLink / PCIe utilization
CPU utilization
```

They explain why a candidate performs well or poorly and help objective-specific strategies select the next experiment.

---

# 13. Priority Is Separate

```text
priority
    = WHEN this OptimizationJob receives scarce platform resources

optimization_policy
    = WHAT kind of qualified solution the client wants

optimization_objective
    = WHICH measurable outcome/search strategy implements that policy
```

Example:

```text
priority = HIGH
optimization_policy = COST_OPTIMIZED
optimization_objective = MINIMIZE_COST_PER_TOKEN
```

---

# 14. Locked Interview Scope

The first interview scope is intentionally:

```text
4 client-facing policies

5 concrete objectives

5 objective-specific search processes
```

We add another objective only after we can explain its search process end to end:

```text
active dimensions
analytical priors
historical priors
experiment design
runtime tuning
feedback algorithm
multi-fidelity promotion
convergence
qualification
```

This is a deliberate platform decision, not a limitation hidden behind a generic framework.
