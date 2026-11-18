# Serving Requirements vs Optimization Policy

> **Purpose:** Clarify the boundary between hard serving constraints, optimization preferences, and the internal search strategy used by the Model Optimization Platform.

---

## 1. Core Mental Model

The Model Optimization Platform solves two different problems:

```text
1. Is this serving realization acceptable?

2. Among acceptable realizations, which ones are preferable?
```

These map to two different domain concepts:

```text
ServingRequirements
    =
hard acceptance constraints

OptimizationPolicy
    =
how the platform prefers and searches among
acceptable serving realizations
```

A useful summary is:

```text
ServingRequirements
    answers:
        CAN this candidate be used?

OptimizationPolicy
    answers:
        HOW desirable is this candidate?
```

---

## 2. Serving Requirements Are Hard Gates

`ServingRequirements` describes the conditions that every qualified serving realization must satisfy.

Example:

```text
max context                 >= 128K

quality regression          <= 0.5%

p99 TTFT                    <= 800 ms

p99 TPOT                    <= 40 ms

minimum output throughput   >= 1500 tok/s
```

These are not optimization preferences.

If a candidate violates any mandatory requirement, it cannot become a qualified `ServingProfile`, even if it is significantly cheaper or faster in some other dimension.

---

## 3. Example Candidate Evaluation

Suppose Model Optimization evaluates four candidates:

```text
                         Hard Requirements
                                |
                                v

Candidate A: vLLM / BF16 / TP4
    TTFT       = 650 ms       PASS
    TPOT       = 35 ms        PASS
    quality    = baseline     PASS
    throughput = 1700 tok/s   PASS
    cost       = $10 / unit

Candidate B: vLLM / FP8 / TP4
    TTFT       = 600 ms       PASS
    TPOT       = 30 ms        PASS
    quality    = -0.2%        PASS
    throughput = 2100 tok/s   PASS
    cost       = $7 / unit

Candidate C: TRT-LLM / FP8 / TP4
    TTFT       = 450 ms       PASS
    TPOT       = 24 ms        PASS
    quality    = -0.3%        PASS
    throughput = 2600 tok/s   PASS
    cost       = $8 / unit

Candidate D: INT4 / TP2
    TTFT       = 420 ms       PASS
    TPOT       = 22 ms        PASS
    quality    = -1.4%        FAIL
```

Candidate D is eliminated immediately.

Its latency and cost advantages do not matter because it violates the quality requirement.

That leaves:

```text
Qualified Candidates

A
B
C
```

At this point the platform still needs to determine which qualified realizations are preferable.

That is where `OptimizationPolicy` applies.

---

## 4. Optimization Policy Defines Preference

Suppose A, B, and C all satisfy `ServingRequirements`.

Different product or platform preferences may favor different candidates.

### Cost-optimized policy

```text
OptimizationPolicy:
    minimize cost per token
```

Possible result:

```text
Candidate B
```

### Latency-optimized policy

```text
OptimizationPolicy:
    minimize TTFT / TPOT
```

Possible result:

```text
Candidate C
```

### Throughput-optimized policy

```text
OptimizationPolicy:
    maximize sustainable token throughput
```

Possible result:

```text
Candidate C
```

The key distinction is:

```text
ServingRequirements
    determine eligibility

OptimizationPolicy
    determines preference
```

---

## 5. Full Candidate Flow

```text
             Candidate Universe
                    |
                    |
          Hard feasibility pruning
                    |
                    v
          Experiment + Measurement
                    |
                    v
          ServingRequirements
             qualification
                    |
                    v
        +-----------------------+
        | Qualified Candidates  |
        |                       |
        | A      B      C       |
        +-----------+-----------+
                    |
                    | OptimizationPolicy
                    v
          preference / frontier
                    |
                    v
            ServingProfile(s)
```

The platform first proves that candidates satisfy the hard contract.

Only then does it optimize among those candidates.

---

## 6. Qualified Frontier

The Optimization Platform does not necessarily need to emit exactly one profile.

There may be several useful qualified realizations:

```text
Qualified Frontier

Profile A
    lowest cost

Profile B
    lowest latency

Profile C
    highest throughput
```

A possible model is:

```text
ServingRequirements
        |
        | hard constraints
        v
+---------------------------+
| Model Optimization        |
|                           |
| Generate Candidates       |
| Run Experiments           |
| Measure                   |
| Eliminate failures        |
+-------------+-------------+
              |
              v
      Qualified Frontier
        /      |       \
       /       |        \
 low-cost   low-latency  high-throughput
    |           |             |
    v           v             v
Profile A    Profile B      Profile C
```

Each published profile must already satisfy the hard `ServingRequirements`.

The Serving Platform should not need to redo Model Optimization or decide whether an unqualified candidate is safe.

---

## 7. Serving Requirements vs Optimization Objective vs Optimization Policy

There are actually three distinct concepts.

### ServingRequirements

Product-level hard constraints.

```text
Examples:

p99 TTFT <= 800 ms

quality regression <= 0.5%

128K context required
```

### OptimizationObjective

Defines what "better" means among qualified candidates.

```text
Examples:

minimize cost/token

minimize latency

maximize throughput
```

### OptimizationPolicy

A platform-owned, versioned policy that implements the chosen objective.

It may define:

```text
search budget

allowed transformation families

allowed runtimes

search strategy

experiment fidelity policy

termination policy
```

This gives the following separation:

```text
ServingRequirements
    =
what MUST be satisfied

OptimizationObjective
    =
what BETTER means

OptimizationPolicy
    =
how the platform searches for better solutions
```

---

## 8. Product Teams Should Not Program the Optimizer

An AI team such as Feed should not submit low-level search mechanics such as:

```text
use TPE

run exactly 80 trials

gamma = 0.2

try TP values {1,2,4,8}

enable this exact quantization backend
```

Those are Model Optimization Platform implementation details.

Instead, Feed may choose a high-level platform-owned policy:

```text
optimization_policy =
    "cost-optimized-v3"
```

Internally that policy may resolve to:

```text
objective:
    minimize cost/token

search strategy:
    TPE

budget:
    max 80 trials

allowed transformations:
    BF16
    FP8
    INT4

termination:
    stop on convergence or budget exhaustion
```

This keeps:

```text
product intent
```

separate from:

```text
optimizer implementation
```

and lets the Model Optimization Platform evolve its search techniques without forcing every AI team to update its pipeline.

---

## 9. Why This Separation Matters

Without this separation, the system can easily mix fundamentally different concerns.

For example:

```text
p99 TTFT <= 800 ms
```

is a product requirement.

But:

```text
prefer lower TTFT after satisfying all requirements
```

is an optimization objective.

And:

```text
use TPE with a 500 GPU-hour budget
```

is an optimization-policy implementation decision.

These must not collapse into one configuration object.

The clean domain model is:

```text
ServingRequirements
        |
        v
Hard acceptance gates

OptimizationObjective
        |
        v
Preference among acceptable candidates

OptimizationPolicy
        |
        v
Search and experiment behavior
```

---

## 10. Locked Principle

The main design principle is:

> **ServingRequirements define the feasible set. OptimizationPolicy determines how the platform searches and prefers solutions inside that feasible set.**

Or more compactly:

```text
ServingRequirements
    = CAN this be used?

OptimizationPolicy
    = WHICH acceptable solutions should we prefer?
```

This distinction should remain consistent throughout:

```text
Candidate Search

Experiment Execution

Qualification

ServingProfile publication

Production re-optimization
```
