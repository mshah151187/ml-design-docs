# Optimization Search Process — Working Design

> **Status:** Saved design discussion; this will be refined when the exact Candidate Search process is designed.  
> **Purpose:** Capture the current search-process model so it can be used as the baseline for the next deep dive.

---

# 1. Core Principle

The optimizer does **not**:

```text
change one knob
run an experiment
change the next knob
run another experiment
```

and it does **not** brute-force the Cartesian product of every engine/runtime parameter.

Runtime knobs interact strongly.

One experiment evaluates one **complete candidate configuration** in the currently active search subspace.

Example:

```text
CandidateConfiguration {
    hardware = H100
    runtime = vLLM
    precision = FP8
    tp = 4

    max_num_seqs = 96
    max_num_batched_tokens = 32768
    prefix_cache = true
    chunked_prefill = true
    kv_cache_dtype = FP8
    cuda_graph_policy = ...
}
```

The experiment measures the joint behavior of that configuration.

---

# 2. Why One-Knob-at-a-Time Tuning Is Wrong

Consider:

```text
max_num_seqs ↑
        +
max_num_batched_tokens ↑
        |
        v
more batching
        |
        +--> throughput may increase
        +--> KV memory increases
        +--> queueing may increase
        +--> TTFT may worsen
```

The effect of `max_num_seqs=128` cannot be understood independently of the token budget, workload distribution, KV configuration, and runtime scheduler.

Therefore knobs must eventually be tuned jointly.

---

# 3. Two Classes of Optimization Dimensions

## Structural Dimensions

These change the fundamental serving realization and may require new artifacts, builds, resource topology, or deployment shape.

```text
hardware

runtime / engine

quantization / model representation

TP / PP / CP / DCP

P/D topology
```

Examples:

```text
BF16 -> FP8
    may require a new derived model artifact

vLLM -> TensorRT-LLM
    may require format conversion / engine compilation

TP4 -> TP8
    changes GPU allocation and collective topology
```

Structural experiments are comparatively expensive.

---

## Runtime Tuning Dimensions

These tune performance inside one structural realization.

Examples:

```text
max_num_seqs

max_num_batched_tokens

prefix caching

chunked prefill

prefill chunk size

KV-cache dtype/config

CUDA graph settings

runtime scheduler knobs
```

Once:

```text
vLLM + FP8 + TP4
```

has been materialized, multiple runtime configurations can often be tested without requantizing/rebuilding the model.

---

# 4. Hierarchical Search

The current search design is hierarchical.

```text
                    Optimization Job
                           |
                           v
              +-------------------------+
              | STRUCTURAL SEARCH       |
              |                         |
              | hardware                |
              | runtime                 |
              | quantization            |
              | TP / PP / CP            |
              | P/D topology            |
              +------------+------------+
                           |
                     promising
                   realizations
                           |
                           v
              +-------------------------+
              | RUNTIME KNOB SCREENING  |
              |                         |
              | identify high-impact    |
              | knobs and useful ranges |
              +------------+------------+
                           |
                           v
              +-------------------------+
              | JOINT RUNTIME TUNING    |
              |                         |
              | model-based optimizer   |
              | proposes complete       |
              | knob combinations       |
              +------------+------------+
                           |
                           v
                  Final Qualification
```

The goal is to avoid spending expensive tuning budget on structural realizations that are already weak.

---

# 5. Structural Screening

A structural screening set might contain:

```text
vLLM BF16 TP4

vLLM FP8 TP4

vLLM FP8 TP8

TensorRT-LLM FP8 TP4

TensorRT-LLM FP8 TP8

vLLM INT4 TP4
```

These candidates receive relatively cheap initial trials.

Weak structural families can be eliminated before deep runtime tuning.

Example:

```text
vLLM BF16 TP4
    -> dominated

INT4 TP4
    -> fails quality

TensorRT-LLM FP8 TP8
    -> too costly for achieved benefit
```

Promising families then receive additional tuning budget.

---

# 6. Runtime Knob Screening

Runtimes such as:

```text
vLLM
SGLang
TensorRT-LLM
```

expose many parameters.

The optimizer should not automatically search every exposed runtime flag.

The Model Optimization Platform maintains a curated optimization-knob catalog containing parameters known to materially influence serving behavior.

Example conceptual catalog:

```text
HIGH IMPACT
    max_num_seqs
    max_num_batched_tokens
    KV dtype
    prefix caching
    chunked prefill
    CUDA graph strategy

CONDITIONALLY RELEVANT
    prefill chunk size
    scheduler policy
    attention backend

PLATFORM FIXED / NOT SEARCHED
    logging flags
    debugging switches
    unrelated networking flags
```

A screening stage determines which knobs materially move the target objective for the current model/workload.

This reduces a large runtime configuration space into a smaller high-impact subspace.

---

# 7. Joint Runtime Optimization

After screening, the remaining runtime knobs are optimized jointly.

Example vLLM subspace:

```text
max_num_seqs:
    integer [16, 256]

max_num_batched_tokens:
    integer [4096, 65536]

prefix_caching:
    boolean

chunked_prefill:
    boolean

prefill_chunk_size:
    conditional integer range

kv_cache_dtype:
    {BF16, FP8}
```

A model-based optimizer proposes a complete point:

```text
Trial 17

max_num_seqs = 96
max_num_batched_tokens = 32768
prefix_caching = true
chunked_prefill = true
prefill_chunk_size = 8192
kv_cache_dtype = FP8
```

The measured result updates the search model, which proposes the next joint configuration.

---

# 8. Conditional Search Subspaces

Different structural choices expose different runtime knobs.

```text
runtime
    |
    +-- vLLM
    |      +-- vLLM tuning subspace
    |
    +-- SGLang
    |      +-- SGLang tuning subspace
    |
    +-- TensorRT-LLM
           +-- TensorRT-LLM tuning subspace
```

Similarly:

```text
if prefix_cache == false:
    prefix-cache-specific knobs are inactive

if chunked_prefill == true:
    prefill-chunk-size becomes active

if precision == INT4:
    INT4-specific quantization parameters become active
```

This is the meaning of a **conditional search space**.

---

# 9. Policy and Traffic Shape the Active Search

The full engine parameter surface should not receive equal search budget.

The active search subspace depends on:

```text
OptimizationPolicy
        +
ExpectedTrafficDistribution
        +
CanonicalModel characteristics
        +
Runtime / hardware capabilities
```

Examples:

```text
high prefix reuse
    -> prefix caching becomes important

high long-context fraction
    -> chunked prefill, context parallelism,
       KV precision/capacity become important

THROUGHPUT_OPTIMIZED
    -> batching/token-budget knobs receive more attention

INTERACTIVE_LATENCY_OPTIMIZED
    -> queueing, batching limits, prefill, CUDA graphs
       receive more attention
```

The policy influences search priors and budget allocation, not correctness.

---

# 10. Multi-Fidelity Experimentation

Not every candidate should immediately receive an expensive full qualification run.

Current design:

```text
Level 1 — Smoke / feasibility
    short run
    detect OOM / incompatibility / obviously bad configuration

Level 2 — Short performance trial
    estimate latency and throughput

Level 3 — Promising-candidate trial
    realistic traffic distribution
    longer measurement window

Level 4 — Final qualification
    stable long run
    tail latency
    quality
    capacity
    failure behavior
```

Conceptually:

```text
many candidates
      |
      | cheap trials
      v
smaller candidate set
      |
      | medium trials
      v
few finalists
      |
      | expensive qualification
      v
Qualified ServingProfiles
```

---

# 11. Search Process Phases

The search process can be summarized as:

```text
Phase 1:
    Analytical Seeding

Phase 2:
    Historical Warm Start

Phase 3:
    Hierarchical Structural Search

Phase 4:
    Runtime Knob Screening

Phase 5:
    Joint Model-Based Runtime Optimization

Phase 6:
    Multi-Fidelity Promotion

Phase 7:
    Final Qualification
```

Analytical and historical seeds may be generated in parallel and merged into the initial candidate set.

---

# 12. Locked Search Principle

> **Do not brute-force the Cartesian product and do not tune knobs independently. Search hierarchically: first screen structural serving choices, then identify the runtime knobs that materially affect the chosen policy/objective, then jointly optimize those knobs inside promising structural realizations, using progressively more expensive empirical trials.**

This document is the baseline for the later exact Candidate Search design.
