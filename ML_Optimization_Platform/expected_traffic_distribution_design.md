# Expected Traffic Distribution — Histogram and Segment Design

> **Status:** Locked reference for the first version of the Model Optimization Platform  
> **Schema:** `expected_traffic_distribution.proto`  
> **Example:** `expected_traffic_distribution.example.json`

## 1. Purpose

`ExpectedTrafficDistribution` describes the workload that a candidate `ServingProfile` is expected to serve.

It is intentionally separate from:

```text
ServingRequirements
    = what the serving realization MUST satisfy

ExpectedTrafficDistribution
    = what workload we EXPECT it to experience

OptimizationPolicy
    = how the Optimization Platform should search
```

The expected traffic artifact is immutable and versioned. An `OptimizationJob` pins one exact version so that every benchmark result and qualification decision can be traced back to the exact workload assumptions used when the profile was optimized.

---

## 2. Why simple percentiles are not enough

A first attempt at describing traffic might look like:

```text
prompt p50 = 2K
prompt p95 = 15K
prompt max = 128K

output p50 = 400
output p95 = 2K

peak QPS = 10K
```

Those values are useful for a presentation, but they are insufficient for a model optimizer.

LLM serving cost is highly nonlinear with sequence length:

```text
short prompt
    -> relatively small prefill cost
    -> smaller KV footprint

long prompt
    -> much larger prefill cost
    -> much larger KV footprint
    -> potentially different TP / CP / batching behavior
```

Two workloads can have the same p50 and p95 while having very different tails, burst shapes, prefix reuse, and prompt/output correlations.

The optimizer therefore needs more than a handful of global percentiles.

---

## 3. The two-level distribution

The chosen model is:

```text
All requests
    |
    +-- Traffic Segment A       request_fraction = ...
    |
    +-- Traffic Segment B       request_fraction = ...
    |
    +-- Traffic Segment C       request_fraction = ...
```

Each segment then contains its own conditional distributions:

```text
TrafficSegment
    |
    +-- input-token histogram
    +-- output-token histogram
    +-- streaming fraction
    +-- prefix-reuse profile
    +-- service class
```

For the example workload:

```text
All production requests
        |
        +-- short_interactive      60%
        |
        +-- medium_context         30%
        |
        +-- long_context           10%
```

The segment fractions are probabilities:

```text
P(short_interactive) = 0.60
P(medium_context)    = 0.30
P(long_context)      = 0.10
```

They must sum to 1.0.

---

## 4. Conditional probability inside a segment

Consider the `long_context` segment:

```text
P(long_context) = 0.10
```

Within that segment, input length is described by explicit histogram buckets:

```text
[16,385,  32,768] tokens     35%
[32,769,  65,536] tokens     40%
[65,537, 131,072] tokens     25%
```

These are conditional probabilities:

```text
P(16K-32K tokens | long_context)   = 0.35
P(32K-64K tokens | long_context)   = 0.40
P(64K-128K tokens | long_context)  = 0.25
```

Therefore, over all production requests:

```text
P(long_context AND 16K-32K)
    = 0.10 * 0.35
    = 0.035
    = 3.5%

P(long_context AND 32K-64K)
    = 0.10 * 0.40
    = 4.0%

P(long_context AND 64K-128K)
    = 0.10 * 0.25
    = 2.5%
```

This gives the optimizer a much more useful representation of the real workload than a single global percentile.

---

## 5. Why buckets are explicitly bounded

An earlier representation used only:

```text
upper_bound_tokens = 32768
```

That creates ambiguity.

Inside a segment called `long_context`, does the first bucket mean:

```text
0-32768?
16K-32768?
something else?
```

The schema now makes every bucket explicit:

```text
TokenCountBucket {
    min_tokens
    max_tokens
    probability
}
```

with closed integer interval semantics:

```text
[min_tokens, max_tokens]
```

Example:

```json
{
  "min_tokens": 16385,
  "max_tokens": 32768,
  "probability": 0.35
}
```

means exactly:

```text
16,385 <= token_count <= 32,768
```

Validation rules:

```text
min_tokens <= max_tokens

buckets must not overlap

sum(bucket.probability) = 1.0

gaps are allowed only when intentional
```

This makes the artifact self-describing and avoids relying on segment names to infer numeric boundaries.

---

## 6. Why segment first, histogram second

We could instead store one global input-token histogram and one global output-token histogram.

That loses correlations.

Imagine that production behaves like:

```text
short prompts
    -> usually short outputs

long prompts
    -> usually larger outputs
    -> higher prefix reuse
```

Independent global histograms can accidentally synthesize combinations that almost never happen in production, such as a very high fraction of:

```text
128K input + 4K output
```

even if those two tails rarely occur together.

Segmenting first lets us retain useful conditional structure:

```text
short_interactive
    input distribution A
    output distribution A
    prefix behavior A

medium_context
    input distribution B
    output distribution B
    prefix behavior B

long_context
    input distribution C
    output distribution C
    prefix behavior C
```

This is still compact enough for control-plane use while being substantially more realistic than independent global percentiles.

---

## 7. Why not make each exact request shape its own segment

The opposite extreme is also undesirable.

We do not want:

```text
segment_1 = 1,024 input / 128 output
segment_2 = 1,031 input / 131 output
segment_3 = ...
```

That would:

```text
explode metadata size
overfit historical traffic
make search reasoning difficult
make workload versions unstable
```

Segments should represent **behaviorally meaningful workload classes**, not individual observations.

Good segment boundaries are those that can change serving behavior, for example:

```text
short interactive
medium context
long context
multimodal
large-output generation
high-prefix-reuse traffic
critical interactive
async
```

The exact grouping strategy can be generated by a traffic-profiling pipeline and versioned.

---

## 8. Why include an optional representative trace

The histogram representation is deliberately summarized.

It is useful for:

```text
search-space reasoning
analytical estimates
candidate prioritization
benchmark-plan construction
qualification reporting
drift comparison
```

But a summary cannot perfectly preserve:

```text
arrival burstiness
cross-field correlations
temporal locality
prefix repetition sequences
request inter-arrival timing
```

For high-fidelity testing, the artifact can therefore reference an immutable sampled trace:

```text
representative_trace_ref =
    traffic-store://feed-gen/expected/v12/sample-1m-requests
```

The intended hierarchy is:

```text
Histogram / segments
    -> compact semantic representation

Representative trace
    -> high-fidelity replay when required
```

The trace does not replace the summarized model; both serve different purposes.

---

## 9. Arrival rate versus server concurrency

For online traffic, the artifact primarily describes external arrival demand:

```text
steady-state QPS
p95 one-minute QPS
peak one-minute QPS
peak duration
```

It intentionally does not declare:

```text
server active-sequence concurrency = N
```

as a primary input.

Why?

By Little's Law:

```text
active concurrency ~= arrival rate * service time
```

and service time depends on the candidate being tested.

For the same workload:

```text
Candidate A
    slower
    -> more requests simultaneously resident

Candidate B
    faster
    -> fewer requests simultaneously resident
```

Therefore server concurrency is partly an output of candidate performance.

If the product team has observed client-side outstanding-request concurrency, it may be carried as an optional validation signal, but the optimizer must discover candidate-specific server concurrency experimentally.

---

## 10. Online versus offline demand

Online and offline workloads use the same overall traffic artifact but different demand profiles.

Online:

```text
QPS
burst duration
request mix
latency-sensitive service classes
```

Offline:

```text
records/job
input tokens/job
output tokens/job
jobs/day
concurrent jobs
```

Completion-time or latency requirements do not belong in the traffic artifact. Those remain in `ServingRequirements`.

This preserves the boundary:

```text
Traffic:
    what arrives

Requirements:
    what performance is acceptable
```

---

## 11. How Candidate Search uses this artifact

Candidate Search can use the expected distribution before launching expensive GPU trials.

Examples:

```text
Long-context fraction is high
    -> prioritize candidates with adequate KV capacity
    -> consider CP/DCP-capable runtime/topologies

Prefix reuse is high
    -> prioritize candidates/runtime settings with prefix caching support

Output-heavy workload
    -> decode throughput becomes more important

Mostly short prompts
    -> large-context topology may be unnecessary overhead
```

These are search heuristics, not qualification results.

Analytical reasoning can prioritize or prune experiments, but empirical trials still determine final qualification.

---

## 12. How Benchmark Planning uses the artifact

The benchmark planner can synthesize or replay requests according to:

```text
segment.request_fraction

segment.input_tokens

segment.output_tokens

segment.streaming_fraction

segment.prefix_reuse

arrival_profile
```

A simple synthetic generator could conceptually do:

```text
1. choose TrafficSegment using request_fraction

2. choose input-token bucket from that segment

3. sample a concrete token length inside the chosen bucket

4. choose output-token bucket

5. sample output length

6. apply segment-specific streaming/prefix behavior

7. schedule arrival according to the requested load profile
```

For high-fidelity qualification, a representative trace can replace or augment synthetic generation.

---

## 13. How Qualification uses the artifact

An `ExperimentResult` is meaningful only relative to the workload under which it was measured.

Qualification therefore binds:

```text
Candidate
+
Exact Artifact
+
Hardware
+
Runtime
+
ExpectedTrafficDistribution version
+
ServingRequirements version
```

A `ServingProfile` should effectively say:

> This realization was empirically qualified against `ExpectedTrafficDistribution v12` and `ServingRequirements v7`.

It should not claim to be universally optimal.

---

## 14. How production drift uses the same model

After deployment, production generates a separate:

```text
ObservedTrafficDistribution
```

with compatible dimensions.

The platform can compare:

```text
ExpectedTrafficDistribution v12

vs.

ObservedTrafficDistribution
```

Examples of meaningful drift:

```text
long-context fraction:
    expected 10%
    observed 32%

peak QPS:
    expected 10K
    observed 15K

prefix reuse:
    expected 40%
    observed 5%

output-length mix:
    expected mostly <= 1K
    observed substantial 4K tail
```

That may invalidate the assumptions under which the current ServingProfile was qualified and trigger requalification or reoptimization.

The expected artifact itself remains immutable.

---

## 15. Example interpretation

Given:

```text
long_context request_fraction = 0.10
```

and:

```text
input histogram:

16,385-32,768      0.35
32,769-65,536      0.40
65,537-131,072     0.25
```

the correct verbal interpretation is:

> We expect approximately 10% of production requests to belong to the long-context segment. Conditional on a request belonging to that segment, about 35% are expected to have 16K-32K input tokens, 40% to have 32K-64K, and 25% to have 64K-128K.

Across all traffic:

```text
3.5% of requests are expected in long-context/16K-32K
4.0% in long-context/32K-64K
2.5% in long-context/64K-128K
```

This conditional interpretation is the central mental model.

---

## 16. Locked design principles

```text
1. ExpectedTrafficDistribution is immutable and versioned.

2. Traffic describes demand, not serving SLOs.

3. Weighted segments represent meaningful workload classes.

4. Histograms inside segments represent conditional distributions.

5. Token buckets have explicit lower and upper bounds.

6. Segment probabilities sum to 1.

7. Bucket probabilities within each histogram sum to 1.

8. Segments preserve important correlations that global independent
   histograms would lose.

9. QPS / arrival demand is primary for online workloads; server-side
   active concurrency is candidate-dependent and measured experimentally.

10. A representative sampled trace is optional for high-fidelity replay.

11. Qualification always references the exact traffic-distribution version
    used during experiments.

12. Production observations are separate artifacts and never mutate the
    expected distribution used for qualification.
```
