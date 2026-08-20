# LLM Routing Layer Design

## Phase 3 — Replica Scheduling Strategy Layers: The “Why” Behind Each Evolution

**Status:** Final design rationale for the V4 scheduler  
**Target:** `SLOCostAwareReplicaScheduler`  
**Purpose:** Explain why each scheduling layer exists, which failure of the previous layer motivates it, and how the final V4 policy emerges without jumping directly to a black-box learned scheduler.

---

## 1. Final Goal

Phase 2 has already selected a quality-approved model revision. Phase 3 must select a small set of serving replicas that can execute the request while respecting runtime objectives.

The final scheduler is not trying to answer only:

```text
Which replica is least loaded?
```

It is trying to answer:

> **Which replicas have a sufficiently high probability of meeting this request's latency SLO, and among those replicas which ones minimize expected execution cost?**

The final policy is:

\[
Feasible(X)=\{R \mid P(Latency(X,R) \le D_X) \ge Q_X\}
\]

and, when the feasible set is non-empty:

\[
R^*=\arg\min_{R \in Feasible(X)} ExpectedCost(X,R)
\]

For queue-arbitrated execution, the scheduler returns **up to three** low-cost SLO-feasible candidates rather than one exact execution owner.

---

## 2. Layer 0 — Round Robin

### Idea

```text
R1 -> R2 -> R3 -> R1 ...
```

### Why it is useful

It is the simplest correctness baseline. It needs almost no runtime state and gives predictable distribution when requests and replicas are homogeneous.

### Why it fails

LLM requests are not homogeneous:

```text
500-token prompt != 50K-token prompt
short decode != long decode
cached prompt != uncached prompt
A100 != H100
vLLM != TensorRT-LLM
idle replica != saturated replica
```

Round Robin has no model of work.

### Why the next layer exists

> We need to observe how much work is already assigned to each replica.

---

## 3. Layer 1 — Least Requests / Least Queue

### Idea

Prefer the replica with the fewest queued/running requests.

\[
Score(R)=QueuedRequests(R)+RunningRequests(R)
\]

### Improvement

It reacts to load and is better than blind distribution.

### Why it fails

One request is not one unit of LLM work.

```text
R1: 2 requests × 50K prompt tokens
R2: 6 requests × 1K prompt tokens
```

Request count may incorrectly prefer R1.

### Why the next layer exists

> Replace request-count load with token/work-aware load.

---

## 4. Layer 2 — Token-Load-Aware Scheduling

### Idea

Track prefill and decode work instead of only request count.

Model-server state contributes:

```text
queued_prefill_tokens
active_prefill_tokens
estimated_remaining_decode_tokens
active_decode_sequences
```

The router contributes work that has been assigned to a replica but has not crossed the model-server admission boundary:

```text
RouterOutstandingPrefillTokens
RouterOutstandingDecodeTokens
```

So:

\[
PrefillWork(R)=EnginePrefillWork(R)+RouterOutstandingPrefill(R)
\]

\[
DecodeWork(R)=EngineDecodeWork(R)+RouterOutstandingDecode(R)
\]

### Improvement

The scheduler now sees meaningful LLM work rather than arbitrary request counts.

### Why it fails

A token is not a universal unit of time.

```text
10K prefill tokens on H100/vLLM
!=
10K prefill tokens on A100/vLLM
```

and:

```text
10K prefill tokens
!=
10K decode tokens
```

### Why the next layer exists

> Convert outstanding work into a common time-domain unit using execution service rates.

---

## 5. Layer 3 — Service-Rate-Normalized Analytical Latency

### Idea

Use the replica's effective prefill/decode rates to convert work to an analytical waiting-time approximation.

\[
T_{prefill\_backlog}(R)=
\frac{EnginePrefillWork(R)+RouterOutstandingPrefill(R)}{PrefillRate(R)}
\]

\[
T_{decode\_backlog}(R)=
\frac{EngineDecodeWork(R)+RouterOutstandingDecode(R)}{DecodeRate(R)}
\]

V1 analytical backlog pressure:

\[
T_{backlog}(R) \approx T_{prefill\_backlog}(R)+T_{decode\_backlog}(R)
\]

This is a **pressure/waiting-time approximation**, not a claim that continuous batching literally serializes all prefill work before all decode work.

### Improvement

Heterogeneous ServingProfiles now compare in a common unit: approximate time.

### Why it fails

It models work already present but still treats the incoming request too generically.

A 30K-token request and a 1K-token request can prefer different replicas even under the same existing backlog.

### Why the next layer exists

> Add request-specific service time.

---

## 6. Layer 4 — Request-Aware Analytical Latency

For incoming request `X`:

\[
T_{request\_prefill}(X,R)=\frac{InputTokens(X)}{PrefillRate(R)}
\]

\[
T_{request\_decode}(X,R)=\frac{ExpectedOutputTokens(X)}{DecodeRate(R)}
\]

Then:

\[
Score(X,R)=T_{backlog}(R)+T_{request\_prefill}(X,R)+T_{request\_decode}(X,R)
\]

### Improvement

The scheduler is no longer merely “least loaded.” It predicts which replica is attractive **for this request**.

### Why it fails

It assumes every prompt token requires new prefill computation.

LLM serving frequently reuses KV-cache prefixes.

### Why the next layer exists

> Make request service time replica-specific using precise prefix locality.

---

## 7. Layer 5 — KV-Aware Analytical Latency

The precise KV index returns:

```text
ReusablePrefixTokens(X,R)
```

Then:

\[
UncachedPrefillTokens(X,R)
=
InputTokens(X)-ReusablePrefixTokens(X,R)
\]

and:

\[
T_{request\_prefill}(X,R)
=
\frac{UncachedPrefillTokens(X,R)}{PrefillRate(R)}
\]

The analytical score becomes:

\[
\boxed{
Score(X,R)
=
\frac{EnginePrefillWork+RouterOutstandingPrefill}{PrefillRate}
+
\frac{EngineDecodeWork+RouterOutstandingDecode}{DecodeRate}
+
\frac{InputTokens-ReusablePrefixTokens}{PrefillRate}
+
\frac{ExpectedOutputTokens}{DecodeRate}
}
\]

Lower is better.

### Improvement

Load, heterogeneous execution speed, incoming request size, and cache locality are all expressed in one interpretable latency approximation.

### Why it still fails

Real LLM serving is nonlinear:

```text
continuous batching
prefill/decode interference
chunked prefill
KV pressure
sequence concurrency
engine scheduler behavior
CUDA graph regions
context-length effects
hardware/engine interactions
```

The additive analytical model cannot capture every interaction accurately, especially at the tail.

### Why the next layer exists

> Learn/predict latency from production execution observations rather than hard-code all interactions.

---

## 8. Layer 6 — Empirical Quantile Scheduler

Before training a learned model, production observations can provide conditional latency distributions.

For each workload/profile/state bucket we can maintain:

```text
P50 / P95 / P99 TTFT
P50 / P95 / P99 TPOT
P50 / P95 / P99 E2E latency
prefill/decode service-rate distributions
```

This immediately improves tail-awareness.

### Important statistical rule

Do not assume:

\[
P99(A+B)=P99(A)+P99(B)
\]

Tail percentiles do not generally compose that way. Prefer direct empirical/predicted TTFT/E2E quantiles when the SLO is defined on those metrics.

### Why the next layer exists

Bucketed statistics become sparse and high-dimensional as we add:

```text
request length
output length
ServingProfile
queue pressure
active sequences
KV reuse
KV capacity
engine configuration
```

> A learned predictor can generalize across this feature space.

---

## 9. Layer 7 — Learned / Predicted-Latency Scheduler

Introduce a predictor:

```text
LatencyPredictor.predict(request, replica_context)
    -> latency distribution / quantiles
```

Useful outputs include:

```text
predicted P50/P95/P99 TTFT
predicted P50/P95/P99 E2E
predicted TPOT quantiles
P(TTFT <= request_ttft_deadline)
P(E2E <= request_e2e_deadline)
```

### Scorer features

#### Request

```text
input_token_count
expected_output_tokens / max_tokens
request type / modality
generation parameters that materially affect runtime
SLO requirements
```

#### Static execution context

```text
ServingProfile
engine / version
hardware
precision / quantization
parallelism
relevant engine features
baseline priors
```

#### Engine dynamic state

```text
queued_prefill_tokens
active_prefill_tokens
active_decode_sequences
estimated_remaining_decode_tokens
admission/capacity state
KV capacity/pressure
```

#### Router-owned state

```text
router_outstanding_prefill_tokens
router_outstanding_decode_tokens
```

#### Request × replica

```text
reusable_prefix_tokens
uncached_prefill_tokens
cache tier if relevant
```

#### Freshness/confidence

```text
state age/version
measurement sample count
performance-model confidence
```

### Feedback requirement

Training examples join:

```text
SchedulingDecisionEvent
+
RequestExecutionObservation
```

The decision event records the features/state versions actually used at scheduling time. The execution observation records what happened on the replica that executed the request.

### Known limitation

Only the executed replica produces a factual latency label. Unchosen replicas are counterfactuals. Controlled exploration/selection-bias correction is a later training-system topic, not required to define the routing architecture.

### Why the next layer exists

Minimizing predicted latency is still not the business objective.

A faster H100 replica may cost more while a slower A100 replica may still comfortably meet the request SLO.

> Use predicted latency as a feasibility signal, then optimize cost.

---

## 10. Layer 8 / Final V4 — `SLOCostAwareReplicaScheduler`

### Request SLO contract

Conceptually:

```text
LatencySLO
{
    metric                 # TTFT / E2E / other supported metric
    deadline_ms
    target_probability     # e.g. 0.99
}
```

A request may carry more than one supported constraint, for example TTFT and E2E.

For each eligible candidate `R`, the predictor produces an SLO probability:

\[
p_R=P(Latency(X,R) \le D_X)
\]

A candidate is SLO-feasible if:

\[
p_R \ge Q_X
\]

### Cost model

The scheduler also receives:

```text
ExpectedCost(X,R)
```

The estimator can use ServingProfile cost rate plus predicted resource/service demand. How cloud/internal accelerator cost priors are generated is separate from the Router architecture.

### Final objective

If at least one replica is feasible:

\[
\boxed{
R^*=
\arg\min_{R:\;P(Latency(X,R)\le D_X)\ge Q_X}
ExpectedCost(X,R)
}
\]

This is intentionally **constraint + optimization**, not an arbitrary weighted sum of latency and dollars.

### Top-K candidate-set semantics

Queue-arbitrated execution means the scheduler does not choose one final owner.

V4 therefore chooses:

```text
up to Kmax=3 lowest-cost SLO-feasible candidates
```

Every candidate allowed to compete in the normal phase should satisfy the request SLO according to the current prediction policy.

This prevents an SLO-infeasible replica from winning queue arbitration merely because its queue happened to drain first.

### No feasible replica

If:

```text
Feasible(X) = empty
```

V4 does not pretend the SLO is satisfiable.

It returns a plan explicitly marked:

```text
SLO_UNSATISFIABLE
```

and ranks degraded candidates by a violation-minimizing policy, for example:

1. highest predicted SLO probability / smallest predicted violation risk;
2. then lower predicted tail latency;
3. then lower cost.

The execution/policy layer can then decide whether degraded execution is allowed or whether to advance to a fallback model/serving class.

---

## 11. Why Queue Arbitration Still Matters in V4

Predicted state is never perfectly current.

The scheduler may think:

```text
R1 P99 TTFT = 170ms
R3 P99 TTFT = 180ms
R5 P99 TTFT = 185ms
```

and all three satisfy a 200ms SLO.

By sending the request into all three candidate queues, the shared queue layer gives a second, much faster-timescale correction for simultaneous Planner decisions and transient contention.

The learned scheduler handles **expected placement quality**.

The queue layer handles **distributed execution arbitration under imperfect state**.

---

## 12. Final Evolution Story

```text
Round Robin
    |
    | request/replica load differs
    v
Least Requests
    |
    | request count != LLM work
    v
Token-Load Aware
    |
    | token work != time across prefill/decode/profiles
    v
Service-Rate Normalized
    |
    | incoming request itself differs
    v
Request-Aware Analytical Latency
    |
    | KV reuse makes service cost replica-specific
    v
KV-Aware Analytical Latency
    |
    | real serving interactions are nonlinear/tail-sensitive
    v
Empirical Quantiles
    |
    | buckets become sparse/high-dimensional
    v
Learned Predicted Latency
    |
    | minimum latency is not minimum cost
    v
SLO Feasibility + Cost Optimization
    |
    v
SLOCostAwareReplicaScheduler
```

The important Staff+ message is:

> **V4 is not a disconnected “ML magic” scheduler. Every layer is a response to a concrete failure mode of the previous one, and the analytical layers remain valuable as baselines, fallbacks, debugging tools, and model features.**
