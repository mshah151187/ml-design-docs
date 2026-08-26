# Principal Staff Gaps — LLM Routing Design

## Current Position

The LLM Routing design is already technically deep enough for a Principal-level discussion:

- request normalization
- session-aware inference
- quality-constrained model routing
- target-model context materialization
- P/D disaggregation and conditional disaggregation
- replica/load-aware scheduling
- TTFT / TPOT / SLO-aware planning
- prefix/KV locality
- KV compatibility + exact transforms
- NIXL/RDMA transfer
- KV Directory + external KV Tier
- queue-arbitrated execution
- failure/retry ownership
- routing-latency degradation

The remaining gap is **not another routing algorithm**.

> **Principal Staff depth comes from showing how this becomes an operated platform: capacity, overload, fairness, lifecycle, control loops, economics and failure domains.**

---

## 1. Capacity Planning + Fleet Control

Router assumes healthy capacity already exists.

Principal-level design should answer:

- How many replicas?
- How many P vs D workers?
- Which GPU types?
- Which TP/PP profiles?
- How much burst reserve?
- When do we autoscale?
- When do we rebalance P/D pools?

Reason from:

- arrival rate
- ISL / OSL distributions
- Prefill tokens/sec
- Decode tokens/sec
- KV capacity
- SLO headroom
- failure reserve
- cost

Important boundary:

```text
Router = request placement
Fleet Controller = capacity placement
```

They form interacting feedback loops.

---

## 2. Admission Control + Overload Policy

A production system must define when **not** to serve.

Example escalation:

```text
NORMAL
  ->
degrade optional routing sophistication
  ->
throttle admission
  ->
use quality-approved fallback
  ->
shed lower-priority traffic
```

Need explicit policy for queue deadlines, priority traffic, reserved capacity, quality fallback, output limits, and load shedding.

`SLO_UNSATISFIABLE` should result in deliberate policy, not accidental queue growth.

---

## 3. Multi-Tenancy + Fairness

Add platform-level resource governance:

- tenant quotas
- fair-share scheduling
- priority classes
- burst credits
- reserved capacity
- concurrency limits
- noisy-neighbor isolation

KV Tier adds new fairness dimensions:

- per-tenant KV capacity
- cache eviction fairness
- cross-tenant reuse policy
- billing attribution
- privacy boundaries

---

## 4. Deployment + Model Lifecycle

Need an operational lifecycle for new model revisions, ServingProfiles, tokenizer/templates, quality predictors, and KV representations.

Example:

```text
QUALIFYING
 -> SHADOW
 -> CANARY
 -> ACTIVE
 -> DRAINING
 -> RETIRED
```

Must explain rolling upgrades, KV namespace isolation, session migration, cold-start calibration, rollback, and stale metadata handling.

---

## 5. Observability + Decision Explainability

Every routed request should answer:

> **Why did it go there?**

Capture a routing trace containing:

- selected model + quality score
- candidate topologies considered
- KV matches
- predicted TTFT / TPOT
- predicted cost
- chosen materialization path
- fallbacks taken

Critical metrics:

- routing latency
- quality calibration
- prediction error
- queue wait
- TTFT / TPOT / E2E
- KV useful-token hit rate
- restore-vs-recompute regret
- transfer bandwidth
- fallback frequency
- cost/request

Especially important:

> **Measure routing regret, not only routing outcomes.**

---

## 6. Control-Loop Stability

Several controllers can react simultaneously:

```text
request router
autoscaler
P/D pool balancer
KV promotion/eviction
admission controller
```

Risks include herding, oscillation, stale telemetry, positive feedback, and capacity chasing load.

Need concepts such as EWMA, hysteresis, rate limiting, randomization, control-loop timescales, and load forecasting.

Different loops should operate at different timescales.

---

## 7. Failure Domains + Multi-Region

Extend cluster-level reasoning to rack, zone, and region.

Questions:

- What survives zone loss?
- Where is SessionStore replicated?
- Is KV replicated cross-zone?
- When is WAN KV transfer worse than recompute?
- How is capacity reserved for zonal failure?
- How do data-residency policies affect routing?

Likely principle:

```text
Session semantic state:
    durability-driven replication

KV:
    locality-driven optimization
```

---

## 8. Economics + Cost Model

Attach real units to routing decisions:

- GPU-ms
- $/GPU-hour
- KV GB-hours
- network bytes
- CPU/DRAM occupancy
- NVMe occupancy
- transfer opportunity cost

For cache:

```text
CacheValue
=
AvoidedPrefillCost
-
StorageCost
-
TransferCost
```

A high cache hit rate can still be economically bad.

For model/topology selection:

```text
meet quality + SLO first
then minimize expected platform cost
```

---

## 9. Prediction Robustness

Prefer:

```text
analytical baseline
+
learned correction
```

Example:

```text
PrefillTime
≈ uncached_tokens / effective_prefill_rate

TransferTime
≈ setup + bytes / effective_bandwidth
```

Then add learned residuals.

Need confidence intervals, calibration, OOD detection, fallback models, and rollback.

Never make correctness depend on a black-box predictor.

---

## 10. Principal-Level System Invariants

Be able to state these immediately:

```text
1. Phase 3 never violates Phase 2's quality contract.

2. SessionStore is the only authoritative
   multi-turn semantic state.

3. KV is reconstructible optimization state;
   loss always has a safe recompute path.

4. Exactly one terminal execution owner exists
   before externally visible streaming.

5. Imported KV becomes usable only after
   compatibility + transfer + validation complete.

6. Every active request belongs to exactly one
   load-accounting domain.

7. Router sophistication may degrade under deadline;
   correctness checks never do.
```

---

## 11. Staff+ Coding Depth

Practice implementation of production components, not CUDA trivia:

- `RequestExecutionRegistry.TryClaim`
- lease expiry + fencing
- `RouterOutstandingState` concurrent state transitions
- `KVDirectoryIndex` prefix insertion/removal, longest-prefix lookup, epoch invalidation
- `SessionStore` BeginTurn / CommitTurn CAS
- `ContextMaterializationCoordinator` async fallback chain and cleanup
- P/D topology candidate builder
- bounded stale-state caches

Primary signal:

> **state ownership + concurrency + failure cleanup + invariants**

---

## Final Principal Staff Framing

A Staff answer says:

> "Here are the components and how they work."

A Principal Staff answer adds:

> "Here are the invariants, SLO and economic objectives, overload behavior, capacity-control loops, tenancy model, deployment lifecycle, failure domains, and the pieces I intentionally did not build yet."

The LLM Routing architecture is technically complete enough.

The next growth area is **fleet/resource orchestration**, not additional routing mechanisms.
