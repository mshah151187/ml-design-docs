# LLM Routing

## What is Routing and Why Does It Exist?

You have multiple LLMs available. They differ in quality, speed, and cost:

```
Model           Quality   Latency   Cost (per 1M tokens)
─────────────────────────────────────────────────────────
GPT-4o          ████████  slow      $15.00
Llama 70B       ██████░░  medium    $0.90  (self-hosted)
Llama 8B        ████░░░░  fast      $0.10  (self-hosted)
Qwen 1.5B       ██░░░░░░  very fast $0.01  (self-hosted)
```

Not every query needs the most capable model:

```
"What is 2+2?"                              → Qwen 1.5B handles perfectly
"Summarize this article in 2 sentences"     → Llama 8B is fine
"Write a legal analysis comparing 3 cases"  → needs GPT-4o
```

Sending everything to GPT-4o: best quality, 150x more expensive than necessary.
Sending everything to Llama 8B: cheap, but quality collapses on hard queries.

**Routing is the system that decides, per query, which model to use — maximizing
quality while controlling cost and respecting infrastructure capacity.**

---

## The Two Axes Every Router Balances

```
         HIGH QUALITY
              │
   GPT-4o  ──┤──  Always use best
              │         ↑ unaffordable at scale
              │
   Llama 70B─┤
              │
   Llama 8B ─┤──  Always use cheapest
              │         ↓ quality degrades on hard queries
              │
         LOW QUALITY
              └────────────────────────────
                 LOW COST      HIGH COST
```

A good router operates on the Pareto frontier: for a given cost budget, achieve
the highest possible average quality across all queries.

---

## Building Blocks of Any Routing System

Every routing system — online or offline — has the same four components:

```
┌─────────────────────────────────────────────────────────────┐
│                     ROUTING SYSTEM                          │
│                                                             │
│  1. FEATURE EXTRACTOR                                       │
│     Input: raw query                                        │
│     Output: signals about query difficulty                  │
│     (length, complexity, topic, entity count, ...)         │
│                                                             │
│  2. PERFORMANCE ESTIMATOR                                   │
│     Input: query features + model profile                  │
│     Output: predicted quality score per model              │
│     (trained offline on historical query-model-score data) │
│                                                             │
│  3. COST ESTIMATOR                                          │
│     Input: query + model                                    │
│     Output: estimated inference cost                        │
│     (usually: input tokens × price + output tokens × price)│
│                                                             │
│  4. DECISION LOGIC                                          │
│     Input: quality scores + costs + constraints            │
│     Output: which model handles this query                  │
│     (per-query: pick best under cost penalty)              │
│     (batch-level: ILP optimizer)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 1: Routing in Online Systems (Real-Time)

### The Constraint

The routing decision itself must be fast — typically under 5–10ms. If routing takes
50ms in a system with a 200ms latency SLA, you've burned 25% of your budget before
inference even starts.

This means:
- No large model calls for routing decisions
- Lightweight estimators only (kNN, XGBoost, small NN)
- Pre-computation wherever possible

### System Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────┐
│  ONLINE ROUTER  (must complete in < 10ms)             │
│                                                       │
│  ┌─────────────┐    ┌──────────────────────────────┐ │
│  │  Feature    │    │  Performance Estimator       │ │
│  │  Extractor  │───▶│  (XGBoost or kNN)            │ │
│  │             │    │  → score per model           │ │
│  │  - length   │    └──────────────┬───────────────┘ │
│  │  - topic    │                   │                 │
│  │  - entities │    ┌──────────────▼───────────────┐ │
│  │  - user ctx │    │  Decision Logic              │ │
│  └─────────────┘    │  pick model within budget    │ │
│                     └──────────────┬───────────────┘ │
└────────────────────────────────────┼─────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
         [GPT-4o]              [Llama 70B]             [Llama 8B]
```

### Performance Estimators

**kNN (k-Nearest Neighbors)**
- Embed the incoming query into a vector
- Find k most similar past queries in a lookup index
- Average the quality scores those queries got on each model
- Fast at inference, requires no retraining when new queries arrive
- Weakness: fails on novel query types with no similar history

**XGBoost**
- Train gradient-boosted trees on (query features, model profile) → quality score
- Input features: token count, average token frequency (rare = harder), topic tags,
  sentence count, presence of numerical constraints, etc.
- Fast inference (microseconds), handles non-linear patterns well
- Needs periodic retraining as query distribution shifts

**Lightweight Neural Network**
- Encode query + model profile into shared embedding, predict quality
- More expressive than XGBoost, slower to retrain
- Used in MIRT (multi-dimensional item response theory) approach

**Which to use**: XGBoost is the practical default. Fast, interpretable, works well
on tabular query features, easy to retrain incrementally.

### Route Cache

For production systems at scale, cache routing decisions for common queries:

```
Incoming query
    │
    ▼
[Route Cache Lookup]  ← Redis, keyed by query embedding bucket
    │
    ├── Cache HIT → use cached model assignment (0ms overhead)
    │
    └── Cache MISS → run router → cache result → route query
```

Common queries (head of the distribution) get free routing. Long-tail queries
pay the routing cost once and get cached for future occurrences.

### Fallback and Circuit Breaker

```
Router assigns query to Llama 70B
    │
    ▼
[Llama 70B] ── unavailable / overloaded / latency > SLA
    │
    ▼
[Circuit Breaker] ── trips after N consecutive failures
    │
    ▼
[Fallback: GPT-4o API]  ← always available, pay per token
```

Always have a cloud API fallback for self-hosted models. When a self-hosted model
goes down, you pay more temporarily but never drop traffic.

### Cost Control Mechanisms

**Per-query penalty (λ)**: subtract cost × λ from quality score. Higher λ = prefer
cheaper models. Simple but hard to enforce a strict budget — λ is a knob, not a
budget dial.

**Hard budget cap**: track running cost per minute/hour. When approaching budget,
shift routing weight toward cheaper models. Implemented as a dynamic λ adjustment.

**Token budget per query**: cap max output tokens. Prevents runaway costs from
verbose queries.

---

## Part 2: Routing in Semantic Search Systems

Semantic search has THREE distinct places where routing decisions happen:

### Layer 1: Query Understanding Routing

Before retrieval starts, a Query Understanding (QU) layer interprets the query.
Routing here decides WHAT KIND of processing the query needs:

```
User types: "jobs near me paying $150K easy apply"
                │
                ▼
        [QU Router — 1.5B SLM]
                │
    ┌───────────┼───────────────────┐
    ▼           ▼                   ▼
[Criteria]  [Self-ref]          [Unsupported]
 extract     inject              keyword-only
 tags        member context      fallback
 title,      rewrite query
 location,   before retrieval
 salary
```

The four routing outcomes at LinkedIn (from the QU paper):
- **Criteria**: extract structured tags → pass to retrieval as filters
- **Self-referential**: query references user context ("near me", "my network")
  → rewrite with member profile before retrieval
- **Unsupported/unclear**: ambiguous intent → safe fallback to keyword search
- **Trust violation**: block or deflect

This routing is done by the same SLM that does tagging and rewriting — routing
is the first token generated, and it gates everything else in the output schema.

### Layer 2: Retrieval Routing

Given the interpreted query, decide HOW to retrieve candidates:

```
Interpreted query + tags
        │
        ▼
[Retrieval Router]
        │
   ┌────┴─────────────────┐
   ▼                       ▼
[ANN / Approximate        [Exhaustive GPU Search]
 Nearest Neighbor]         (full corpus scan)
 Fast, lower recall        Slower, higher recall
 Good for simple queries   Required for complex
 with strong signal        queries needing precision
```

At LinkedIn: GPU-accelerated exhaustive retrieval runs over 1.3B documents.
For simpler people-search queries, approximate search with attribute pre-filtering
handles most of the load.

**Structured tag injection into retrieval**: when the QU layer extracts a company
name, inject it directly into the embedding prompt:
```
Without routing output:  embed("software engineer")
With routing output:     embed("software engineer at Google")  ← much more precise
```
This improved Precision@50 by 1.8% and Recall@50 by 2.2% at LinkedIn.

### Layer 3: Reranking Routing

Top-K candidates from retrieval get reranked. Not all K candidates need the most
expensive reranker:

```
Top-250 candidates from retrieval
        │
        ▼
[Confidence Scorer — fast, cheap]
  score each candidate against query
        │
   ┌────┴────────────────────────┐
   ▼                              ▼
High confidence               Low confidence
(clear match or mismatch)     (borderline candidates)
        │                              │
[Accept score from            [Expensive LLM
 lightweight model]            cross-encoder reranker]
```

This is cascade reranking. The expensive cross-encoder only processes candidates
where the cheap model is uncertain. In practice, 70-80% of candidates have
clear scores — only 20-30% need the expensive model.

**Depth routing**: dynamically adjust how many candidates get expensive reranking
based on current load:
```
Low traffic  → rerank top-250 with expensive model
High traffic → rerank top-100 with expensive model, rest with cheap
```
LinkedIn's QU paper mentions a PID-based Dynamic Scoring Depth controller that
adjusts scoring depth per-query from 250 to 130 during peaks (48% less GPU compute).

---

## Part 3: Routing in Offline LLM Inference

Offline removes the two hardest online constraints:
- No latency SLA — you have minutes or hours per task
- All records available upfront — you see the entire dataset before routing

This makes routing both easier AND more powerful in offline.

### Static Routing: Classify First, Then Infer

```
STAGE 1: COMPLEXITY ESTIMATION (fast, cheap, parallelized)
─────────────────────────────────────────────────────────
275K records
    │
    ▼
[Complexity Estimator]  ← runs as preprocessing step, not on GPU
    │                       takes seconds for 275K records
    │
    ├── complexity score per record (0.0 → 1.0)
    │
    ▼
BIN into buckets:
  Simple  (score < 0.3):  165K records  (60%)
  Medium  (score < 0.7):   82K records  (30%)
  Complex (score ≥ 0.7):   28K records  (10%)


STAGE 2: PARALLEL INFERENCE (each bin runs independently)
──────────────────────────────────────────────────────────
Simple bucket  ──▶  [Llama 8B  — 2 GPUs]  ──▶  HDFS
Medium bucket  ──▶  [Llama 70B — 4 GPUs]  ──▶  HDFS
Complex bucket ──▶  [GPT-4o API]           ──▶  HDFS


STAGE 3: MERGE RESULTS
───────────────────────
[Merge + Validate]  ──▶  final output
```

**How to compute complexity score (no model needed):**
- Input token length — single best proxy, fast to compute
- Number of entities/constraints in the text
- Presence of numerical constraints, comparisons, conditionals
- Vocabulary rarity score (sum of inverse frequency of content words)
- Task-specific signals (e.g., for Feed: multi-hop reasoning required?)

### Padding Efficiency Gain from Sorting

This is a hidden benefit of routing in offline that does not exist in online.

GPU processes queries in batches. If queries in a batch have wildly different
lengths, the GPU pads all of them to the longest:

```
WITHOUT routing/sorting:
  Batch: [12 tokens, 450 tokens, 8 tokens, 380 tokens]
  GPU pads all to 450 tokens
  Compute wasted: (450-12 + 450-450 + 450-8 + 450-380) / (4×450) = 57% waste

WITH routing + sorting by length within each bucket:
  Batch: [420 tokens, 430 tokens, 445 tokens, 450 tokens]
  GPU pads all to 450 tokens
  Compute wasted: 2.2% waste
```

Sort each complexity bucket by token length, then batch consecutive records.
This alone can improve GPU utilization by 30-50% on heterogeneous datasets.

### Cascade Routing: Small Model First

An alternative to static routing. Send all records through the small model first,
then escalate uncertain outputs to the large model.

```
275K records
    │
    ▼
[Llama 8B — fast, cheap]
    │
    ├── generates output
    ├── computes confidence score
    │
    ┌─────────────────────────────────────┐
    │         CONFIDENCE CHECK            │
    │                                     │
    │  HIGH confidence (≥ threshold)      │
    │  → accept Llama 8B output           │
    │  (e.g., 80% of records)             │
    │                                     │
    │  LOW confidence (< threshold)       │
    │  → escalate to Llama 70B or GPT-4o  │
    │  (e.g., 20% of records)             │
    └─────────────────────────────────────┘
            │
            ▼
    [Large Model — expensive]
            │
            ▼
    Final output for escalated records
```

**How to measure confidence:**

Token log-probabilities (built into vLLM, no extra model needed):
```
High confidence output:
  "The candidate has strong Python skills." 
  token probs: [0.92, 0.88, 0.91, 0.89, 0.94, 0.90]
  avg log prob: -0.09  ← model was sure of each word

Low confidence output:
  "The candidate [may/might/could] have [some/relevant] skills."
  token probs: [0.45, 0.23, 0.61, 0.38, 0.71]
  avg log prob: -0.87  ← model was hedging at each step
```

Self-consistency (for higher-stakes records, worth the cost):
```
Generate 3 outputs with temperature=0.7
If all 3 agree on key facts → confident, accept
If 3 outputs diverge significantly → uncertain, escalate
```

**Trade-off**: cascade pays double inference cost on escalated records.
Only use cascade when escalation rate is expected to be < 20-25%.
Above that, static routing is cheaper overall.

### Pipeline Architecture for Offline Routing (Flyte/Ray)

```
FLYTE TASK STRUCTURE
─────────────────────────────────────────────────────────────────

Task 1: COMPLEXITY CLASSIFICATION
  Input:  275K raw records (from HDFS)
  Process: compute complexity features per record (CPU-only, fast)
  Output: (record_id, complexity_score, bucket_label) for all records
  Runtime: ~5 minutes for 275K records

         ┌──────────────────────────────────────────────┐
         │           PARALLEL FORK                      │
         └──────┬──────────────────┬────────────────────┘
                │                  │                    │

Task 2a:        │   Task 2b:       │    Task 2c:        │
SIMPLE BUCKET   │   MEDIUM BUCKET  │    COMPLEX BUCKET  │
165K records    │   82K records    │    28K records     │
→ Llama 8B      │   → Llama 70B    │    → GPT-4o API    │
  2 GPUs        │     4 GPUs       │      (pay per use) │
  ~45 min       │     ~90 min      │      ~60 min       │
                │                  │                    │
         ┌──────┴──────────────────┴────────────────────┘
         │           MERGE + VALIDATE
         └─────────────────────────────────────────────▶
                     final 275K results → HDFS
```

The three inference tasks run in parallel. Total wall time ≈ max(45, 90, 60) = 90 min
instead of running all 275K through one model serially.

**Within each inference task, the producer-consumer pattern still applies:**

```
[Tokenizer Process]
  - reads records from assigned bucket
  - computes token length per record
  - sorts by length (for padding efficiency)
  - batches and pushes to queue

         Queue

[GPU Inference Process]
  - drains queue continuously
  - runs vLLM on sorted, uniform-length batches
  - dispatches async HDFS write per completed batch
```

### Model Instance Allocation (Offline)

Before the pipeline runs, decide how many GPU instances to allocate to each model.
This is an offline optimization problem:

Given:
- Total GPU budget: G GPUs available
- B historical calibration batches
- Quality estimates per model per record

Find: how many GPU instances I_j to give each model to maximize throughput × quality
subject to GPU budget G.

Practical heuristic (no formal solver needed):
```
Expected records per model tier:
  Simple  (60%): 165K records → Llama 8B  → needs 2 GPUs for target throughput
  Medium  (30%):  82K records → Llama 70B → needs 4 GPUs
  Complex (10%):  28K records → API       → 0 GPUs (pay per token)

Check: 2 + 4 + 0 = 6 GPUs total ← within budget?
If over budget: reduce medium tier first (quality impact is smaller than complex)
If under budget: add GPUs to medium tier (biggest throughput gain per GPU)
```

---

## Part 4: Monitoring and Feedback Loop

Routing is not a one-time decision. Query distributions shift over time.

### What to Monitor

```
ROUTING HEALTH METRICS
─────────────────────────────────────────────────────────
Routing distribution     % of queries going to each model
                         → if suddenly 90% go to GPT-4o, something wrong

Cost per batch           actual vs budget
                         → alert if > 110% of budget

Quality proxy            downstream metric (NDCG, CTR, task success)
                         per model tier
                         → if quality drops on simple tier, model may have
                           regressed or query distribution shifted

Escalation rate          (cascade routing only)
                         % of records escalated from small to large model
                         → if escalation rate spikes, complexity estimator
                           may be underestimating difficulty

Confidence calibration   predicted confidence vs actual quality
                         → is the confidence score still predictive?
```

### Feedback Loop

```
Routing decisions + model outputs
        │
        ▼
[Quality Evaluation]
  - human eval sample
  - automated judge (LLM-as-judge)
  - downstream metric (task success, engagement)
        │
        ▼
[Label: was routing decision correct?]
  - was simple record actually simple? (model succeeded)
  - was escalation actually needed? (small model would have been fine)
        │
        ▼
[Retrain Complexity Estimator]
  Weekly or when distribution shift detected
        │
        ▼
[Update Routing Thresholds]
  Adjust bucket boundaries based on latest calibration data
```

---

## Summary: When to Use What

| Setting | Routing Type | Why |
|---|---|---|
| Real-time search (QU layer) | SLM unified router (schema-constrained) | Must be < 10ms; routing is part of model output |
| Real-time reranking | Cascade (cheap scorer first) | Expensive model only for borderline candidates |
| Offline inference, clear difficulty signal | Static routing (classify upfront) | Simple, parallelizable, enables length sorting |
| Offline inference, unclear difficulty | Cascade (small model first) | Let model confidence determine escalation |
| Mixed fleet (on-prem + cloud API) | Batch-level ILP | Enforces hard cost + GPU capacity constraints |
| High adversarial batching risk | Robust ILP (lower bound estimates) | Prevents worst-case cost blowout |

---

## Key System Design Principles

1. **Routing must be cheaper than the problem it solves.** A 50ms routing decision
   on a 100ms inference call is not a win. Keep estimators lightweight.

2. **Sort within routing buckets.** In offline, sorting by length before batching
   is one of the highest-ROI optimizations — free throughput gain, no quality cost.

3. **Always have a fallback.** Every self-hosted model needs an API fallback.
   Circuit breakers prevent cascading failures when a model tier goes down.

4. **Cascade only when escalation rate is low.** If more than 25% of records
   escalate, just route statically — you're paying for two model calls on too many records.

5. **Route at every layer.** In semantic search, routing happens at QU, retrieval,
   and reranking independently. Each layer has different cost-quality tradeoffs.

6. **Monitor routing distribution, not just quality.** A shift in routing
   distribution (more records going to expensive model) is an early signal of
   query distribution shift, model regression, or estimator drift.
