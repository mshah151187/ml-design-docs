# ML Serving Infrastructure Optimizations

These patterns operate **above the GPU layer** — they control when the model runs, how
deeply it runs, and how load is distributed over time. They are not specific to any one
system. Any retrieval-ranking pipeline (semantic search, feed ranking, ads, recommendations,
PYMK) with an expensive model and variable traffic faces the same problems and can apply
the same patterns.

**Source:** LinkedIn Semantic Search paper (Borisyuk et al., 2026, §2.5.5 Middle Tier
Serving Optimizations). The numbers cited are from that paper but the patterns are generic.

---

## Pattern 1 — Result Caching

Relevance scores are deterministic: same searcher + same query + same candidate always
produces the same score for a fixed model version. Cache the result.

```
Key:   hash(searcher_id, query_signature, entity_id)
Value: relevance_score (float)

Request arrives:
  → probe distributed cache (e.g. Couchbase, Redis)
  → HIT  (>50% of requests): return cached score, skip GPU entirely
  → MISS: run ranker, write result to cache
```

**Why hit rate is high:** users repeat similar searches; the same items appear across
many searchers with similar profiles. The (searcher, query, entity) space has significant
overlap in practice.

**Impact (LinkedIn Semantic Search):** >50% requests served from cache →
median/mean latency -8–10%, tail latency also improves (fewer GPU requests = less queuing).

**Applies to:** any ranking system where scores are deterministic and query/item space
has repetition — feed ranking (same posts shown to many users), ads (same ad scored for
similar users), recommendations, PYMK.

**Invalidation:** cache entries must be invalidated on model updates or item/profile changes.
TTL-based expiry is the simplest approach; event-driven invalidation (CDC from item store)
is more precise.

---

## Pattern 2 — Retry Policy with Latency Budget

Under bursty load, a GPU worker may be slow to respond. Without a retry policy, the
request waits indefinitely — wasting latency budget and possibly timing out.

A latency budget retry enforces a deadline: if the primary worker hasn't responded within
T ms, issue a retry to a different worker. First response wins, the slow one is cancelled.

```
t=0ms:   request → worker A
t=50ms:  worker A still processing → retry → worker B
t=80ms:  worker B responds → return result, cancel worker A
```

Prevents one overloaded worker from becoming a cascading bottleneck. Redundant requests
occasionally run on two workers, but the net effect is better throughput by eliminating
stalled compute waiting for slow workers.

**Impact (LinkedIn):** +10% throughput by reducing redundant stalled GPU computation.

**Applies to:** any system with multiple GPU workers behind a load balancer — feed
ranking, ads serving, search ranking.

**Tuning:** T (retry threshold) must be set below p99 of normal latency to avoid
unnecessary retries under normal load. Typically T = p90 of healthy latency.

---

## Pattern 3 — PID-based Dynamic Scoring Depth

Ranking depth (number of candidates scored by the expensive model) is a tunable parameter
at serving time. Under peak load, reducing depth lowers GPU compute per query — at the
cost of ranking quality on the tail candidates.

```
Off-peak:  depth = 250 candidates → better ranking quality
Peak load: depth = 130 candidates → 48% less GPU compute per query
```

**PID controller mechanics:**
```
Error signal:   current GPU queue depth vs target queue depth
Control signal: scoring depth per query (increase/decrease)
Feedback loop:  depth change → GPU load changes → error signal updates

Thermostat analogy:
  GPU overloaded → reduce depth → GPU cools down
  GPU underutilized → increase depth → better quality
```

A PID controller reacts continuously and smoothly — no abrupt step changes that cause
oscillation. The system self-stabilizes under arbitrary load patterns without manual
re-tuning for each traffic shape.

**Impact (LinkedIn):** peak depth drops from 250 to 130 (48% less per-query GPU compute).

**Applies to:** any ranking system where candidate pool size is tunable at serving time
— feed ranking (how many posts to score), ads (how many creatives to rank), PYMK (how
many candidates to rerank), semantic search (how many retrieved candidates to rerank).

**Alternative approaches:**
- Static threshold: switch depth at fixed QPS levels — simpler but oscillates at boundaries
- ML-based controller: predict required depth from query features — higher quality but
  adds latency and complexity
- PID: good balance — no offline training needed, stable under smooth load changes

---

## Pattern 4 — Traffic Shaping (Defer Latency-Insensitive Requests)

Not all requests have the same urgency. Real-time queries need a response in <500ms.
Nearline re-scoring (model refresh, offline evaluation, A/B logging) can wait seconds
or minutes.

Traffic shaper classifies requests by urgency and defers low-priority work into idle
GPU windows:

```
Peak window:  only real-time queries execute → GPU fully occupied by urgent work
Idle window:  deferred nearline/batch jobs fill the gap → GPU never sits idle
              (model refresh re-scores, A/B experiment scoring, feature backfills)
```

**Impact (LinkedIn):** +25% effective GPU throughput — GPU utilization increases by
filling gaps that would otherwise be empty between real-time bursts.

**Applies to:** any system with mixed real-time and batch workloads on shared GPU
infrastructure — feed ranking (online vs offline scoring), recommendation model refresh
pipelines, embedding index rebuild, A/B experiment scoring.

**Implementation:** priority queue at the GPU worker. High-priority items preempt or
are scheduled ahead of low-priority items. Deferred items get a deadline (must complete
within X minutes) to prevent indefinite starvation.

---

## LLM-as-Ranker Optimizations

These two optimizations apply specifically when an LLM is used as a **scorer or ranker**
rather than a generator. The model reads a prompt and produces a relevance score — it
never generates output tokens. That constraint unlocks two large speedups.

---

### Pattern 5 — Scoring-Only Prefill (Skip Decode)

Standard LLM inference has two phases:
- **Prefill**: process all input tokens in parallel → compute KV cache
- **Decode**: generate output tokens one at a time (autoregressive, sequential)

For ranking tasks you need one number — a logit or probability at the last prompt token.
You do not need any generated output. So skip the decode step entirely.

```
Standard generation:
  Prefill([system, query, candidate]) → KV cache
  Decode → token 1 → token 2 → token 3 → ... → "Yes, this is relevant"
  Cost: prefill + N × decode steps

Scoring-only:
  Prefill([system, query, candidate, score_token]) → KV cache
  Read logit at position[-1] → relevance score
  Cost: prefill only
```

**Why the saving is large:** each decode step is a full forward pass through the model,
serially dependent on the previous token. For a 0.6B model producing even a 5-token
response, that is 5 serial forward passes. Scoring-only is 1 prefill pass — the cheapest
possible operation.

Decode latency scales with output length. Prefill latency is fixed for a fixed prompt
length. For ranking tasks the output is always 1 number — the win is always the same size.

**Applies to:** any system using an LLM to assign scores instead of generate text —
feed post ranking, ad creative ranking, PYMK candidate reranking, document reranking in
RAG pipelines, cross-encoder reranking in search.

**Impact (LinkedIn Semantic Search):** one of the primary contributors to the 75×
throughput gain over a naive generation-based scoring baseline.

---

### Pattern 6 — Shared Prefix KV Cache Reuse

When scoring N candidates against a common context, that context is recomputed N times
in a naive implementation. The shared prefix contains the same tokens for every candidate
— system prompt, query, user profile. Computing it once and reusing the KV tensors
eliminates N−1 redundant forward passes over that prefix.

```
Naive (N=250 candidates):
  Prefill([system + query + profile + candidate_1]) → KV cache → score_1
  Prefill([system + query + profile + candidate_2]) → KV cache → score_2   ← recomputes prefix
  ...
  Prefill([system + query + profile + candidate_250]) → KV cache → score_250
  Cost: 250 × (prefix_tokens + suffix_tokens)

Shared prefix:
  Prefill([system + query + profile]) → KV_prefix  ← computed once
  For each candidate:
    Prefill([candidate_i], kv_prefix=KV_prefix) → score_i   ← only suffix tokens
  Cost: 1 × prefix_tokens + 250 × suffix_tokens
```

**Why this is large:** prefix is typically long (system prompt + query + user features),
suffix is short (1 candidate description). If prefix = 99 tokens, suffix = 1 token
(MixLM case), then naive cost is 250×100 = 25,000 token-steps; shared prefix cost is
100 + 250×1 = 350 token-steps. **71× reduction.**

Even with longer suffixes (e.g. 20 tokens), a 100-token prefix saves:
- Naive: 250 × 120 = 30,000
- Shared: 100 + 250 × 20 = 5,100
- **~6× reduction**

**Requirement:** all candidates must share an identical prefix — same bytes, same token
IDs. Any per-candidate data must be in the suffix. This constrains prompt design: user
profile and query go in the shared prefix; candidate description goes in the per-candidate
suffix.

**Applies to:** any multi-candidate ranking with a common context — feed (same user
profile × N posts), ads (same user × N creatives), PYMK (same viewer × N candidates),
RAG reranking (same query × N retrieved chunks).

**Impact (LinkedIn Semantic Search):** combined with scoring-only prefill, reduced
effective compute per query from ~250 full forward passes to ~1 shared prefix pass +
250 single-token suffix passes. Core driver of the 75× throughput gain.

---

### Patterns 5 + 6 Together

These two optimizations are multiplicative — each independently reduces cost, but they
stack:

```
Baseline (generation + no prefix sharing):
  250 candidates × (prefill + decode) = very expensive

After scoring-only (Pattern 5):
  250 × prefill_only = cheaper

After shared prefix (Pattern 6):
  1 × prefix_prefill + 250 × suffix_prefill = much cheaper

Combined:
  prefill cost reduced to the short-suffix contribution only
```

The reason LinkedIn could use a 0.6B LLM as a reranker at tens of thousands QPS is
that patterns 5 + 6 together reduce per-query GPU compute to near the minimum physically
possible for a transformer model.

---

## The Four Midtier Patterns Together

| Pattern | Fires when | What it saves |
|---|---|---|
| Result cache | Always — probe first | Skip GPU for >50% requests |
| Retry with latency budget | Bursty load | Eliminate stalled compute, +10% throughput |
| PID depth controller | Peak load | 48% less compute per query (250→130 depth) |
| Traffic shaping | Idle windows | Fill GPU gaps with deferred work, +25% GPU util |

These stack independently — each targets a different inefficiency:
- Cache eliminates redundant computation
- Retry eliminates wasted wait time
- Depth control reduces per-query cost under load
- Traffic shaping maximizes GPU utilization across time

**The common thread:** GPU inference is the most expensive resource. Every technique
here is about using it less (cache, depth control), using it better (traffic shaping),
or protecting it from cascading failure (retry policy).

---

## Where These Show Up

| System | Cache | Retry | Depth control | Traffic shaping |
|---|---|---|---|---|
| LinkedIn Semantic Search | ✓ Couchbase | ✓ latency budget | ✓ PID 250→130 | ✓ +25% GPU util |
| Feed Ranking | ✓ post scores per user | ✓ | ✓ feed depth | ✓ offline re-scoring |
| Ads Ranking | ✓ ad-user scores | ✓ | ✓ auction depth | ✓ batch scoring |
| Recommendations | ✓ item scores | ✓ | ✓ candidate pool | ✓ model refresh |
| PYMK | partial (less repetition) | ✓ | ✓ | ✓ |
