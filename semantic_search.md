# Semantic Search at LinkedIn — ML System Design Reference

Source: "Semantic Search At LinkedIn" (Borisyuk et al., Feb 2026, arXiv:2602.07309)  
System: LinkedIn AI Job Search + AI People Search  
Retrieval foundation: LiNR (Borisyuk et al., CIKM 2024)

---

## System at a Glance — All Key Numbers

### Scale
| Dimension | Number |
|---|---|
| Corpus size | 1.3B+ documents (jobs + member profiles) |
| Queries per second | Tens of thousands (QPS) |
| Retrieval output | Top-1,000 candidates per query |
| Ranking input | Top-250 (SLM reranks top 250 from retrieval) |
| End-to-end latency budget | p99 ≤ 500ms |
| LiNR full-scan latency | 4ms for 15M–1B entries on GPU |

### Model Sizes
| Model | Params | Role |
|---|---|---|
| SAGE LLM oracle | 8B | Offline relevance judge — generates training labels |
| Relevance Teacher | 1.7B | Trains on oracle signal, used for student distillation |
| Engagement Teacher | 1.7B | Trains on action logs, used for student distillation |
| Student SLM (dense) | 0.6B | Live ranker — predicts relevance + engagement |
| Student SLM (pruned) | 375M | Production ranker after OSSCAR pruning |
| RL Summarization model | 1.7B | Compresses member profiles + job descriptions |
| Embedding model | 0.6B–4B | Bi-encoder for retrieval |

### Throughput (375M pruned SLM, single H100, p99 ≤ 500ms)
| Optimization stage | Throughput (items/s/GPU) | Incremental gain |
|---|---|---|
| Baseline | 750 | — |
| + Batch tokenization | 900 | +150 |
| + Scoring-only prefill | 1,300 | +400 |
| + Python/runtime (multi-process gRPC) | 1,600 | +300 |
| + In-batch prefix caching (IBPC) | 2,000 | +400 |
| + Piecewise CUDA graph | **2,200** | +200 |
| MixLM (text-embedding mix) | **22,000** | ~76× over raw-text SLMs |
| **Overall claimed gain** | **75× over naive** | under fixed latency |

### Inference Optimizations Impact
| Optimization | Gain |
|---|---|
| Context summarization (Job Search) | 4× throughput improvement |
| Context summarization (People Search) | 95th-pct prompt: 1,500 → 500 tokens |
| Model pruning (600M → 375M) | 290 → 2,200 items/s/GPU (7.5×) |
| Shared prefix amortization (IBPC) | +25% throughput (1,600 → 2,000) |
| Scoring-only prefill (no decode) | +44% throughput (900 → 1,300) |

### Training Data Scale
| Dataset | Scale |
|---|---|
| Job Search labeled pairs | 200k → 8M (40× expansion to saturation) |
| People Search | Scaled until quality plateau observed |

### Quality Results (Production A/B)
| Metric | Job Search | People Search |
|---|---|---|
| NDCG@10 improvement | +7.73% | >10% |
| Poor Match Rate@10 (PMR@10) | -46.88% | — |
| DAU lift | — | +1.2% |

### Ranking Model Quality (Ablation, Job Search, Table 1)
| Technique stack | NDCG@10 |
|---|---|
| Open source SLM baseline | ~0.76 |
| + Domain reasoning + soft-label SFT + ranking loss | 0.877 |
| + Relevance Teacher distillation | 0.895 |
| + 40× data volume | **0.943** |

### Retrieval Quality (People Search, Table 9)
| Model | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| Baseline 4B | 0.33 | 0.70 | 0.71 |
| + GPU RAR | **0.47** | **0.79** | **0.79** |

### Training Infrastructure
| Optimization | Saving |
|---|---|
| Online multi-teacher distillation | 3× speedup in distillation |
| Offline multi-teacher distillation (cached) | 35% training time reduction, 25% GPU-hours saved |
| Agentic GPU optimization (BFS + RAG config tuning) | 13% training time reduction (256 GPU-hours on 64 H100s) |
| LiGer (gradient + optimizer state sharing) | 2× larger batch sizes, reduced memory |
| FSDP2 + multi-node (up to 3.5×) | Additional 30% reduction |

---

## 1. Why Semantic Search? The Keyword Gap

Keyword-based retrieval matches tokens. A query "software engineer machine learning" only
retrieves documents containing those exact words. It misses:
- "ML engineer", "applied scientist", "AI researcher" (same intent, different tokens)
- Implicit constraints: seniority from profile, location from filter, network distance

**Semantic search** retrieves by meaning. It maps queries and documents into a shared
embedding space and retrieves by cosine similarity — "ML engineer" and "machine learning
engineer" land near each other even without token overlap.

The challenge at LinkedIn scale: tens of thousands of queries per second, billions of
documents (jobs/members), strict latency budgets (p99 ≤ 500ms). LLM cross-encoders
give the best quality but are too slow for this scale. The paper's answer: a
**two-stage pipeline** — fast embedding retrieval + efficient SLM reranker.

---

## 2. Retrieval and Ranking Architectures

Before understanding LinkedIn's system, it helps to know the three mechanisms in play
and why each exists.

### Bi-Encoder (= Two-Tower with LLM encoder)

Same architecture as the two-tower model from `embeddings.md`, just with an LLM inside
each tower instead of a shallow MLP:

```
RecSys world (Two-Tower):           IR/NLP world (Bi-Encoder):

member features ──→ tower_u ──→ e_u      query ──→ encoder_q ──→ e_q
                                   dot                               cos
  item features ──→ tower_v ──→ e_v      doc   ──→ encoder_d ──→ e_d
```

Different names, same structure — the two sides encode **independently**. The score is
a single dot product / cosine at the end. Query and document never see each other during
encoding.

**Why this enables retrieval at scale:**
The document side `e_d` can be pre-computed offline for all 1B+ documents and stored in
an index. At query time you only encode the query (`e_q`) and do a nearest-neighbor scan
over the pre-built index. This is fast regardless of how many documents exist.

**The bottleneck:** all information about a document is compressed into one fixed-size
vector (e.g., 768 dims) before it ever sees the query. The vector must represent the
document generically — it cannot specialize for "how relevant is this to *this specific
query*." Nuance is lost.

| Era | Encoder inside each tower |
|---|---|
| Classic Two-Tower | Shallow MLP over handcrafted features |
| LLM Bi-Encoder | Full LLM (BERT/GPT etc.) — raw text → embedding |

The LLM variant reads raw text directly, removing the need for handcrafted features and
capturing richer semantics. But the independent-encoding constraint stays.

---

### Cross-Encoder

Query and document are **concatenated** and fed together through a single model:

```
[query + document] ──→ LLM ──→ score
```

Query and document tokens see each other in the same sequence — the model can reason at
token level: "this query says 'senior role' and this document says 'lead a team of 10'
— alignment." A bi-encoder can never do this because the two sides encode separately.

**Important nuance on attention direction:** the SLM is a **decoder-only** model
(causal mask), not fully bidirectional like BERT. The sequence is laid out as:

```
[query tokens] [document tokens]
   position 1..Q   position Q+1..Q+D
```

- Query tokens at positions 1..Q can only attend to earlier positions — they **cannot**
  attend to document tokens (those come later in the sequence).
- Document tokens at positions Q+1..Q+D can attend to all earlier tokens — they **can**
  attend to all query tokens.

So the attention is **one-directional**: document → query. The model reads the query
first (building query context into the KV cache), then reads the document tokens while
attending back to every query token. In practice this gives strong cross-encoder quality
— the critical relevance signal flows from document → query, which is the direction
that matters: "given this job description text, how well does it match the query terms?"

**Why it gives the best quality:** No fixed-size bottleneck. The full document text is
available when processing each specific query. It captures:
- Subtle token-level alignment ("ML engineer" ↔ "machine learning lead")
- Context-dependent relevance (same doc relevant for one query, not another)
- Long-range reasoning within the document

**Why it cannot be used for retrieval at scale:**

```
Cost per query = O(num_candidates × seq_len²)

seq_len = len(query) + len(document) = 50 + 500 = 550 tokens
attention FLOPs ∝ 550² = 302,500 per candidate

For 1B documents: 302,500 × 1,000,000,000 → completely infeasible
```

You cannot pre-compute anything — the document encoding depends on the query. So
cross-encoders are only practical after retrieval has narrowed the candidate set down
to hundreds.

---

### LLM as Oracle / Judge (Offline Only)

An 8B+ LLM (like LinkedIn's SAGE) is the strongest ranker — it can reason about
relevance with full language understanding. But at tens of thousands of QPS it is
economically infeasible to run online.

**Solution: use it offline as a teacher.** Generate relevance grades for training pairs,
then distill that quality into a smaller model (0.6B SLM) that can serve at production
QPS. The oracle's quality transfers via knowledge distillation; its compute cost does not.

---

### Why Two Stages?

These three mechanisms have a quality-vs-speed tradeoff that motivates a pipeline:

| Mechanism | Quality | Speed | Used for |
|---|---|---|---|
| Bi-Encoder / Two-Tower | Medium (no query-doc interaction) | Very fast (pre-computed) | Retrieval: 1B → 1000 |
| Cross-Encoder / SLM | High (full query-doc attention) | Slow (no pre-compute) | Ranking: 1000 → 250 |
| LLM Oracle (8B+) | Highest | Too slow for online | Offline teacher / label generation |

Stage 1 (bi-encoder) trades quality for speed to narrow the candidate space.
Stage 2 (SLM cross-encoder) applies full attention only on the shortlist — affordable
because the candidate count is now small.

---

## 3. The 4 Models at Play

LinkedIn's semantic search system has four distinct model tiers. Understanding what
each does, how it is trained, and when it runs is the key to understanding the whole
system. (Figure 1 in the paper shows all four feeding into the SLM prompt.)

→ See `distillation.md` for a deep dive on the distillation mechanism.

---

### Model 1 — SAGE Oracle (8B)

**What it is:** A large language model serving as the policy-aligned relevance judge.
It defines what "relevant" means for LinkedIn search — not just semantic overlap but
the full product policy (seniority match, location, skills, marketplace health).

**How it is trained:**
- Base: open-source 8B LLM
- Fine-tuned on human-labeled relevance precedents curated by LinkedIn policy teams
- Produces ordinal grades {1, 2, 3, 4} with natural language reasoning

**Input / Output:**
```
Input:  system prompt ("you are a job relevance judge") +
        query text + candidate document (structured attributes + text)
Output: grade {1,2,3,4} + reasoning text
        e.g. grade=1, "seniority mismatch: query=senior, job=junior"
```

**When it runs:** offline only — never on live traffic. Runs in batch over search
traffic logs to annotate (query, document) pairs with relevance grades. These
annotated pairs become the training data for everything else.

**Why not serve it directly:** at tens of thousands QPS with p99 ≤ 500ms budget,
an 8B model is economically infeasible. Its quality is transferred to smaller models
via distillation.

---

### Model 2 — Teacher Models (1.7B × 2)

Two specialized teachers — one per objective — trained on oracle signal and user logs.

**Relevance Teacher (1.7B):**
- Trained directly on oracle grades via soft-label SFT + domain reasoning distillation
  + ranking loss
- Maps oracle grades {1,2,3,4} → calibrated Yes/No soft probabilities
- Output: relevance probability per (query, doc) pair

**Engagement Teacher (1.7B):**
- Trained on real user action logs: click, apply, dismiss, shortlist, connect, follow, message
- Predicts engagement probability per action type
- Uses loss masking for rare actions (follow, message) to avoid probability collapse

**How they interact with the student:**
Teachers are run on the training dataset to generate soft label targets. The student
learns to match these distributions via KL divergence loss — not to match hard 0/1
labels, not to match the oracle grades directly.

```
Training batch → Relevance Teacher → soft relevance labels ──┐
Training batch → Engagement Teacher → soft engagement labels ─┤→ Student update
                                                              ┘
```

**When they run:** training time only (online per batch, or pre-computed offline and
cached). Never serve live traffic. LinkedIn uses offline caching to eliminate live
teacher inference → 35% training time reduction.

---

### Model 3 — Student SLM (0.6B → 375M pruned)

**What it is:** The production ranker. A 0.6B transformer cross-encoder distilled from
both teachers, then pruned to 375M for serving.

**How it is trained — 3 phases:**
```
Phase 1 — Relevance specialization:
  Warm-start from open-source SLM
  Train with relevance teacher only (single-teacher distillation)
  → strong relevance baseline

Phase 2 — Multi-teacher distillation (MTD):
  Warm-start from Phase 1 relevance SLM
  Distill from BOTH teachers simultaneously
  Loss = w_rel × KL(student || rel_teacher) + w_eng × KL(student || eng_teacher)
  → learns relevance + engagement jointly

Phase 3 — Pruning:
  OSSCAR structured pruning: remove 50% MLP neurons + final 8 transformer layers
  600M → 375M params
  Fine-tune to recover quality → matches or slightly exceeds dense baseline
```

**Input / Output:**
```
Input:  [system prompt][query text][candidate: title, company, location, seniority, text]
Output: Yes/No first-token logit → relevance_prob + engagement_probs (click, apply, etc.)
        e.g. relevance_prob=0.91, click_prob=0.43, apply_prob=0.21
```

**When it runs:** every live request — scores top 250 candidates from retrieval.

---

### Model 4 — RL Summarization Model (1.7B)

**What it is:** A 1.7B language model fine-tuned with reinforcement learning (GRPO) to
compress long, heterogeneous member profiles and job descriptions into short, factual,
engagement-predictive summaries before they are fed to the SLM ranker.

**Why it exists — and why summaries are needed at all (not just the bi-encoder embedding):**

A common point of confusion: the bi-encoder already compresses each document into a
768-dim embedding for retrieval. Why do we need a text summary too?

The answer is that the bi-encoder embedding and the summary serve completely different
stages and completely different models:

```
Stage 1 (retrieval) — uses bi-encoder embedding:
  doc text → bi-encoder → e_d (768-dim vector)  → stored in ANN index
  At query time: dot product e_q · e_d → top-1000 doc IDs

Stage 2 (ranking) — uses text summary:
  top-1000 doc IDs → fetch content by ID → inject summary text into SLM prompt
  SLM reads actual text tokens (not vectors) and attends across query + document
```

The SLM is a cross-encoder — it needs **text tokens** to attend to. If you gave it
the 768-dim embedding instead of text, it would have nothing to perform attention over.
The entire point of the cross-encoder is token-level interaction between query and
document text. The bi-encoder embedding is useless to it.

So summarization matters because: without it, the SLM prompt would include the full
raw document (~1,500 tokens). With a summarizer: ~500 tokens. That is a 3× prompt
reduction → ~4× throughput gain. And if summarization ran at query time for 250
candidates you'd be invoking a 1.7B model 250× per request — infeasible. Running it
nearline and caching the result is what makes it practical.

Inference cost scales linearly with context length. Raw member profiles can be
~1,500 tokens (headline, experience, skills, activity history, connections). Feeding
that directly to the SLM ranker is expensive and the noise hurts quality. A dedicated
summarizer compresses this to ~500 tokens with negligible quality loss.

**How it is trained — GRPO with a composite reward:**
```
R(s) = 𝟙[â_n = a_n] × (1 - λ_len · ℓ(s) + λ_qual · q(s))

Where:
  ℓ(s) = normalized length penalty (shorter → less penalty)
  q(s) = factuality/saliency score from a 32B judge model
  â_n  = predicted action, a_n = actual action (engagement correctness)
```
The summarizer learns: produce short summaries that are factually correct AND that
improve the engagement teacher's prediction accuracy. This is RL, not supervised learning
— there is no "correct summary" label, only a reward signal.

**Input / Output:**
```
Input (People Search — member profile, ~1500 tokens):
  Name: Alice Chen | Headline: Senior ML Engineer at TechCorp
  Summary: Passionate about NLP and large-scale ML systems...
  Experience: [3 roles, 8 years total — last 3 at TechCorp doing LLM fine-tuning]
  Skills: [47 skills]
  Activity: [last 10 jobs interacted with: 7 ML engineer, 3 data science]

Output (~100 tokens):
  Senior ML Engineer, 8 years exp, specializes in LLM fine-tuning and NLP.
  Active job seeker (ML engineer roles). Strong ML/AI network.
```

**When it runs:** nearline — not per query, but per document update. Summaries are
pre-computed when profiles or job descriptions are updated and cached. At serving time
the SLM ranker reads the pre-computed summary, not the raw document.

**Impact:**
- Job Search: ~4× throughput improvement (documents compressed ~4×)
- People Search: 95th-percentile prompt 1,500 → 500 tokens
- +4.40% Connect AUROC, +11.60% Message AUROC (People Search, Table 3 in paper)

**Key distinction from the Oracle and Teachers:** the summarization model runs at
nearline serving time (its output is consumed per live request), while the Oracle and
Teachers only run during training. The summarization model's output is part of the
live SLM ranker's prompt.

---

### How All 4 Work Together

```
OFFLINE (training pipeline):
─────────────────────────────────────────────────────
Search traffic logs (query, document) pairs
        ↓
  SAGE Oracle (8B)  →  grades {1,2,3,4}
        ↓                     ↓
  Relevance Teacher (1.7B)    Bi-Encoder training
  trains on oracle grades     (oracle labels → pos/neg pairs)
        ↓
  Relevance Teacher soft labels
        ↓
  Student SLM (Phase 1 — relevance only)
        ↓
  Multi-Teacher Distillation (Phase 2)
  [Relevance Teacher + Engagement Teacher → Student]
        ↓
  Pruning → 375M Student SLM (production model)

NEARLINE (per document update — not per query):
─────────────────────────────────────────────────────
Member profile / job description updated
    ↓ RL Summarization Model (1.7B)
    → compressed summary (~500 tokens) cached in document store
    → Bi-Encoder produces document embedding → stored in retrieval index

ONLINE (serving a live search request):
─────────────────────────────────────────────────────
User query
    ↓ QU Layer (structured attribute extraction)
    ↓ LiNR: ABM pre-filter + Bi-Encoder (query tower) + GPU RAR → top-1000 doc IDs
         [bi-encoder embedding used ONLY here — returns IDs, not embeddings, to next stage]
    ↓ Fetch content: look up pre-computed summaries by doc ID (from feature store)
         [SLM needs text tokens, not vectors — summary text is what goes in the prompt]
    ↓ Student SLM (375M) scores top-250
        prompt = [system][query][cached summary text ~500 tok][structured features]
        → relevance_prob + engagement_probs per candidate
    ↓ Calibration layer → position-aware scores
    ↓ Business logic + policy enforcement
    ↓ Final ranked results

Oracle: not running
Teachers: not running
RL Summarizer: not running per query (output already cached)
Only Student SLM + Bi-Encoder (query tower) run per live request.
```

---

## 4. System Architecture — Two Stages

```
Search Query
     ↓
Query Understanding (QU) Layer
     ↓ deterministic signals (routing, normalized attributes, query reformulations)
     ↓
┌─────────────────────────────────────┐
│  Stage 1: Retrieval                 │
│  LLM bi-encoder (embedding-based)   │
│  GPU exhaustive scan of 1B+ docs    │
│  → top-K=1000 candidates           │
└─────────────────────────────────────┘
     ↓ top 1000
┌─────────────────────────────────────┐
│  Stage 2: Ranking                   │
│  SLM (0.6B) reranker               │
│  scores top 250, jointly predicts  │
│  relevance + engagement            │
└─────────────────────────────────────┘
     ↓
Business logic + policy enforcement
     ↓
Final results
```

**Why two stages?** Cross-encoders (query × document full attention) give best quality
but scale as O(K × seq_len²). You can't run them on 1B documents. So:
- Stage 1 (retrieval): fast approximate search, high recall — finds the right 1000
- Stage 2 (ranking): expensive model, only on 1000 candidates

### Query Understanding (QU) Layer

Before retrieval, raw text queries are converted into deterministic, machine-interpretable
signals: routing decisions (Job Search vs People Search), normalized attributes, query
reformulations. This creates a **stable semantic contract** — all downstream stages
(retrieval, ranking, filtering) see the same clean signal regardless of how the user
typed the query.

QU is also what makes ABM (Attribute-Based Matching) in LiNR work. ABM needs structured
attributes to build pre-filter masks — it cannot operate on raw query text. QU extracts
those attributes first, then hands them to ABM:

```
Raw query: "senior machine learning engineer remote TensorFlow"
      ↓ QU Layer
{
  seniority: "senior",
  role:      "ML engineer",
  modality:  "remote",       ← becomes ABM hard filter: exclude all on-site jobs
  skills:    ["TensorFlow"]  ← becomes ABM clause: at least one skill must match
}
      ↓ ABM pre-filter (LiNR)
Candidate pool with hard constraints already eliminated
      ↓ Bi-encoder semantic similarity
      ↓ GPU RAR personalization reweighting
      ↓ Top-1000 → SLM ranker
```

Without QU, ABM has nothing to filter on — you'd have to infer structure from raw
text at retrieval time, which is expensive and error-prone. QU is what translates
fuzzy natural language into a deterministic contract that ABM, retrieval, and ranking
all consume reliably.

---

## 5. Stage 2 — SLM Ranking

### The Problem with Pure Relevance

A ranking model that only predicts relevance ("is this job right for this person?")
misses engagement. A job can be highly relevant but unclickable if the title is
misleading. A job can be clicked often but actually irrelevant (bait-and-click).
Production ranking must balance relevance AND engagement.

### Architecture: Multi-Teacher Distillation

The 0.6B student SLM is trained via **multi-teacher distillation (MTD)**:

```
              8B SAGE LLM "oracle"
              (policy-aligned judge)
                     ↓
         ┌───────────┴────────────┐
         ▼                        ▼
  Relevance Teacher         Engagement Teacher
      (1.7B)                    (1.7B)
  trains on oracle           trains on action logs
  relevance signal           (clicks, applies, etc.)
         └───────────┬────────────┘
                     ▼
              Student SLM (0.6B)
        jointly predicts relevance + engagement
        (multi-teacher distillation)
```

The student distills from both teachers simultaneously via **KL divergence losses**
(soft labels, not hard 0/1 labels). Soft labels preserve the teacher's uncertainty
near decision boundaries — a job that's borderline relevant gets a probability of 0.6,
not a hard 1, and the student learns that nuance.

---

## 6. Relevance Training — Four Ingredients

The SAGE 8B model serves as the "oracle" — a policy-aligned judge that given a query
and candidate document (structured attributes + selected text) produces ordinal
relevance grades {1,2,3,4}. Everything else distills from this oracle.

### Ingredient 1 — Domain Reasoning Distillation

Initialize from an intermediate base model. Distill from the oracle via **distribution
matching** (forward/reverse KL). The oracle outputs reasoning text alongside grades —
this transfers domain-specific decision rules to the student.

Example: the oracle learns "if location=remote AND seniority=senior AND skills overlap
is high → grade 4". The KL loss transfers this decision logic even without explicit rules.

### Ingredient 2 — Soft-Label Fine-Tuning

Map oracle ordinal grades {1,2,3,4} to continuous [0,1] probabilities and train binary
Yes/No prediction (no rationales at inference). Key: the **mapping function matters**.

| Mapping | NDCG@10 |
|---|---|
| Linear → sigmoid | 0.8608 |
| Sigmoid mapping | 0.8632 (+0.28%) |

Sigmoid mapping better preserves the distribution shape near the decision boundary.

The student outputs first-token Yes/No logits and converts to probabilities — this
matches the serving interface exactly (no extra decode steps).

### Ingredient 3 — Ranking Loss (List-wise)

For each query, include top-k EBR (embedding-based retrieval) documents plus k random
documents. Form ordered pairs and optimize a **pairwise ranking loss**:

```
L_rank = -Σ log σ(score(q, d+) - score(q, d-))
```

Where d+ ranks above d- according to oracle. This goes beyond pointwise prediction to
directly optimize the ordering of candidates.

### Ingredient 4 — Chat-Template Inference Interface

System prompt specifies the binary relevance question. User message combines query +
compact candidate representation (title, company, location, seniority, curated text).
At inference, extract first-token Yes/No logits only.

**Why compact representation?** Inference cost scales linearly with context length.
Full job descriptions (~1500 tokens) are compressed to ~500 tokens via an RL
summarization model with negligible quality loss.

### Cumulative Impact (Job Search):

| Technique | NDCG@10 | Δ vs baseline |
|---|---|---|
| Open Source SLM | — | — |
| + Soft-label SFT (linear) | 0.7583 | baseline |
| + Domain reasoning | 0.8500 | +12.09% |
| + Schedule-free LR tuning | 0.8608 | +13.52% |
| + Soft-label SFT (sigmoid) | 0.8632 | +13.83% |
| + Chat template | 0.8718 | +14.97% |
| + Ranking loss | 0.8772 | +15.68% |
| + 0.5B → 0.6B model | 0.8910 | +17.50% |
| + Relevance Teacher distillation | 0.8950 | +18.03% |
| + 40× data volume | **0.9432** | **+24.48%** |

---

## 7. Multi-Task Engagement Teacher

The engagement teacher predicts user actions: click, apply, dismiss, badfit, shortlist
(Job Search); long-dwell, connect, follow, message (People Search).

### Loss Masking for Rare Actions

In People Search, follows and messages are far sparser than clicks/connects. If you
train all heads with independent pointwise losses, the model sees rare actions mostly
as negatives (no interaction = not engaged), which drives predicted probabilities to
near-zero and collapses rare-action scores.

**Loss masking fix:** treat a document as negative for a given action **only when at
least one document received that action for the same query**. This generalizes
hard-negative training to multi-objective settings.

```
Query: "data scientist"
  doc_A: clicked ✓, applied ✓
  doc_B: clicked ✓, not applied
  doc_C: not clicked

Loss for "apply" head:
  doc_A = positive (received apply)
  doc_B = negative (query had apply signals, doc_B didn't get one)
  doc_C = MASKED (no apply signal for this query at all — don't learn from it)
```

Effect: rare action heads produce calibrated scores 5× higher on average while
keeping ranking metrics unchanged.

---

## 8. Multi-Teacher Distillation (MTD)

### Warm-Start Strategy

Single-stage distillation from an open-source checkpoint is inefficient — the student
must simultaneously learn relevance and engagement from scratch.

**Better approach:**
1. First train a relevance-specialized SLM (§4 above) → strong relevance baseline
2. Warm-start MTD from this relevance SLM
3. MTD now focuses on learning engagement objectives while assigning small loss
   weight to prevent relevance drift

| Initialization | NDCG@10 | Click AUROC |
|---|---|---|
| Open-source model | 0.9088 | 0.6574 |
| Warm-start from relevance SLM | **0.9150** | **0.6704** |

+0.68% NDCG@10, +1.98% Click AUROC in Job Search.

### KL Divergence Losses (not hard labels)

Each batch is forwarded through teachers to get task-specific logits. Student distills
via **weighted KL divergence** — not hard 0/1 binary labels. This preserves teacher
uncertainty (a borderline positive remains a borderline signal for the student).

The weighted sum of KL losses also gives emphasis on challenging or application-specific
tasks.

---

## 9. Calibration

Deep ranking scores are often miscalibrated — discriminative losses and time-sampling/
downsampling shift probability distributions away from true rates.

### Calibration Layer Design

A PyTorch isotonic-regression-style calibration layer:
- **Multi-head:** each head (click, apply, connect...) trained independently with its
  own features and calibration labels (primary, proxy, business-aligned)
- **Position-conditioned:** a position-conditional vector `{p̂(r)}` where p̂(r) is the
  outcome likelihood if placed at rank r. Learned via loss masking.

Position conditioning matters because items shown at rank 1 get clicked more than
items at rank 10 regardless of intrinsic quality — the calibration layer learns
rank-specific priors and produces position-aware allocation scores.

**Result:** Click AUROC 0.6704 → 0.7095 in Job Search (position conditioning alone).
O/E ratio moves toward 1.0 from a broad pre-calibration spread.

---

## 10. Feature Engineering with SLMs

### Member Summary via RL Summarization

Member profiles are heterogeneous (text, structured attributes, activity history).
Naive summarization degrades performance. LinkedIn trains a **1.7B RL summarization
model** (GRPO training) using the engagement teacher as reward:

```
Reward R(s) = 𝟙[â_n = a_n] × (1 - λ_len · ℓ(s) + λ_qual · q(s))
```

Where ℓ(s) = normalized length penalty, q(s) = factuality/saliency score from a 32B
model. The summarizer learns to produce concise, factual, engagement-predictive summaries.

Effect: +4.40% Connect AUROC, +11.60% Message AUROC (People Search).

### Numerical Features in Prompt

Network distance, historical CTR, connection counts are numerical signals that need
careful prompt encoding:

| Encoding | ΔAUC |
|---|---|
| Short feature identifiers | baseline |
| Descriptive feature identifiers | +5.8% |
| Binary feature values as True/False | +1.7% |
| CTR feature | +5.1% |
| Truncation to 2 decimal places | 0.0% |

Descriptive identifiers ("network_distance_2nd_degree" vs "nd2") help the LLM parse
the feature meaning without extra context tokens.

### Network Features

Social graph features (past impressions, clicks, network size, distance, common
connections) add +17.6% Connect AUROC (People Search, Table 6).

---

## 11. Stage 1 — Semantic Retrieval

### Foundation: LiNR (LinkedIn Neural Retrieval)

The retrieval layer in this paper is built directly on top of **LiNR** — LinkedIn's
GPU-based retrieval system (Borisyuk et al., CIKM 2024). The semantic search paper
cites LiNR [ref 5] in three places:

- Introduction: "our work builds on a GPU-accelerated exhaustive retrieval stack [LiNR],
  enabling full scans over billion-scale indices with rich attribute-based filtering and
  without ANN liquidity issues"
- §2.2.3: "we replace pure cosine similarity scoring with GPU retrieval-as-ranking
  (GPU RAR) scoring [LiNR]"
- §2.2.4: "GPU-accelerated exhaustive retrieval stack with attribute-based filtering [LiNR]"

**What LiNR provides (and semantic search inherits):**

| LiNR capability | How semantic search uses it |
|---|---|
| Exhaustive GPU search (no ANN) | Full scan of 1B+ jobs/members at 4ms latency |
| ABM — Attribute-Based Matching | Pre-filters hard constraints (location, modality) before embedding scan |
| GPU RAR scoring | Weighted cosine + personalization features as the retrieval score |
| Live-updated indexes | Model weights and item embeddings updated without index rebuilds |

**Why exhaustive search instead of ANN (FAISS/ScaNN)?**

ANN algorithms use clustering or graph structures to skip most of the index — fast, but
they create a "liquidity" problem: items that pass attribute filters (e.g., remote jobs)
may not be in the same cluster as the query embedding, so they get skipped and never
retrieved. Post-filtering (retrieve first, filter after) makes this worse — you retrieve
1000 candidates, filter out 900 for being on-site, and are left with 100, hurting recall.

LiNR solves this with **attribute-based pre-filtering**: apply hard constraint masks
before the embedding similarity computation, then do exhaustive search over the
filtered set. No liquidity issue — every valid candidate is considered.

```
Query: "senior ML engineer, remote, TensorFlow"
  ↓
ABM pre-filter: mask all on-site jobs → filtered candidate pool
  ↓
Exhaustive GPU matrix multiply: cos(e_q, e_d) over filtered pool
  ↓
GPU RAR: reweight by personalization features
  ↓
Top-1000 → Stage 2 SLM ranker
```

→ See `LinedIn_GPU_RaR.pdf` (downloaded) for full LiNR system details including
  KNN with similarity masking, quantized KNN, and live-update infrastructure.

---

### Bi-Encoder Architecture

A contrastive LLM bi-encoder maps queries and documents into a shared embedding space:

```
query q  ──→ encoder_q ──→ e_q  ┐
                                  ├→ score = cos(e_q, e_d)
doc   d  ──→ encoder_d ──→ e_d  ┘
```

Bidirectional attention (unlike decoder-only LLMs) captures full document context.
The paper notes: decoder-only LLMs can be converted to strong embedders via contrastive
objectives and that limited embedding-specific fine-tuning suffices for universal embeddings.

### Training Objective: InfoNCE + Pairwise Margin

Two complementary losses:

**Global InfoNCE** — preserves global distribution structure:
```
L_InfoNCE = -log [ exp(<e_q, e_d+>/τ) / (exp(<e_q, e_d+>/τ) + Σ_{d-∈B-} exp(<e_q, e_d->/τ)) ]
```
τ = temperature controlling sharpness. B- = in-batch negatives + explicitly mined hard negatives.

**Pairwise Margin Loss** — sharpens local decision boundaries:
```
L_pair = Σ_{d- ∈ D-} max(0, m - <e_q, e_d+> + <e_q, e_d->)
```
Where D- = curated hard negatives, m = minimum separation margin.

Combined: `L = λ·L_InfoNCE + (1-λ)·L_pair`

### Hard Negative Mining

Training negatives are sampled from top-ranked production candidates that are labeled
non-relevant by the LLM judge. This is critical — random negatives are too easy
(the model trivially separates them) and don't train decision boundaries near
the relevance threshold.

### GPU RAR — Retrieval-as-Ranking

Pure cosine similarity ignores personalization and engagement. GPU RAR adds learned
features on top of embedding similarity:

```
S(q, d) = w₀·cos(e_q, e_d) + Σᵢ wᵢ·fᵢ(q, d)
```

Where fᵢ = personalization and engagement features (network proximity, profile popularity,
historical engagement rates). GPU RAR is trained with a multi-task objective:

```
L_RAR = λ·L_BCE(S, L_R) + (1-λ)·L_BCE(S, L_E)
```

L_R = relevance label, L_E = engagement label. This replaces multiple production
retrieval calls with a single call — simpler infra, better quality.

**Retrieval metrics (People Search):**

| Model | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| Baseline-4B | 0.33 | 0.70 | 0.71 |
| + Chat template | 0.36 | 0.70 | 0.74 |
| + HardNeg Mining | 0.41 | 0.75 | 0.76 |
| + Quality upsampling | 0.46 | 0.78 | 0.78 |
| + GPU RAR model | **0.47** | **0.79** | **0.79** |

---

## 12. Inference Optimizations — 75× Throughput Gain

The central challenge: LLM rankers are dominated by prefill (the shared query prefix +
each candidate document). No iterative decoding — just a single forward pass outputting
final-token logits. This means all classical LLM inference tricks for decode don't apply.

### Optimization 1 — Context Summarization *(Throughput/Cost)*

Inference cost scales linearly with context length. Long job descriptions and member
profiles are the dominant cost. LinkedIn trains a 1.7B summarization LLM to compress:
- Job Search: ~4× throughput improvement (documents compressed by ~4×)
- People Search: 95th-percentile prompt 1,500 → 500 tokens, negligible quality loss

### Optimization 2 — Model Pruning (OSSCAR) *(Throughput/Cost)*

Structured compression: remove 50% of hidden neurons from each MLP block + remove
final 8 transformer layers.

```
Before: 600M params
After:  375M params
```

Fine-tuning recovers quality. Result: 290 → 2,200 items/s/GPU (7.5×) with context
summarization under same latency budget.

People Search ranking quality with pruning:

| Model | P@10 | R@10 | NDCG@10 |
|---|---|---|---|
| Dense (600M) | 0.5369 | 0.8597 | 0.8629 |
| Pruned (375M) | 0.5434 | 0.8578 | 0.8652 |

Pruned model slightly exceeds dense baseline — fine-tuning recovers the lost capacity
and the smaller model generalizes slightly better.

### Optimization 3 — MixLM (Text-Embedding Mix Interaction) *(Throughput/Cost)*

Full cross-encoder interactions are expensive because every candidate's full text is
in the context. MixLM replaces document text with a **single learned embedding token**
cached nearline (confirmed in paper Table 13: "when items are represented by a single
embedding token"):

```
Standard cross-encoder:
  prompt = [system][query text][candidate full text][suffix]
           ← long, can't cache candidate portion →

MixLM:
  prompt = [system][query text][candidate embedding tokens][suffix]
           ← candidate tokens are fixed-size, cached, injected as binary payloads →
```

Encoder compresses each document → embedding tokens (offline). Ranker consumes query +
embedding tokens — preserving cross-encoder interaction quality while reducing context.

With a 1-token suffix, GPU compute per candidate becomes negligible relative to the shared
prefix — throughput is now dominated by **CPU orchestration**, not GPU math. This is why
multiprocess gRPC matters specifically at the MixLM stage: single-process saturates at
~10k items/s/GPU; multiprocess + CUDA graphs scales to ~22k items/s/GPU (Table 13).

**Table 13 — Infrastructure ablation under MixLM (0.6B ranker, H100, p99 ≤ 500ms):**

| Infrastructure configuration | Throughput (items/s/GPU) |
|---|---|
| Single gRPC servicer + single SGLang worker | ~10,000 |
| Multi-process serving (6 gRPC + 6 SGLang) | ~19,500 |
| Multi-process serving + CUDA graph | **~22,000** |

Single-process hits a hard ceiling at ~10k — CPU orchestration saturates one core.
6 gRPC + 6 SGLang worker processes nearly double it to ~19.5k by parallelizing across cores.
CUDA graphs eliminate the remaining per-kernel dispatch overhead → 22k.

Throughput vs approach:
| Approach | Throughput |
|---|---|
| Full text SLM | 290 items/s/GPU |
| Summarized + Pruned | 2,200 items/s/GPU |
| MixLM + multiprocess + CUDA graph | 22,000 items/s/GPU (~76× over raw-text) |

### Optimization 4 — Scoring-Optimized Prefill Execution *(Latency/Throughput)*

#### Ranking Requirements vs Standard LLM Generation

Ranking is **classification**, not generation. The SLM only needs to answer one question
per candidate: "Yes or No — is this relevant?" That's a single logit. But standard LLM
inference infrastructure is built for generation — outputting sequences of tokens one by one.

**Standard generation workflow:**
```
1. Tokenize input
2. Prefill: forward pass over full input → build KV cache
3. Decode loop (per token):
   - Forward pass on last token
   - Compute log-probs over FULL vocabulary (50k tokens)
   - Apply temperature / top-k / top-p sampling
   - Transfer result CPU ↔ GPU for sampling decision
   - Append new token, repeat
4. Return generated sequence
```

**Ranking workflow — what is actually needed:**
```
1. Tokenize input
2. Single forward pass over full input (prefill only)
3. Extract logit for "Yes" token at final position
4. Done — that is the relevance score
```

Everything in the standard decode loop is wasted for ranking:

| Standard path step | Needed for ranking? |
|---|---|
| Autoregressive decode loop | No — no tokens to generate |
| Log-probs over full 50k vocab | No — only Yes/No logits needed |
| Temperature / top-k / top-p sampling | No |
| KV cache kept alive for future steps | No — release immediately after score |
| CPU↔GPU transfers for sampling decisions | No |

**SGLang scoring-only prefill path** eliminates all of this:
single forward pass → grab final-position Yes/No logits → release KV state → return score.

Effect: 900 → 1,300 items/s/GPU (+44%) on a 375M pruned model.

The 44% gain purely from removing decode overhead — paid 250× per request (one per candidate)
at tens of thousands QPS.

### Optimization 5 — Shared-Prefix Amortization *(Throughput)*

#### The Core Observation

Every candidate prompt for a given query shares the same long prefix:

```
Candidate 1:   [system (100 tok) | query (50 tok) | searcher ctx (100 tok)] | [job_1 (150 tok)]
Candidate 2:   [system (100 tok) | query (50 tok) | searcher ctx (100 tok)] | [job_2 (150 tok)]
...
Candidate 250: [system (100 tok) | query (50 tok) | searcher ctx (100 tok)] | [job_250 (150 tok)]
               ←─────────────────── same 250 tokens ──────────────────────→
```

In naive inference the KV cache for that 250-token prefix is computed **250 times** — once per
candidate. That is pure wasted compute.

#### How Transformers Compute Attention — Why KV Cache Matters

When processing a sequence, every token computes K and V matrices (stored in KV cache) and a Q
matrix to attend over all previous K/V pairs. For a 400-token sequence [prefix + candidate]:

```
Each suffix token attends to:
  - all 250 prefix tokens  (reads prefix KV cache)
  - previous suffix tokens (reads suffix KV cache)
```

The prefix KV computation is O(T_q²) — quadratic in prefix length. Recomputing it 250 times is the waste.

#### With Prefix Amortization

```
Step 1 (once):  forward pass over prefix → build prefix KV cache
                store in GPU HBM — shared across all 250 candidates

Step 2 (×250):  for each candidate, only compute suffix forward pass
                suffix tokens attend to: shared prefix KV + their own KV
```

#### FLOPs Math

```
Naive:     N_i × (T_q + T_i)²
Amortized: T_q² + N_i × (2·T_i·T_q + T_i²)

Savings  = T_q² × (N_i - 1)   ← scales quadratically with prefix length
```

With T_q = 50 (small prefix), T_i = 150, N_i = 50:
- Naive:     50 × (200)² = 2,000,000
- Amortized: 2,500 + 50 × 37,500 = 1,877,500  → ~6% saving

With T_q = 500 (real system prompt + query + searcher context), T_i = 150, N_i = 250:
- Naive:     250 × (650)² = 105,625,000
- Amortized: 250,000 + 250 × 172,500 = 43,375,000  → **~59% reduction**

The longer the prefix, the more amortization saves — quadratically. Richer query context makes
prefix amortization more valuable, not less.

#### Two Implementation Strategies

**1. In-Batch Prefix Caching (IBPC):**
Compute prefix KV once per batch, store in HBM. All 250 items read from the same physical KV
tensors during their suffix forward pass. Each item only materializes its own suffix KV.

**2. Multi-Item Scoring:**
Concatenate all 250 items into one giant prompt:
```
[prefix | job_1 | job_2 | ... | job_250]
```
Use an attention mask to prevent job_i from attending to job_j (they are independent).
Prefix is computed once by construction — it is just one sequence.

**Impact: 1,600 → 2,000 items/s (+25%)**

#### Why MixLM Makes This Even Better

With MixLM, item suffix shrinks from 150 text tokens to **1 embedding token**. Now T_q >> T_i —
the prefix dominates the context entirely. Amortizing it saves nearly all the attention compute:

```
Without MixLM: T_q=250, T_i=150  → prefix is 63% of context
With MixLM:    T_q=250, T_i=1    → prefix is ~99% of context
```

MixLM shrinks the suffix; IBPC eliminates redundant prefix compute. The two optimizations compound.

### Optimization 6 — CPU and Runtime *(Latency/Throughput)*

After GPU-side work, CPU becomes bottleneck:
- **Batch tokenization + batch send:** single tokenization pass → 750 → 900 items/s/GPU (+20%)
- **Multi-process gRPC:** parallelize CPU preprocessing across cores (bypass Python GIL) → 1,300 → 1,600 items/s/GPU (+23%)
- **Python heap freeze:** `gc.freeze()` post-warmup stabilizes tail latency

#### Zero-Copy Feature Transfer — PyArrow Buffers

Features for the SLM ranker (CTR, network distance, impression counts) are fetched from
the feature store and transferred as **PyArrow buffers**. This is the zero-copy technique
applied to structured feature data.

**Why it matters — the normal deserialization problem:**
```
JSON / protobuf path:
  feature store → serialize to JSON/protobuf bytes → send via gRPC
  ranker receives bytes → parse field by field → build Python dicts/lists → build tensors
  cost: CPU cycles for parsing + memory allocation for intermediate objects
```

**Arrow's zero-copy property:**
Arrow's wire format = Arrow's in-memory format. The binary layout of an Arrow buffer is
already a typed columnar array — no transformation needed on receipt:

```
Arrow path:
  feature store → Arrow columnar binary (schema embedded) → send via gRPC
  ranker receives bytes → wrap in PyArrow view → read fields directly as typed arrays
  cost: one memory view creation, no parsing, no field-by-field deserialization

  features["ctr"]              → float32 array, ready to use
  features["network_distance"] → int32 array, ready to use
  features["impression_count"] → int64 array, ready to use
```

**Arrow vs OS shared memory:**

| | OS shared memory | Arrow buffer |
|---|---|---|
| Mechanism | Both processes map same physical pages | Structured binary, zero deserialize on receipt |
| Data movement | None — same physical RAM | Bytes sent via gRPC, but no transform on receipt |
| Schema | Manual agreement between processes | Embedded in the buffer itself |
| Works cross-node? | No — same machine only | Yes — travels over gRPC to any host |
| Used in | Ray Plasma store, same-node IPC | LinkedIn feature store → ranker |

OS shared memory eliminates data movement entirely (same physical RAM). Arrow eliminates
the deserialization step (bytes are already in the right format). Same zero-copy spirit,
different layer — Arrow works cross-node where shared memory cannot.

### CUDA Graph Execution *(Latency/Throughput)*

#### Why Kernel Launch Overhead Becomes the Bottleneck

Every GPU operation (matmul, layernorm, attention) is a **kernel**. To run a kernel, the CPU dispatches
it to the GPU queue — fixed cost ~5–30μs per kernel regardless of how large or small the GPU work is.

When kernels are large (big matmuls, long sequences), GPU execution time >> dispatch overhead → negligible.
But after all previous optimizations — shorter sequences, smaller model, no decode loop, cached prefix —
each remaining kernel is small and fast. GPU finishes quickly, then **waits for CPU to dispatch the next
kernel**. Bottleneck shifts from GPU compute to CPU dispatch overhead.

#### What CUDA Graphs Do

CUDA graphs let you **record** a sequence of GPU operations once, then **replay** the entire sequence
with a single CPU command:

```
Without CUDA graph:
  CPU → dispatch kernel_1 → GPU executes
  CPU → dispatch kernel_2 → GPU executes     ← per-kernel overhead × N kernels
  CPU → dispatch kernel_3 → GPU executes

With CUDA graph:
  Record (once):   CPU records kernel_1, kernel_2, kernel_3 ...
  Replay (every):  CPU issues one "run graph" command
                   GPU executes all kernels without waiting for CPU  ← no per-kernel overhead
```

#### The Dynamic Shape Problem — and How It's Solved

CUDA graphs are static — they capture exact tensor shapes and memory addresses. Variable-length
inputs break the graph.

**Solution: padding + bucketing.** Define discrete size buckets, pad inputs to the nearest bucket,
capture one CUDA graph per bucket:

```
Buckets: 32, 64, 128, 256 tokens

query_len=40  → pad to 64  → replay graph_64
query_len=90  → pad to 128 → replay graph_128
```

Padding tokens are masked out in attention — the model ignores them. Shape is what matters for the graph.

**This applies to candidates too.** The same bucketing logic works for candidate suffixes:

```
candidate_len=40  → bucket_64  → graph_64
candidate_len=150 → bucket_256 → graph_256
```

This is called **length-sorted batching** — group candidates of similar length into the same batch,
pad to bucket, replay the matching graph.

**Why candidates are harder to bucket than queries:**
- Query length variance is low (5–50 words) — 3-4 buckets cover 95% of traffic
- Raw candidate length variance is high (50–2,100 tokens) — many buckets needed, short candidates
  padded to 256 waste 80% of compute
- Continuous batching mixes candidates from multiple queries of different lengths in one batch —
  needs length sorting before batching (adds latency) or heavy padding within batch

**What's truly fixed vs bucketed vs dynamic:**

```
System prompt       → completely fixed, always identical  → full CUDA graph
Query               → variable, bucketed (3-4 sizes)      → one graph per bucket
Searcher context    → fixed-format numerical features     → fixed shape
Candidate suffix    → variable length (raw text)          → dynamic / eager execution
```

This is what "piecewise" means — different segments have different levels of dynamism. Stable segments
get graphs, dynamic segments fall back to eager execution.

#### Why MixLM Solves the Candidate Shape Problem Entirely

MixLM compresses every candidate to exactly **1 embedding token**. Candidate suffix is
**always** `[1, dim]` — no bucketing needed at all:

```
Without MixLM: [prefix (bucketed) | candidate (variable 50–2100 tok)] → piecewise graphs
With MixLM:    [prefix (bucketed) | 1 embedding token]                → single full graph per bucket
```

With MixLM, the entire sequence becomes fixed-shape within a bucket. No dynamic segments, no eager
fallback. This is why MixLM and CUDA graphs compound — MixLM eliminates the shape variability that
forced piecewise execution in the first place.

**Impact: 2,000 → 2,200 items/s/GPU (+10%), cumulative 2.93× over baseline.**

The +10% is the last mile — modest in isolation but only visible after all other bottlenecks are removed.

### Full Throughput Stack — Table 12

375M pruned decoder-only ranker, 50 query + 150 item tokens, batch size 50, single H100, p99 ≤ 500ms. Total speedup: **2.93×**.

| Optimization stage | Incremental gain | Throughput (items/s/GPU) |
|---|---|---|
| Baseline | — | 750 |
| + Batch tokenization & batch send | +150 | 900 |
| + Scoring-only prefill execution | +400 | 1,300 |
| + Python/runtime optimizations | +300 | 1,600 |
| + In-batch prefix caching (IBPC) | +400 | 2,000 |
| + Piecewise CUDA graph (prefill) | +200 | **2,200** |

---

## 13. Middle Tier Serving Optimizations

The serving stack applies infrastructure-level optimizations above and at the GPU layer.
These patterns are **generic to any retrieval-ranking system** — see the shared reference doc
for full details: [`ML_Design/ml_serving_infra_optimizations.md`](ml_serving_infra_optimizations.md)

**LLM-as-ranker optimizations** (Patterns 5–6 in the doc — apply to any system using an LLM as a scorer):

| Pattern | Mechanism | Impact |
|---|---|---|
| Scoring-only prefill | Read logit at last prompt token, skip decode entirely | Eliminates N serial decode steps per candidate |
| Shared prefix KV cache reuse | Compute shared prefix (query + profile) once, reuse KV tensors for all 250 candidates | 71× token reduction in MixLM case (99-token prefix, 1-token suffix) |

**Midtier infrastructure patterns** (Patterns 1–4 in the doc — generic to any retrieval-ranking system):

| Pattern | Mechanism | Impact |
|---|---|---|
| Result caching | Couchbase, key = hash(searcher, query, entity) | >50% hit rate, -8–10% latency |
| Retry with latency budget | Retry to different worker if primary exceeds T ms | +10% throughput |
| PID-based dynamic scoring depth | GPU queue depth controls candidate pool size | 250→130 at peak, -48% GPU compute |
| Traffic shaping | Defer nearline/batch to idle GPU windows | +25% GPU utilization |

---

## 14. Production Results

| Metric | Job Search | People Search |
|---|---|---|
| NDCG@10 improvement | +7.73% | >10% |
| Poor Match Rate@10 (PMR@10) | -46.88% | — |
| DAU lift | — | +1.2% |

The new system delivers substantial gains over the prior DLRM-style ranking baseline
while meeting production latency (p99 ≤ 500ms) and throughput constraints.

---

## 14. Key Design Decisions Summary

| Decision | Why |
|---|---|
| Two-stage (retrieval + SLM ranking) | Cross-encoders too slow for 1B+ candidates; bi-encoder gives recall, SLM gives quality |
| 0.6B student, not 8B oracle directly | 8B at tens of thousands QPS is economically infeasible; distillation transfers quality |
| Soft labels (not hard 0/1) | Preserves uncertainty near decision boundary; student learns gradations |
| Warm-start MTD from relevance SLM | Student focuses on learning engagement, avoids relevance drift |
| Loss masking for rare actions | Prevents rare action probabilities collapsing to zero due to false negatives |
| Context summarization before pruning | Reduces input length first, then model size — both multiplicative |
| Scoring-only prefill path | No decode needed for ranking → eliminates all autoregressive overhead |
| Shared prefix amortization | Prefix dominates context length; amortizing it scales quadratically |
| GPU RAR over pure cosine | Incorporates personalization without separate retrieval calls |
