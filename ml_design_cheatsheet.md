# ML System Design — Interview Cheat Sheet

---

## The Framework (45 min total)

```
1. Problem Framing        5-7 min   ← set the foundation
2. High-Level Design      2-3 min   ← lightweight scaffolding
3. Data & Features       10 min    ← heavy, win here
4. Modeling              10 min    ← heavy, win here
5. Inference & Eval       7 min    ← separates lab from prod engineers
6. Deep Dives            remaining ← follow interviewer's lead
```

---

## 1. Problem Framing (5-7 min)

**Three things to do in order:**

### A. Clarify the Problem
Ask about:
- Who are the users? What is their pain point?
- What does the current solution look like (if any)?
- Scale: DAU, requests/sec, catalog size
- Latency: real-time vs batch inference
- Constraints: privacy, interpretability, cost

> ❌ Don't: jump into solution before clarifying scope
> ✅ Do: probe what makes this problem hard / interesting

### B. Establish Business Objective
State what success looks like for the **business** — not the model.

| Too vague | Specific |
|-----------|----------|
| "Improve user experience" | "Increase 7-day retention by improving feed relevance" |
| "Increase revenue" | "Increase CVR on recommendations" |
| "Reduce bad content" | "Reduce harmful content exposure on high-reach posts" |

> Key insight: the business objective and the naive ML objective are often NOT the same.
> Content moderation: a post seen by 10M people matters infinitely more than one seen by 1.

### C. Define ML Objective
Translate business objective → ML task + metric.

- What type of task? (classification, regression, ranking, retrieval, generation)
- What is the primary metric you'll optimize?
- What guardrails must you not break?

**The three-layer chain (always state all three):**
```
Business metric  →  what the company cares about      (retention, revenue, engagement)
Proxy metric     →  what the model actually optimizes  (CTR, watch time, acceptance rate)
Guardrail metric →  what must NOT regress              (spam rate, latency, diversity)
```

---

## 2. High-Level Design (2-3 min)

Draw or verbally describe:
```
[Data Sources] → [Feature Extraction] → [Model] → [Action/Output]
                       ↑
                 [Feature Store]
```

- Cover the full lifecycle: raw data → features → model → action taken
- Don't go deep on SWE details (DB choice, API design) — that's ML infra, not applied ML
- Purpose: give context for the rest of the discussion, not a deliverable

---

## 3. Data & Features (10 min)

### A. Training Data
Think in three buckets:

| Bucket | Examples | Notes |
|--------|----------|-------|
| Supervised | purchase history, explicit ratings, labels | Direct signal, usually scarce |
| Semi-supervised | clicks, dwell time, shares | Proxy labels, abundant |
| Unsupervised | all user sessions, all content | Pre-training, embeddings |

> ✅ Strong candidates use all three buckets, not just supervised data
> Ask: do I need to collect new data? How do I handle labeling? Cold start?

### B. Features
Don't dump a list. Pick 5-7 high-impact features and state **why each will be predictive**.

Common feature categories:
| Category | Examples |
|----------|----------|
| User features | demographics, past behavior, preferences, account age |
| Item features | category, price, metadata, popularity |
| Interaction features | past clicks, purchases, dwell time, ratings |
| Contextual features | time of day, device, location, session context |
| Derived/cross features | user-item similarity, recency × frequency, co-purchase patterns |

> ❌ Don't: rattle off 20 features with no reasoning
> ✅ Do: pick impactful features + state the hypothesis for each

### C. Feature Representation
Address how each type of data gets encoded:

| Data type | Representation |
|-----------|---------------|
| Categorical (low cardinality) | One-hot encoding |
| Categorical (high cardinality) | Learned embeddings |
| Numerical | Normalize / standardize, handle missing values |
| Text | Bag-of-words → TF-IDF → word2vec → transformer embeddings |
| Images | CNN features, pre-trained ViT embeddings |
| User + Item together | Two-tower embedding into shared latent space |

Online vs offline features:
- **Online**: queried at inference time, always up-to-date (current session context)
- **Offline**: pre-computed in batches, stored in feature store (historical embeddings)

---

## 4. Modeling (10 min)

### A. Baseline First
Always start with the simplest thing that could work. Sets a yardstick.

| Problem | Simple baseline |
|---------|----------------|
| Recommendations | Popularity-based ("most purchased") |
| Feed ranking | Reverse-chronological |
| Content moderation | Keyword blocklist |
| Search | BM25 / TF-IDF |
| Fraud detection | Rule-based thresholds |

> Baselines often perform surprisingly well and are fast to ship.
> They anchor the complexity tradeoff conversation.

### B. Model Selection
Discuss options + tradeoffs, then pick one to go deep on.

**Classical vs Deep Learning tradeoff:**

| | Classical (LR, GBDT, SVM) | Deep Learning |
|---|---|---|
| Data needed | Low-medium | High |
| Training cost | Low | High |
| Latency | Low | Higher |
| Interpretability | High | Low |
| Feature engineering | Manual | Learned |
| Best when | Tabular data, low latency needed | Unstructured data, lots of data |

Common model families by problem type:
| Problem | Options |
|---------|---------|
| Binary classification | Logistic Regression → GBDT (XGBoost) → Neural net |
| Ranking | LambdaMART → Two-tower → LLM reranker |
| Recommendations | Matrix factorization → Two-tower → Sequential models (SASRec) |
| NLP tasks | TF-IDF → BERT → fine-tuned LLM |
| Retrieval (ANN) | FAISS, ScaNN — complement to ranker, not a replacement |

### C. Model Architecture
Go one level deeper on your chosen model. Cover:
- Key components / layers
- Activation functions (ReLU for hidden, sigmoid for binary output, softmax for multi-class)
- Regularization (dropout, L2, early stopping)
- Loss function (cross-entropy for classification, MSE for regression, listwise/pairwise for ranking)

> ✅ Show you've actually built something similar
> ❌ Don't say "4 fully connected layers with 1024 neurons" — you can only know this empirically
> ✅ OK to say "I'm less familiar with ranking models but here's how I'd reason about it..."

---

## 5. Inference & Evaluation (7 min)

### A. Evaluation — Offline then Online

**Offline** (historical data, before shipping):

| Metric | Use case |
|--------|----------|
| AUC-ROC | Classification, fraud, moderation |
| Precision@K, Recall@K | Recommendations, search |
| NDCG | Ranked lists, order matters |
| RMSE / MAE | Regression problems |
| F1 | When precision-recall balance matters |

**Online** (A/B test on real users, after shipping):
- Define control (old) vs treatment (new model)
- Primary metric: the proxy metric you're optimizing
- Guardrail metrics: things that must not regress
- Run until statistical significance (typically 1-2 weeks)

> Always tie offline metric → online metric → business metric. A model that wins offline
> but loses online means your proxy metric is misaligned.

### B. Inference Considerations
Ask: what are the scale, latency, and cost requirements?

| Concern | Technique |
|---------|-----------|
| Model too slow | Quantization (FP32 → INT8), pruning, distillation |
| Too expensive to run for every user | Candidate generation → ranking (two-stage) |
| Same user/item queried repeatedly | Cache embeddings or predictions |
| Embeddings need to be searched fast | Approximate Nearest Neighbor (FAISS, ScaNN) |
| Model needs to update frequently | Online learning, micro-batch retraining |

> ✅ Mentioning these = you think in production terms
> ❌ Ignoring these = "notebook engineer" signal

---

## 6. Deep Dives (remaining time)

Follow the interviewer's lead. Common topics:

| Topic | Key points to hit |
|-------|------------------|
| Cold start | New user: use content-based / demographic defaults. New item: use metadata embeddings |
| Data bias | Training data reflects past behavior → model reinforces it. Mitigation: reweighting, fairness constraints |
| Scaling | Distributed training, sharding embeddings, ANN index for retrieval |
| Monitoring | Track: model accuracy, feature drift, prediction drift, business metrics. Alert on degradation. |
| Retraining | Trigger: metric drops below threshold OR time-based. Strategy: full retrain vs fine-tune |

---

## Metrics Reference

### Business → Proxy → Guardrail: Real Examples

**LinkedIn Feed Ranking:**
```
Business:   Engagement → retention → MAU growth
Proxy:      CTR + dwell time on posts
Guardrail:  Spam rate, feed diversity, professional content ratio
```

**LinkedIn PYMK (People You May Know):**
```
Business:   Network growth → more engagement → retention
Proxy:      Connection acceptance rate
            NOT invitation send rate (easy to spam, low quality signal)
Guardrail:  Invitation spam rate, diversity of suggestions
```

**YouTube Feed:**
```
Business:   Session time, next-day retention
Proxy:      Watch completion rate + satisfaction score
            NOT pure CTR (clickbait inflates CTR, kills watch time)
Guardrail:  Clickbait rate (high CTR + low completion = penalized), harmful content rate
```

**Amazon Recommendations:**
```
Business:   Revenue, repeat purchase rate
Proxy:      Add-to-cart rate, purchase conversion rate
Guardrail:  Recommendation diversity, return rate (recommending bad products)
```

**Content Moderation (Facebook/Instagram):**
```
Business:   Reduce legal risk + protect users from harm
Proxy:      Recall on harmful content — weighted by reach (10M-view post >> 1-view post)
            NOT accuracy (class imbalance makes accuracy misleading)
Guardrail:  False positive rate (don't remove legitimate content), latency
```

### Full Metric Lookup Table

**Business Metrics:**
| Metric | Meaning |
|--------|---------|
| DAU / MAU | Daily / Monthly Active Users |
| D7 / D30 Retention | % users still active after 7 / 30 days |
| Session length | Avg time per visit |
| CTR | Clicks / Impressions |
| CVR | Conversions / Clicks (purchase, signup, etc.) |
| ARPU / LTV | Revenue per user / Lifetime value |
| Churn rate | % users who stopped using product |

**ML / Proxy Metrics:**
| Metric | Meaning | Used for |
|--------|---------|---------|
| CTR | Click-through rate | Feed, ads, search |
| Dwell time | Time spent after click | Feed ranking |
| Watch completion | % of video watched | Video recommendations |
| Acceptance rate | % of suggestions acted on | PYMK, friend suggestions |
| Precision@K | % of top-K that are relevant | Recommendations |
| Recall@K | % of relevant items in top-K | Recommendations |
| NDCG | Ranked list quality, position-weighted | Search, ranking |
| AUC-ROC | Classifier discrimination | Fraud, moderation, binary tasks |
| F1 | Precision-recall balance | Imbalanced classification |
| MRR | Mean Reciprocal Rank — rank of first relevant result | Search |

**Guardrail Metrics:**
| Guardrail | Protects against |
|-----------|-----------------|
| Spam / false positive rate | Over-aggressive moderation or recommendations |
| Feed diversity | Filter bubble, echo chamber |
| Latency p99 | Tail latency degradation |
| Fairness metrics | Demographic performance gaps |
| Harmful content exposure | Engagement-at-all-costs |

---

## Worked Example: E-Commerce Recommendation System

**Problem:** "Design a product recommendation system for an e-commerce platform."

---

**1. Problem Framing**

Clarify:
- Catalog size: 10M products. DAU: 5M users. Recommendations shown on homepage + product pages.
- Latency: <100ms for homepage load. Batch OK for email recommendations.
- Cold start: ~20% of users are new each month.

Business objective:
> "Increase purchase conversion rate on recommended products."

ML objective:
> Ranking model predicting purchase probability given user context.
> Proxy: purchase CVR on recommended items.
> Guardrail: recommendation diversity (not just high-margin items), latency <100ms.

---

**2. High-Level Design**

```
[User activity logs]  →  [Feature extraction]  →  [Candidate generator]
[Product catalog]     →  [Embeddings]          →  [Ranker]
                                                →  [Top-K results]  →  [User]
```

Two-stage: candidate generation (retrieve ~500 from 10M) → ranking (score and sort 500).

---

**3. Data & Features**

Training data:
- Supervised: purchase history (strong signal, sparse)
- Semi-supervised: clicks, add-to-cart, dwell time on product page (abundant proxy)
- Unsupervised: all browsing sessions for user/item embeddings

Key features:
| Feature | Why predictive |
|---------|---------------|
| User's past purchase categories | Strong intent signal |
| Item popularity in user's segment | Social proof within cohort |
| User-item embedding similarity | Captures latent affinity |
| Time since last purchase in category | Recency drives repeat purchase |
| Contextual: device, time of day | Mobile users convert differently |

Representation:
- Users and items → two-tower embeddings (shared latent space)
- Categories → learned embeddings (high cardinality)
- Numerical (price, ratings) → normalized

---

**4. Modeling**

Baseline: popularity-based recommendations per category. Fast to ship, strong baseline.

Candidate generator: two-tower model trained on click/purchase data. User tower + item tower → dot product similarity → ANN search (FAISS) to retrieve top 500.

Ranker: GBDT (XGBoost) on retrieved candidates. Features: user-item embedding similarity + contextual features. Trained on purchase labels.

Why GBDT for ranking vs neural ranker: lower latency, interpretable, less data needed, easier to debug in production. Can upgrade to neural ranker once data volume justifies it.

Architecture (ranker):
- Input: concatenated user + item features (~100 dims)
- XGBoost with 500 trees, max depth 6
- Output: purchase probability score
- Loss: binary cross-entropy on purchase labels

---

**5. Inference & Evaluation**

Offline metrics:
- Candidate generator: Recall@500 (do we retrieve the item the user eventually bought?)
- Ranker: NDCG@10, Precision@10 on held-out purchases

Online (A/B test):
- Primary: purchase CVR on recommended items
- Secondary: CTR (engagement signal)
- Guardrail: recommendation diversity score, p99 latency <100ms

Inference:
- User embeddings: pre-computed nightly, stored in feature store (batch — not needed real-time)
- Item embeddings: updated daily, ANN index rebuilt nightly
- Ranker: <5ms per request (XGBoost is fast), fits latency budget
- Cache: homepage recommendations cached per user for 1 hour

---

**6. Deep Dives**

Cold start:
- New user: use demographic defaults + popularity-based recs. After 3 interactions, switch to personalized.
- New item: use product metadata embeddings (category, description, price tier) as proxy until purchase data accumulates.

Monitoring:
- Track: CVR daily, embedding drift weekly, feature distribution shift
- Alert: CVR drops >5% vs baseline → trigger investigation
- Retrain: weekly full retrain on rolling 90-day window

---

## Green / Red Flag Summary

| Section | ✅ Green | ❌ Red |
|---------|---------|-------|
| Framing | Business ≠ ML objective, clear chain | Jumped to ML without business context |
| Features | 5-7 with hypotheses, all three data buckets | Feature dump, only supervised data |
| Modeling | Baseline first, tradeoffs discussed | Jumped to transformer, no baseline |
| Architecture | Enough detail to show you've built it | Hand-waved ("some layers and stuff") |
| Evaluation | Offline + online + tied to business | Only offline metrics, no A/B discussion |
| Inference | Mentioned latency, caching, quantization | Only talked about accuracy |
