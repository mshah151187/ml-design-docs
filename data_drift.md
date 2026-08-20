# Data Drift — Detection and Handling

Data drift is when the statistical properties of production data change over time,
causing model performance to degrade. The model was trained on data from the past;
the world it operates on keeps changing.

---

## Types of Drift

```
Feature drift (covariate shift):
  Input distribution P(X) changes, but P(Y|X) stays the same.
  Example: users start uploading higher-resolution images — pixel statistics shift
           but the relationship between image content and label is unchanged.

Label drift (prior shift):
  Output distribution P(Y) changes.
  Example: fraud rate spikes from 0.1% to 1% — model's base rate assumption breaks.

Concept drift:
  The relationship P(Y|X) itself changes — what a feature means shifts.
  Example: "work from home" in job postings meant something different pre-2020 vs post.
           Same text, different signal. Model trained pre-2020 misreads it.
```

Concept drift is the hardest — there is no statistical test that catches it directly.
You only see it in performance metrics.

---

## Detecting Drift

### 1. Monitor Prediction Distributions

Track the distribution of model outputs over time. Sudden shifts signal something changed.

```
Fraud model:
  Week 1: flags 0.1% of transactions
  Week 4: flags 1.0% of transactions  ← 10× spike

Could mean:
  (a) actual fraud increased — model working correctly
  (b) input feature distribution shifted — model misfiring
  (c) upstream data pipeline bug — garbage in

Alert triggers investigation. Don't auto-retrain on this signal alone.
```

Easy to implement — no ground truth labels needed, just output distribution statistics.

### 2. Monitor Feature Distributions

Track statistics on input features over time. Flag when they drift beyond thresholds.

```
Numerical features:   mean, variance, percentiles (p50, p95, p99)
Categorical features: category frequencies, new category appearance rate
Text features:        vocabulary overlap, OOV (out-of-vocabulary) token rate
Embedding features:   cosine similarity of current batch centroids vs training centroid
```

```
Member age distribution in training data:  mean=34, std=9
Member age distribution this week:         mean=28, std=11  ← shifted younger

Alert: age_mean drifted beyond 2σ threshold
Investigation: new user acquisition campaign targeting younger demographics
Action: retrain with recent data or reweight training sample
```

Statistical tests used in practice:
- **KS test** (Kolmogorov-Smirnov) — compares two continuous distributions
- **PSI** (Population Stability Index) — common in finance, measures shift magnitude
- **Chi-squared test** — categorical feature distribution comparison
- **MMD** (Maximum Mean Discrepancy) — embedding space drift detection

### 3. Monitor Performance Metrics on Labeled Production Data

Track accuracy, precision, recall, NDCG, AUC on production data where ground truth
is available. Degradation is the most direct signal of drift.

```
Week 1:  NDCG@10 = 0.82
Week 4:  NDCG@10 = 0.74  ← -8 points over 3 weeks
Week 8:  NDCG@10 = 0.68  ← continuing degradation → retrain triggered
```

**The challenge: label latency.** Ground truth labels arrive with delay:
- Click/engagement: hours
- Purchase: days
- Fraud confirmation: weeks
- Medical outcome: months

Fast-feedback labels (clicks) are noisy proxies. The drift that matters most often
has the slowest label feedback.

### 4. Holdout Test Set Degradation

Keep a fixed holdout test set from initial training. Evaluate the production model
against it periodically — the test set never changes, so performance degradation is
purely due to model/world mismatch.

```
Monthly evaluation on fixed holdout:
  Month 0 (launch):  AUC = 0.91
  Month 3:           AUC = 0.89
  Month 6:           AUC = 0.85  ← threshold crossed → retrain
  Month 7 (retrain): AUC = 0.92  ← back to baseline with fresh data
```

Clean signal but limited — doesn't tell you which features drifted or why.

---

## Handling Drift

### Strategy 1 — Regular Retraining (First Line of Defense)

Retrain on a rolling window of recent data on a fixed cadence.

```
Weekly retraining pipeline:
  Monday 2am:
    1. Pull last 90 days of labeled data
    2. Train new model
    3. Evaluate on holdout
    4. If eval >= threshold: promote to production
    5. Else: alert on-call, keep current model

Tradeoffs:
  Pro:  simple, catches slow drift reliably
  Con:  misses rapid drift between retraining cycles
        expensive if model is large
```

This is the default answer in an interview. Most teams that do nothing else do this.

---

### Strategy 2 — Online Learning (Continuous Weight Updates)

Update model weights continuously as new labeled data arrives. Best for systems where
drift is rapid and labels are available quickly.

```
Fraud detection with online logistic regression:
  Transaction arrives → model scores it → fraud/not-fraud label arrives (hours later)
  → model immediately updates weights on this single example
  → next transaction gets slightly updated model

Adaptation lag: hours (not weeks)
```

**Works well for:** logistic regression, linear models, shallow models — simple update rule.

**Hard with deep learning:** gradient updates on single examples are noisy. Large models
need mini-batches to get stable gradient estimates. Full online learning with a 7B
parameter model is not practical.

**Risk: catastrophic forgetting.** The model overwrites old knowledge while learning new
patterns. If fraud patterns cycle (attackers adapt, then revert), online learning can
forget the old patterns that are now relevant again.

Mitigations:
- Elastic Weight Consolidation (EWC) — penalize changes to weights that were important
  for old tasks
- Replay buffer — mix new examples with a sample of old ones in each update batch
- Learning rate scheduling — small LR prevents large weight movements

---

### Strategy 3 — Online Embedding Learning (Frozen Weights, Live Embeddings)

Keep model weights frozen between retraining cycles. Continuously refresh the
**representations** of individual items/entities as they change.

```
Architecture split:
  Model weights:  retrained weekly/monthly (slow, expensive, needs labeled data)
  Embeddings:     refreshed continuously via nearline pipeline (fast, cheap, unsupervised)
```

**Why this works:** drift often comes from the world changing (new content, updated
profiles, new products) rather than the model's decision boundary being wrong. Refreshing
representations catches this without retraining.

#### LinkedIn Implementation (Feed + Semantic Search)

```
Change event arrives (CDC — Change Data Capture):
  Member updates job title: "Software Engineer" → "Staff Engineer"
  New post published
  Member adds new skill

Nearline pipeline:
  1. CDC event triggers embedding refresh job
  2. Frozen bi-encoder runs inference on updated content
  3. Fresh embedding written to ANN index (embedding store)
  4. Retrieval now uses updated representation

Latency: minutes from profile change to updated embedding in production
Model weights: unchanged — retrained on separate weekly cadence
```

This decouples two different cadences:

| Cadence | What changes | Cost | Frequency |
|---|---|---|---|
| Model retraining | Weights, decision boundary | High — labeled data, GPU hours | Weekly / monthly |
| Embedding refresh | Item representations | Low — just inference | Continuous / minutes |

**Applies to:** any retrieval system where items change — feed posts, job listings,
member profiles, product catalogs, news articles.

**Why inference efficiency matters here:** the nearline pipeline is a continuous,
high-throughput inference system. At LinkedIn scale, millions of profile updates per
day → millions of encoder forward passes per day. This is exactly where batching,
CUDA graphs, and quantization pay off — not just for serving latency but for embedding
freshness cost.

---

### Strategy 4 — Ensemble Over Time Windows

Maintain multiple models trained on different time windows. Weight their predictions
based on which time period best matches current data.

```
Models in production simultaneously:
  model_t0  — trained on data 90–180 days ago
  model_t1  — trained on data 30–90 days ago
  model_t2  — trained on data 0–30 days ago (most recent)

Weighting at inference:
  If current data resembles recent window → upweight model_t2
  If current data resembles older pattern → upweight model_t0 or model_t1

Implemented via multi-armed bandit:
  Treat each model as an arm, reward = prediction accuracy on recent labeled data
  Bandit continuously adjusts weights based on which model is performing best
```

**Hedges against drift** — you always have a model tuned to recent data and older models
as fallback. **Increases serving cost** proportionally to ensemble size.

Used when you have other reasons for an ensemble (uncertainty estimation, model diversity)
and drift handling is an additional benefit.

---

### Strategy 5 — Human-in-the-Loop for High-Stakes Predictions

Route uncertain or high-impact predictions to human reviewers. Their feedback becomes
fresh labeled training data that reflects current patterns.

```
Medical diagnosis system:
  High-confidence prediction → serve directly
  Low-confidence prediction → route to radiologist for review
  Radiologist label → immediately added to retraining queue

Benefit: fresh labels with exact current distribution
Cost: human review is expensive and slow → only viable for high-value decisions
```

---

## Strategy Selection

| Drift speed | Label latency | Scale | Recommended approach |
|---|---|---|---|
| Slow (months) | Days–weeks | Any | Regular retraining |
| Medium (weeks) | Hours–days | Any | Regular retraining + feature monitoring |
| Fast (hours–days) | Hours | Simple models | Online learning |
| Fast, representation-driven | None needed | Large systems | Online embedding learning |
| Variable / cyclical | Days | Multiple models already | Ensemble + bandit |
| High-stakes, low volume | Immediate | Small | Human-in-the-loop |

**Default interview answer:**
> "I'd start with weekly automated retraining on a rolling data window, with feature
> distribution monitoring to detect drift early. If the domain changes rapidly — like
> fraud or news recommendation — I'd add a nearline embedding refresh pipeline to keep
> representations fresh between retraining cycles without the cost of full retraining."

---

## Related Docs

- `semantic_search.md` — LinkedIn bi-encoder + nearline embedding refresh pipeline
- `ml_serving_infra_optimizations.md` — result caching, traffic shaping (serving infra
  that supports high-throughput nearline inference)
