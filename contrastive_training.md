# Contrastive Training

## What It Is

Contrastive training teaches a model to produce embeddings where **similar things
are close** and **dissimilar things are far apart** in vector space — without
requiring explicit class labels.

You give the model pairs:
- **Positive pair**: (query, relevant item) — should be close in embedding space
- **Negative pair**: (query, irrelevant item) — should be far apart

The model learns by being trained to rank the positive higher than all negatives.
This is the foundation of two-tower / bi-encoder models used in search, feed ranking,
and recommendation systems.

```
Before training:                After training:
  query ●                         query ●──── positive ●
           ● positive                          
  negative ●                      negative ●  (pushed far away)
```

---

## How Positive Labels Are Generated

The hardest part of contrastive training is not the model — it is getting good
positive pairs. Three approaches:

---

### 1. Human Curated Labels

A human annotator explicitly marks pairs as relevant.

- **Examples**: query-document relevance judgements, image-caption pairs rated by
  annotators, manually curated FAQ (question, answer) pairs
- **Quality**: highest signal — humans understand nuance, context, and intent
- **Cost**: expensive and slow to scale. A few thousand pairs require significant
  annotation budget and time.
- **When to use**: cold start, high-stakes domains (medical, legal), or when
  interaction data doesn't exist yet

```
annotator sees: query="senior software engineer jobs in NYC"
                doc="Staff Engineer opening at Stripe, New York"
annotator labels: RELEVANT → positive pair
```

---

### 2. Labels Generated from Real Interactions

Mine positive pairs from **observed user behavior** — clicks, engagement, purchases,
dwell time. If a user clicked it, it was relevant to them.

- **Examples**:
  - LinkedIn Feed: (member, post they liked/commented/shared) → positive pair
  - LinkedIn Search: (query, job posting clicked) → positive pair
  - E-commerce: (search query, product purchased) → positive pair
- **Quality**: implicit signal — noisy (accidental clicks, clickbait) but massive
  volume. Noise is acceptable when you have millions of pairs.
- **Cost**: near-zero — logs already exist
- **When to use**: production systems where user behavior data is abundant

```
user types: "python machine learning"
user clicks: "Hands-On Machine Learning with Scikit-Learn" listing
→ positive pair: ("python machine learning", listing_embedding)
```

The in-batch negatives come for free — other items in the batch that this user
did NOT click become negatives. One forward pass computes a B×B similarity matrix;
the diagonal entries are positives.

---

### 3. Data Augmentation via Generated Labels

Create positive pairs synthetically — either through **rule-based augmentation**
or **LLM-generated pairs** — when human labels are scarce and interaction data
doesn't exist.

**Rule-based augmentation:**
- Paraphrase the query (synonym replacement, back-translation)
- Crop/flip/rotate for image embeddings (same image = positive pair)
- Shuffle sentences in a document (same document = positive pair)

**LLM-generated pairs:**
- Prompt an LLM: *"Generate 5 queries a user might type to find this document"*
- Each (generated query, document) becomes a positive pair
- Used to bootstrap training data for a new domain where no logs exist

- **Quality**: lower than human labels, higher than random. LLM-generated pairs
  can introduce hallucinations or mismatches — need filtering.
- **Cost**: cheap at scale with a strong LLM. Quality scales with model quality.
- **When to use**: new product/domain launch, multilingual expansion, or to
  supplement sparse interaction data

```
document: "How to negotiate salary at a tech company"
LLM generates queries:
  → "salary negotiation tips software engineer"     ← positive pair
  → "how to ask for raise in tech job"              ← positive pair
  → "tech company compensation strategies"          ← positive pair
```

---

## Loss Function: InfoNCE / Cross-Entropy

Once you have (query, positive, negatives), you train with **InfoNCE loss** —
which is cross-entropy applied to a similarity distribution.

**Intuition:**
For each query, treat it as a classification problem: **which of these B items is the positive one?** The model must assign the highest score to the positive.

```
For a batch of B (query, positive) pairs:

similarity matrix [B × B]:
  rows = queries
  cols = all positives in the batch (each query's positive + (B-1) others as negatives)

  sim(q₁, p₁)  sim(q₁, p₂)  sim(q₁, p₃)  ...   ← row 1: q₁ must rank p₁ highest
  sim(q₂, p₁)  sim(q₂, p₂)  sim(q₂, p₃)  ...   ← row 2: q₂ must rank p₂ highest
  ...

diagonal = positives, off-diagonal = in-batch negatives
```

**Cross-entropy loss (per query):**

```
logits = [sim(q, p), sim(q, n₁), sim(q, n₂), ..., sim(q, n_{B-1})]  / τ

loss = -log( exp(sim(q, p)/τ) / Σ exp(sim(q, nᵢ)/τ) )
             ─────────────────────────────────────────
             softmax over all B items, target = position of positive
```

- `τ` (temperature) controls sharpness: low τ → model penalized more heavily
  for any positive that doesn't clearly outrank all negatives
- Loss → 0 when positive has much higher similarity than all negatives
- Loss is large when the positive is buried in the ranking

**Why cross-entropy:** it forces the model to assign probability mass to the
positive and away from all negatives simultaneously — not just pull the positive
closer, but also push all negatives away in one unified objective.

---

## Summary

```
Positive label source    Signal quality    Scale        Cost
─────────────────────────────────────────────────────────────
Human curated            Highest           Low          High
Real interactions        Medium (noisy)    Very high    Near-zero
Generated / augmented    Medium            High         Low-medium

Loss: InfoNCE = cross-entropy over similarity scores
      diagonal of B×B matrix = positives
      off-diagonal = in-batch negatives (free from batch construction)
```
