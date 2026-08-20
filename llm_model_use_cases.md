# LLM / Transformer Model Use Cases in Production ML Systems

Transformers are used in multiple distinct ways in production ML systems. The word
"transformer" does not imply the same architecture or purpose across these uses.
This doc maps each use case with a concrete example input and output.

---

## 1. DLRM — Deep Learning Recommendation Model

**Family:** Non-transformer. Embedding table + feature cross + MLP.  
**Used in:** LinkedIn Feed (prior ranker), Meta Feed/Ads ranking.  
**Purpose:** Score a (user, item) pair using handcrafted features.

No attention. No sequence. No raw text. The model works on pre-engineered numerical
and categorical signals.

**Example input:**
```
{
  user_id:          12345        → embedding lookup → 64-dim vector
  item_id:          67890        → embedding lookup → 64-dim vector
  user_country:     "US"         → embedding lookup → 16-dim vector
  item_age_hours:   3.5          → numerical
  user_item_affinity_score: 0.72 → numerical (pre-computed)
  network_distance: 2            → numerical
}
```

**Example output:**
```
click_prob:      0.14
long_dwell_prob: 0.31
```

**What the transformer does:** nothing — there is no transformer.  
**Limitation:** requires extensive feature engineering. Cannot understand raw text. Cannot
model temporal order of interactions. "Affinity score" itself often comes from a
separate embedding model.

---

## 2. Sequential Recommender (Feed SR style)

**Family:** Transformer over interaction history (time sequence).  
**Used in:** LinkedIn Feed SR, SASRec, BERT4Rec, HSTU.  
**Purpose:** Model how a user's interests evolve over time. Predict their next action
on a new candidate given their full interaction history.

The transformer processes a **sequence of past interactions** — not text, not a
query-document pair. Each position in the sequence is one (item, action) pair from
the user's history. Causal (decoder-style) attention — each position attends only to
earlier positions, preserving temporal order.

**Example input:**
```
Sequence of T=1000 past impressions (oldest → newest):

Position 1:  [post_embed: "ML news article",  action: long_dwell]
Position 2:  [post_embed: "job posting at Google", action: click]
Position 3:  [post_embed: "Python tutorial",   action: skip]
...
Position 999: [post_embed: "LLM fine-tuning guide", action: like]

→ Candidate to score:
Position 1000: [post_embed: "Transformer architecture deep dive"]
```

Each post_embed is itself a structured representation: content embedding (from a
separate text model), author ID embedding, engagement counts, post age, etc. — NOT
raw text passed directly to Feed SR.

**Example output:**
```
long_dwell_prob:    0.68   ← will user read this for >15s?
click_prob:         0.41
contribution_prob:  0.12   ← like, comment, share?
```

**What the transformer does:** attends across the 1000-item history to capture
long-term interests ("this user consistently engages with ML content") and short-term
session context ("in this session they've been reading deep-learning papers"). The
candidate at position 1000 is scored in the context of everything before it.

**Key trait:** the "query" is the user's entire history, not a text string.
The model never reads raw text — text understanding is offloaded to a separate
embedding model whose output is fused in as a feature.

---

## 3. LLM Cross-Encoder / SLM Ranker

**Family:** Language model transformer over text tokens.  
**Used in:** LinkedIn Semantic Search SLM ranker (0.6B), BERT-based re-rankers.  
**Purpose:** Given a (query, document) pair as raw text, predict relevance. No history,
no sequence of interactions — just two pieces of text side by side.

The transformer processes **text tokens** with full bidirectional attention across the
concatenated query + document. Every query token can attend to every document token
and vice versa — this is what makes it a cross-encoder.

**Example input (Job Search):**
```
System: "You are a job relevance judge. Given the query and job posting below,
         predict whether this job is relevant. Answer Yes or No."

User:   "Query: senior machine learning engineer, remote, TensorFlow

         Job: Software Engineer - ML Platform
         Company: Acme Corp | Location: Remote | Seniority: Senior
         Skills: TensorFlow, Python, distributed systems
         Description: Lead ML infrastructure for model training and serving..."
```

**Example output:**
```
First-token logit: Yes → 0.91, No → 0.09
→ relevance_prob = 0.91
```

**What the transformer does:** full bidirectional attention lets the model match
"remote" in query against "Remote" in job location, "senior" against "Senior"
seniority, "TensorFlow" against the skills list — token-level constraint checking
that a bi-encoder cannot do.

**Key trait:** stateless per request — no user history, no sequence. Scored independently
for each (query, candidate) pair. The "small" in SLM refers to parameter count (0.6B
vs frontier 70B+), not the architecture family.

---

## 4. LLM Bi-Encoder / Two-Tower

**Family:** Language model transformer, run independently per side.  
**Used in:** LinkedIn Semantic Search retrieval, LinkedIn Feed GR retrieval (OON).  
**Purpose:** Encode query and document independently into a shared embedding space.
Enables pre-computation of document embeddings offline — fast retrieval at scale.

Same transformer architecture as the cross-encoder, but the two sides **never see
each other** during encoding. The interaction is only a single dot product at the end.

**Example input:**
```
Query tower input:
  "senior machine learning engineer remote TensorFlow"
  → tokenize → transformer → pool → e_q ∈ ℝ⁷⁶⁸

Document tower input (pre-computed offline):
  "Software Engineer - ML Platform | Remote | Senior | TensorFlow, Python"
  → tokenize → transformer → pool → e_d ∈ ℝ⁷⁶⁸
```

**Example output:**
```
score = cos(e_q, e_d) = 0.83  → retrieved into top-1000
```

**Key trait:** document embeddings computed once, stored in index, reused for every
query. Only the query tower runs at query time. Fast enough for 1B+ document corpora.

---

## 5. LLM as Oracle / Judge (Offline Only)

**Family:** Large language model (8B+), frontier quality.  
**Used in:** LinkedIn SAGE (8B) — generates relevance grades for training data.  
**Purpose:** Produce high-quality relevance labels at scale that would otherwise require
human annotation. Run offline — never serves live traffic.

**Example input:**
```
System: "You are an expert recruiter. Rate the relevance of this job for this query
         on a scale of 1-4. Provide a grade and brief reasoning."

User:   "Query: senior ML engineer, remote, TensorFlow

         Job: Junior Data Analyst | On-Site | Chicago
         Skills: Excel, SQL, Tableau"
```

**Example output:**
```
Grade: 1
Reasoning: Seniority mismatch (junior vs senior), role mismatch (analyst vs engineer),
           location mismatch (on-site vs remote), no TensorFlow.
```

These (query, document, grade) tuples become the training data for smaller models
(Relevance Teacher 1.7B → Student SLM 0.6B) via distillation.

**Key trait:** quality of a 70B+ model, economics of running it only once per
training example — not per live request.

---

## 6. LLM for Summarization (RL-trained)

**Family:** Language model with RL fine-tuning (GRPO).  
**Used in:** LinkedIn Semantic Search — 1.7B summarization model for member profiles
and job descriptions.  
**Purpose:** Compress long heterogeneous documents into short, factual, engagement-
predictive summaries before they are fed to the SLM ranker.

**Example input:**
```
Member profile (raw, ~1500 tokens):
  Name: Alice Chen
  Headline: Senior ML Engineer at TechCorp
  Summary: Passionate about NLP and large-scale ML systems...
  Experience: [3 roles, 8 years total, last 3 at TechCorp doing LLM fine-tuning]
  Skills: [47 skills listed]
  Activity: [last 10 jobs interacted with: 7 ML engineer roles, 3 data science roles]
  Connections: 1,243 | Network: mostly ML/AI professionals
```

**Example output:**
```
Senior ML Engineer, 8 years exp, specializes in LLM fine-tuning and NLP.
Active job seeker (ML engineer roles). Strong ML/AI network.
```

**What the RL reward trains for:** summary length (penalty for too long), factual
accuracy (score from a 32B judge), and engagement predictiveness (whether the
engagement teacher's score improves with this summary vs without).

---

## Summary — Which Model for Which Job

| Use Case | Model Type | Input | Scales to | LinkedIn example |
|---|---|---|---|---|
| Classic feature scoring | DLRM | Feature vectors | Billions of items | Prior Feed ranker |
| User history modeling | Sequential Transformer | Sequence of past interactions | 1000-item histories | Feed SR |
| Text-based retrieval | LLM Bi-Encoder | Raw text (each side independently) | 1B+ docs (pre-computed) | Semantic Search retrieval |
| Precise text-based ranking | LLM Cross-Encoder / SLM | Raw text (query + doc together) | ~1000 candidates | Semantic Search SLM ranker |
| Label generation | LLM Oracle (8B+) | Raw text + instructions | Offline only | SAGE judge |
| Document compression | LLM Summarizer (RL) | Long raw text | Offline + nearline | Profile/job summarization |

### The key question to ask: "what is the transformer processing?"

- **A sequence of past user interactions?** → Sequential Recommender (Feed SR)
- **Two text strings independently?** → Bi-Encoder / Two-Tower
- **Two text strings concatenated?** → Cross-Encoder / SLM Ranker
- **One long text to compress?** → Summarization LLM
- **A (query, doc) pair to grade?** → Oracle / Judge LLM
- **Feature vectors, no text?** → DLRM (no transformer)
