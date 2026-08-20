# Embeddings — ML System Design Reference

Examples drawn from LinkedIn Feed (member + post embeddings).

---

## 1. How Embeddings Are Trained — Contrastive Learning

### The Core Idea

An embedding model learns to map inputs into a vector space where **similar items are
close** and **dissimilar items are far apart**. The training signal comes from pairs of
(query, relevant item) and (query, irrelevant item) — not from explicit labels.

This is contrastive learning: push positives together, push negatives apart.

### Evolution: Matrix Factorization → Two-Tower → LLM Two-Tower

Understanding why two-tower exists requires understanding what came before it.

**Matrix Factorization — purely interaction-driven embeddings:**

You have a user-item interaction matrix R (users × items), where each cell is an
engagement signal (rating, click, view). MF decomposes it into two embedding matrices:

```
              post_A  post_B  post_C  post_D
member_Alice    5       0       3       0
member_Bob      0       4       0       5
member_Carol    2       0       0       4

R (users × items) ≈ U (users × d) × Vᵀ (d × items)

score(user, item) = dot(U[user_id], V[item_id])
```

U and V are the learned parameters — just lookup tables, one row per user/item.
No features. No encoder. The model learns purely from interaction patterns:
"Alice and Carol both liked post_A → they are similar → surface Carol's liked posts to Alice."

**The fundamental limitation: MF solely depends on interaction data.**
If that source is not there, it does not learn:
- **New member** joins with no interactions → no row in U → no embedding → no recommendations
- **New post** published seconds ago → no interactions yet → no row in V → never retrieved
- **Niche topic** with sparse interactions → embedding is noisy, undertrained
- **Long-tail items** (most posts) never get enough interactions to learn good embeddings

**Two-tower — learns from features, not just interactions:**

```
Matrix Factorization:
  embedding(user) = U[user_id]              ← lookup, needs interactions to train
  embedding(item) = V[item_id]              ← lookup, needs interactions to train

Two-Tower:
  embedding(user) = encoder_U(user_features)  ← features → encoder → embedding
  embedding(item) = encoder_V(item_features)  ← features → encoder → embedding
```

Same dot product objective — but now a new member with a job title and skills list
gets a meaningful embedding immediately from those features, without any interactions.
A new post gets an embedding from its text content the moment it is published.

Two-tower can still learn from other features provided even when interaction data is
absent — profile, content, metadata are sufficient for a first-pass embedding.

**LLM two-tower — adds world knowledge on top:**

Even a feature-based two-tower has limits without LLM:
- A new member whose profile says "Software Engineer in San Francisco"
- Non-LLM encoder: maps location to co-occurrence patterns seen in training — "San Francisco"
  matches jobs that literally say "San Francisco"; no generalization to nearby regions
- LLM encoder: world knowledge knows San Francisco is inside the Bay Area → surfaces
  jobs tagged "Bay Area", "South Bay", "Peninsula" immediately, without any training examples
  of SF↔Bay Area job matching

LLM reduces the cold start problem dramatically because it does not wait for
co-occurrence patterns in training data — it brings pretraining knowledge from
internet-scale text directly into the embedding.

```
Cold start severity:

Matrix Factorization:   ████████████  severe — zero signal without interactions
Two-Tower (no LLM):     ████░░░░░░░░  moderate — features help, but shallow semantics
LLM Two-Tower:          █░░░░░░░░░░░  minimal — world knowledge fills the gap immediately
```

**Summary of progression:**

| | MF | Two-Tower | LLM Two-Tower |
|---|---|---|---|
| Learns from | Interactions only | Features + interactions | Features + interactions + world knowledge |
| Cold start | Breaks completely | Partial — features help | Near-solved via world knowledge |
| Semantic depth | None (co-occurrence) | Shallow (trained features) | Deep (LLM pretraining) |
| New item | No embedding until interactions | Embedding from content immediately | Rich embedding from content immediately |
| Example failure | New member → no recs | "SF engineer" → misses Bay Area jobs | Handles it via geographic world knowledge |

---

### Two-Tower Architecture

Both the member and the post pass through their own encoder tower. The towers have
separate parameters and separate architectures — but both output into the SAME vector
space. That shared space is the whole point: it makes dot products meaningful.

```
Member Tower                        Post Tower
────────────────────────            ──────────────────────────
professional title / skills         post text
past engagement history             author identity + connections
connections graph                   hashtags / topics
job function, seniority             post type (article, video, job)
        ↓                                       ↓
   Encoder (MLP /                      Encoder (MLP /
   Transformer)                        LLM encoder)
        ↓                                       ↓
  member_embedding ∈ R^d           post_embedding ∈ R^d
        └──────────────────┬──────────────────┘
                      dot product
                   (relevance score)
```

Both towers project into the **same d-dimensional space** so dot products are meaningful.

---

### Why Two Separate Towers? — Modality Difference

The short answer: use the **same encoder** when both inputs are the same modality.
Use **separate encoders** when inputs have fundamentally different structures.

**Same encoder works (text + text):**
```
Sentence similarity:  "I love pizza"  vs  "I enjoy pasta"
                       ↓ same BERT encoder ↓
                       embedding_A        embedding_B
                       └──── cosine similarity ────┘
```
Both inputs are natural language text — the same encoder naturally handles both.
Used in: sentence-BERT, DPR (Dense Passage Retrieval), semantic search.

**Two-tower needed (different modalities):**

Member and post inputs are structurally very different:
```
Member features:                        Post features:
  job title          ← text               post content      ← text
  skills list        ← text               hashtags          ← text
  connections graph  ← graph structure    author metadata   ← structured
  interaction history← time series        engagement counts ← numerical
  seniority level    ← ordinal number     post type         ← categorical
  industry code      ← categorical
```

A text encoder cannot natively process a graph embedding or a time-ordered interaction
sequence. A graph encoder cannot process raw text. The two sides need different
architectures that each understand their native input format.

**The modality mismatch problem:**
```
Option A — force same encoder:
  Run member features through a text encoder → crush graph/sequential structure into
  a bag of tokens → lose rich structural signal

Option B — separate encoders:
  Member tower: GNN for connections graph + Transformer for interaction sequence + MLP for tabular
  Post tower:   LLM encoder for text content
  Each encoder is optimal for its input type
  Both independently project to same R^d output space
  Dot product is still valid — same dimensionality, jointly trained
```

**There is ONE shared vector space, not two.**

This is the critical point: the two towers have separate parameters and separate
architectures, but they both output into the SAME vector space. Same dimensionality
AND same meaning per dimension:

```
dimension 7 in member_embedding  ←→  dimension 7 in post_embedding
         both = "professional expertise / content depth"
```

If the spaces were truly separate, dot products would be meaningless — multiplying
apples by oranges. The shared space is what makes the dot product a valid relevance score.

With a siamese encoder (same weights, same input type — e.g. sentence-sentence), the
shared space is guaranteed by construction. With two-tower (different encoders), the
shared space is learned via the contrastive loss:

```
Early training:  member[7] = "seniority",  post[7] = "post length"   → dot product = noise
After training:  member[7] = "expertise level", post[7] = "content depth" → dot product = signal
```

Neither tower knows the other's architecture — but they are jointly penalized whenever
a positive pair (member, post they engaged with) is not close in this shared space.
Over millions of training steps, both towers learn to project their very different inputs
into the same semantic space where dot products are meaningful.

---

### Solving the Modality Problem — LinkedIn's Prompt Library Approach

LinkedIn took a different path: instead of separate encoders, they convert ALL features
to text first via a prompt library, then run the **same LLM** for both sides.

```
Member structured features              Post structured features
        ↓  prompt library                       ↓  prompt library
Member text prompt                      Post text prompt
        ↓                                       ↓
   Same LLM encoder                  Same LLM encoder (shared weights)
        ↓                                       ↓
  member_embedding ∈ R^d             post_embedding ∈ R^d
        └───────────── cosine similarity ────────┘
```

**Prompt templates:**
```
Member prompt:
  "Profile: Senior ML Engineer, skills: Python NLP,
   Work: Google → Meta → current
   Education: MS CS, Stanford
   Recently engaged with (chronological):
     [scaling laws for LLMs, t-3d]
     [hiring post at OpenAI, t-7d]
     [intro to Python, t-14d]"

Post prompt:
  "Format: article
   Author: AI researcher, DeepMind, Technology industry
   <view_percentile>91</view_percentile>
   <engagement_rate>85</engagement_rate>
   Post: New findings on scaling laws for fine-tuning LLMs..."
```

**The percentile bucket technique — solving numerical features in LLMs:**

Raw numbers tokenize poorly because LLMs process text character by character:
```
"views: 12345"  →  tokens: ["views", ":", "1", "2", "3", "4", "5"]
                →  5 separate digit tokens, no ordinal meaning learned
                →  correlation between popularity and similarity score: -0.004
```

Percentile buckets with special tokens:
```
"views: 12345"  →  <view_percentile>71</view_percentile>
                →  single token "71" = "this post is in the 71st percentile"
                →  model learns stable vocabulary: <30 = low, 40-60 = avg, >90 = viral
                →  correlation improvement: 30×
                →  recall@10 improvement: 15%
```

Applied to all numerical features: engagement rates, recency signals, affinity scores,
connection strength. Any continuous value → percentile bucket → single special token.

**Why same LLM works here (and the advantages it brings):**

| | Separate encoders | Same LLM + prompt library |
|---|---|---|
| Handles heterogeneous features | Natively (each encoder is purpose-built) | Via serialization (prompt library) |
| World knowledge | Only where pre-trained | Full LLM pretraining on both sides |
| Cold start | Needs engagement history | Job title alone → meaningful embedding |
| Semantic depth | Shallow co-occurrence | "San Francisco" → "Bay Area" jobs missed |
| Maintenance | Multiple separate systems | One unified retrieval system |
| Same vector space | Learned jointly via loss | Guaranteed — same weights define the space |

**Cold start example:**
```
New member, no engagement history:
  Profile only: "Software Engineer in San Francisco"

Non-LLM embedding: matches only jobs literally tagged "San Francisco" — co-occurrence
                   patterns from training data; no generalization to nearby regions

LLM embedding: world knowledge knows SF ⊂ Bay Area → also surfaces jobs tagged
               "Bay Area", "South Bay", "East Bay", "Peninsula"
               → candidate sees the full relevant job market immediately,
                 without any training examples of SF↔Bay Area job matching
```

This is why LinkedIn replaced multiple specialized retrieval sources (chronological index,
trending content, collaborative filtering, topic-based) with a single unified LLM retrieval
system — one embedding captures what previously required many separate systems.

---

### Training Objective — InfoNCE Loss

For each training example you have:
- One **positive pair**: (member, post they engaged with) — liked, commented, shared, clicked
- N **negatives**: (member, posts they did not engage with)

```
For a batch of B examples, each with 1 positive and (B-1) in-batch negatives:

loss = -log [ exp(sim(m, p) / τ) / Σ exp(sim(m, nᵢ) / τ) ]

where:
  sim(m, p)  = dot(member_embedding, post_embedding_positive)
  sim(m, nᵢ) = dot(member_embedding, post_embedding_negative_i)
  τ          = temperature (softmax sharpness, typically 0.05–0.1)
```

Intuition: the model is trained to rank the positive post higher than all negatives.
The loss is low when the positive has a much higher dot product than all negatives.

Low temperature τ → sharper distribution → model is penalized more for any positive
that doesn't rank clearly above all negatives. High τ → softer, easier training signal.

### In-Batch Negatives

The simplest and most GPU-efficient negative sampling strategy:
- In a batch of B (member, engaged_post) pairs, each member treats the other (B-1)
  positive posts as its negatives
- No extra data pipeline — negatives come for free from the batch
- GPU-efficient: one forward pass computes a B×B similarity matrix; diagonal = positives

```
Batch of 4 pairs:
  (member_A, post about Python engineering)
  (member_B, post about product management)
  (member_C, post about job search tips)
  (member_D, post about AI research paper)

For member_A (software engineer):
  positive  = post about Python engineering
  negatives = product management post, job search post, AI paper  ← other batch items
```

Larger batch = more negatives per member = harder training signal = better embeddings.
This is why two-tower training uses large batch sizes (4096, 8192).

### Hard Negatives

In-batch negatives are mostly **easy negatives** — random posts the member didn't engage
with are obviously irrelevant (a job search post shown to a senior engineer).

Hard negatives are posts that are **close in embedding space but not relevant**:
- A post on the same topic that appeared in the feed but was scrolled past (impression, no engagement)
- A post from a 2nd-degree connection on a topic the member usually engages with, but skipped

```
Easy negative: member (ML engineer) shown post about wedding photography tips
               → model trivially learns to separate these

Hard negative: member (ML engineer) shown post about "intro to Python for beginners"
               → topic is relevant (Python, engineering) but seniority mismatch
               → forces model to learn professional seniority alignment
```

**Mining hard negatives:**
1. **Offline**: run current model, find posts with high similarity score but no engagement
   (appeared in feed, member scrolled past — impression without click)
2. **Online (ANN retrieval during training)**: for each member, retrieve top-K posts from
   current index, use non-engaged posts as hard negatives

Hard negatives dramatically improve recall@K — the model learns fine distinctions like
seniority level, niche vs mainstream within a topic, and author relationship strength.

### LinkedIn Feed Example

```
Positive pair:
  member (ML engineer, follows LLM researchers, engaged with 3 AI papers this week)
  + post ("New paper: scaling laws for fine-tuning LLMs — key findings")
  → high dot product, low loss

Easy negative:
  same member + post ("5 tips for your first job interview")
  → low dot product trivially learned

Hard negative:
  same member + post ("Getting started with machine learning in Python")
  → high dot product without hard negative training (ML + Python = match)
  → after hard negative training: model learns member is senior,
    intro-level content is not relevant despite topic overlap
```

---

## 2. Graph Embeddings — Inductive Methods (GNN)

### Why Graphs Carry Signal Text Features Miss

LinkedIn is fundamentally a graph. Members, companies, skills, and posts are nodes.
Connections, follows, and endorsements are edges. That structure is real signal.

```
member_A ──connected to──→ member_B ──connected to──→ member_C
    │                           │
    └──follows──→ OpenAI         └──follows──→ AI researcher Y
    │
    └──has_skill──→ "Machine Learning"
                         │
                    endorsed_by 47 members
```

**The software engineer problem:**

```
Profile text alone: "Software Engineer at a tech company"
                     ↑ could be iOS, backend, ML, DevOps, embedded — no way to tell

But the graph reveals:
  connected to:  50 ML researchers
  follows:       OpenAI, DeepMind, Hugging Face
  2nd-degree:    300+ more ML practitioners
  skill endorsements: "PyTorch" endorsed by 30 ML engineers

Graph embedding: places this member deep inside the ML/AI community
                 → LLM papers, AI hiring posts, research content surface immediately
```

Text features say "Software Engineer." The graph says "ML community member."
For feed personalization, the graph wins — it captures who you actually are in the
professional network, not just what your title says.

---

### Inductive GNN — Learning a Function, Not a Lookup Table

The key idea: instead of learning one fixed embedding per node (which breaks for new nodes),
learn a **function** that produces an embedding from a node's own features and its
local neighborhood.

```
embedding(node) = f(own_features, neighborhood)

For member_A:
  embedding(A) = f(
      own features:         [job title, skills, work history, education],
      neighbor embeddings:  aggregate(embeddings of A's connections)
  )
```

**GraphSAGE — the standard inductive GNN:**

GraphSAGE (Sample and Aggregate) computes a node's embedding in layers:

```
Layer 1 — aggregate immediate neighbors:
  h_neighbor = MEAN(embeddings of A's direct connections)
  h_A^(1)    = σ( W · concat(h_A^(0), h_neighbor) )

Layer 2 — aggregate 2-hop neighbors:
  h_A^(2)    = σ( W · concat(h_A^(1), aggregate(h_neighbor^(1))) )

Final embedding = h_A^(2)
```

Each layer expands the neighborhood radius by one hop. Two layers = information from
2-hop neighbors flows into A's embedding. The model sees not just who A is connected to,
but who those people are connected to.

**What each layer captures for LinkedIn:**
```
Layer 1 (1-hop): immediate connections
  → "A is connected to 50 people, mostly ML engineers at FAANG companies"

Layer 2 (2-hop): connections of connections
  → "A's network is embedded in the broader ML research + industry community"
  → surfaces signal that A's own profile text doesn't contain
```

**Why inductive matters for LinkedIn:**

New member joins with 5 initial connections:
```
Transductive (node2vec): no embedding → no recommendations until full graph retrain
                         unacceptable — LinkedIn adds millions of members monthly

Inductive (GNN):         embedding(new_member) = f(profile_features, 5 neighbors)
                         meaningful embedding from day 1, no retraining required
```

When A makes a new connection to B:
```
Recompute embedding(A) using updated neighborhood — just one forward pass
No retraining, no global graph operation
```

---

### How Graph Embeddings Compose with the Two-Tower

The GNN does not replace the two-tower — it feeds into it as one feature among many.
This is the standard production pattern: GNN produces a node embedding, that embedding
becomes a feature in the member tower.

```
Member Tower inputs:
  ┌──────────────────────────────────────────────────┐
  │  profile text + skills         ← text features   │
  │  work history + education      ← text features   │
  │  interaction history           ← temporal signal  │
  │  GNN embedding ◄───────────────── graph signal    │
  └──────────────────────────────────────────────────┘
                    ↓
              Member encoder (LLM)
                    ↓
            member_embedding ∈ R^d
```

Each feature type contributes a different kind of signal:

| Feature | What it captures |
|---|---|
| Profile text | Stated identity — title, skills, education |
| Interaction history | Revealed preferences — what you actually engaged with |
| GNN embedding | Social identity — who you are in the professional network |

The GNN embedding captures the social graph signal that neither text nor interaction
history can express: your position in the professional community, the collective
expertise of your network, the topics your connections care about.

**Concrete example — three members, same profile text:**
```
All three: "Software Engineer, Python, 5 years experience"

Member A: connected to ML researchers, follows AI companies
  GNN embedding → ML/AI community → feed: LLM papers, AI hiring, research
  
Member B: connected to iOS developers, follows Apple, mobile startups
  GNN embedding → Mobile community → feed: Swift posts, app design, WWDC coverage

Member C: connected to DevOps engineers, follows cloud vendors
  GNN embedding → Infrastructure community → feed: Kubernetes, cloud architecture, SRE

Profile text is identical. Graph makes them completely different people in the feed.
```

---

### Combining GNN Embeddings with LLM Encoder — The Multimodality Problem

A GNN embedding is a raw float vector: `[0.23, -0.81, 0.44, ...]`. You cannot paste
that into a text prompt. The LLM tokenizer has no vocabulary for raw floats.

Two practical solutions:

---

**Approach 1 — Serialize graph signal to text**

Don't pass the raw GNN vector. Convert graph structure into human-readable text
before it hits the LLM:

```
Raw GNN output: [0.23, -0.81, 0.44, ...]   ← cannot go into prompt

Serialize instead:
  "Network: primarily connected to ML researchers and AI practitioners
   Community: AI/ML cluster
   2nd-degree reach: 300+ members in AI industry"
         ↓
Added to member text prompt → same LLM encodes everything
```

LinkedIn's engagement history in the prompt already does this implicitly:
"recently engaged with: LLM paper, AI hiring post, research article" = graph
signal (who you interact with) serialized as text. The LLM picks up the
community signal from the content of what you engaged with, not from the
raw graph structure.

Tradeoff: loses precision (128 floats → a few labels) but keeps the pipeline
simple — one LLM encoder handles everything uniformly.

---

**Approach 2 — Late fusion (most common in production)**

Each modality gets its own encoder. Outputs are fused after encoding via a small MLP:

```
Text features (profile, skills, engagement history)
        ↓
   LLM encoder
        ↓
text_embedding ∈ R^d          GNN embedding ∈ R^d'
        └──────────┬───────────┘
          concat([text_emb, gnn_emb])
                   ↓
                  MLP
                   ↓
          final member_embedding ∈ R^d
```

The MLP learns how much weight to give each modality from training signal:
- Rich engagement history → text_embedding carries more weight
- New member, few interactions but strong connections → GNN embedding carries more weight

Each encoder is optimal for its modality — LLM for text semantics, GNN for graph
structure — and the MLP fuses them into one final embedding for the two-tower dot product.

**Why MLP — what it actually learns**

A raw concatenation `[text_emb, gnn_emb]` stacks two vectors from different semantic
spaces. A dot product over this treats all dimensions equally, which is wrong. The MLP
learns how the two spaces interact — specifically, cross-modal combinations that neither
embedding alone can express.

```
text_emb dim 7  = "seniority signal"
gnn_emb  dim 3  = "ML community membership"

Interesting signal is their COMBINATION:
  high seniority + in ML community = "senior researcher" → surfaces papers
  low seniority  + in ML community = "junior learner"   → surfaces tutorials

A dot product over raw concatenation cannot detect this AND condition.
The MLP can — through its weight matrix and non-linearity.
```

**What happens inside the projection (e.g. 256 → 1024):**

Each output dimension is a weighted sum of all input dimensions, gated by ReLU:

```
h_i = ReLU( W[i,:] · v )    W is (1024 × 256)
```

Each row of W is a feature detector — fires when a specific pattern in the input is
present. With 1024 rows you have 1024 detectors. Going higher-dimensional lets the
network disentangle features that were compressed together in the input:

```
256-dim input:
  dim 47 ≈ "seniority + industry" combined  ← entangled, can't separate

1024-dim projected space:
  dim 312 = "high seniority in tech"         ← disentangled
  dim 313 = "high seniority in finance"      ← disentangled
  dim 314 = "junior in ML community"         ← new interaction feature
```

**The non-linearity is critical — without it, depth adds nothing:**

```
Without activation (linear only):
  W2 × (W1 × x) = (W2·W1) × x   ← collapses to one matrix, no expressiveness gained

With ReLU:
  W2 × ReLU(W1 × x)              ← ReLU creates AND conditions, detects combinations
```

**Typical shape — expand then compress:**

```
concat [text(512) + gnn(256)] = 768-dim
        ↓  W1 (1024 × 768) + ReLU
      1024-dim  ← space to represent interaction features
        ↓  W2 (256 × 1024) + ReLU
       256-dim  ← compress back to match post embedding; forces network to keep
                  only the most useful combinations, discards noise
```

**Summary:** The MLP doesn't blend embeddings — it learns new features that are
functions of combinations of the original features, which neither embedding alone
could express. Higher dimensionality gives it space to represent those combinations
without collision; non-linearity is what makes AND-type conditions learnable.

---

**Which approach when:**

| | Approach 1 (serialize to text) | Approach 2 (late fusion) |
|---|---|---|
| Complexity | Low — one LLM for everything | Medium — separate encoders + MLP |
| Graph information preserved | Partial — discretized to labels | Full — raw vector preserved |
| Graph-text interaction | LLM attends over serialized text | None — fused after encoding |
| Cold start | Engagement history → implicit graph signal | GNN embedding contributes directly |
| LinkedIn usage | Engagement history in prompt | GNN + LLM fused for richer member signal |

---

## 3. Bi-Encoder vs Cross-Encoder

### Bi-Encoder (Two-Tower) — Used at Retrieval

Each input is encoded **independently**. The similarity is computed as a single dot product
after both encodings are done.

```
member ──→ Encoder A ──→ member_embedding  ─┐
                                             ├──→ dot product → score
post   ──→ Encoder B ──→ post_embedding    ─┘
```

**Key property: post embeddings are pre-computed offline.**
At feed generation time, you only run the member tower. The post tower ran during the
nearline refresh pipeline (triggered by new post CDC event). This is what makes retrieval fast.

```
Feed generation time cost:
  1. Run member tower: ~5ms (one forward pass, or fetch pre-computed from Redis)
  2. Fetch pre-computed post embeddings: ~2ms (Redis pipeline for candidates)
  3. N dot products: ~1ms (matrix multiply)
  Total: ~8ms
```

**Limitation:** the two inputs never attend to each other during encoding. The model
cannot capture fine-grained interaction between member context and post content —
each embedding is computed in isolation, without knowing what it will be compared against.

Example of what bi-encoder misses:
```
Member context: "currently job hunting, open to ML roles in NYC"
Post: "We're hiring ML engineers in NYC — apply now!"

Bi-encoder: scores based on general member interest in ML + job posts
Cross-encoder: sees "job hunting" + "hiring" + "NYC" together → much higher relevance score
```

### Cross-Encoder — Used at Re-ranking

Both inputs are **concatenated and encoded together**. The model sees the full interaction
between member and post at the same time.

```
[member context + post content] ──→ Transformer encoder ──→ relevance score
```

```
Input to transformer:
  [CLS] member: ML engineer, NYC, open to work, follows NLP researchers [SEP]
        post: "Hiring ML engineers in NYC — LLM fine-tuning team, remote-friendly" [SEP]
        ──→ relevance score: 0.97

vs.

  [CLS] member: ML engineer, NYC, open to work, follows NLP researchers [SEP]
        post: "Hiring ML engineers in London — relocation required" [SEP]
        ──→ relevance score: 0.41   ← cross-encoder catches geo mismatch; bi-encoder might not
```

Cross-attention layers let the model ask: "does 'NYC' in the member context match 'NYC'
in the post?" — a connection the bi-encoder cannot make since each side encodes independently.

**Limitation: cannot pre-compute.** You must run a full forward pass for every
(member, candidate_post) pair at feed generation time. At 500 candidates × 10ms = 5 seconds.
Completely unusable for retrieval over millions of posts.

### Why Each Stage Uses a Different Architecture

```
Stage 1 — Retrieval (GR system — Graph Retrieval):
  Candidate pool: millions of posts (connections' posts, followed topics, viral content)
  Output: ~500 candidates
  Latency budget: ~10ms
  → Must be pre-computed. Bi-encoder is the only option.

Stage 2 — Re-ranking (LiNR — LinkedIn Relevance):
  Input: ~500 candidates
  Output: top 20 feed items
  Latency budget: ~50ms
  → 500 cross-encoder calls at ~0.05ms each = ~25ms. Feasible.
  → Full member-post interaction → better relevance quality
```

The pipeline buys quality cheaply: use fast bi-encoder to narrow millions → 500,
then spend the re-ranking budget on accurate cross-encoder scoring over 500.

### Quality vs Latency Tradeoff

| | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Encoding | Independent | Joint (member + post together) |
| Post embeddings | Pre-computed offline | Cannot pre-compute |
| Feed gen time cost | O(1) dot products | O(N) forward passes |
| Interaction modeling | None (dot product only) | Full cross-attention |
| Relevance quality | Lower | Higher |
| Scales to | Millions of posts | ~500 candidates max |
| Used at | Retrieval (GR) | Re-ranking (LiNR) |

### LinkedIn Feed Pipeline with Both

```
Millions of posts in the candidate pool
        ↓
Graph-based filtering (connections, follows, topics)  [no embeddings]
        ↓
Bi-encoder dot product: 500 × dot(member_emb, post_emb)   [~1ms, pre-computed]
        ↓
Top 100 candidates
        ↓
Cross-encoder re-ranking: 100 × forward pass              [~5ms total]
        ↓
MoL (Mixture of Logits) final score: relevance + virality + diversity
        ↓
Top 20 feed items returned
```

The cross-encoder stage captures fine-grained member-post interaction that the
bi-encoder dot product cannot — particularly useful for:
- Seniority matching (senior member should not see intro-level content)
- Geo-specific posts (job posts, local events)
- Nuanced professional intent ("open to work" member gets job posts ranked higher)

---

## 4. Embedding-Based Retrieval — RecSys vs Semantic Search

Both RecSys and Semantic Search use the same bi-encoder / two-tower architecture for
retrieval. Yet in RecSys, bi-encoder is good enough to serve as Stage 1 without much
concern, while in Semantic Search it is explicitly called out as having a "bottleneck"
requiring a cross-encoder ranker on top. The architecture is identical — so what's
different?

### The Gap Exists in Both — But Its Size Differs

The bi-encoder gap: a document/post embedding is fixed and pre-computed offline. It
cannot specialize for a specific query at inference time. The embedding must represent
the item generically for all possible queries.

This gap is the same structurally. But how much it hurts depends on what "relevance"
means in each domain.

---

### RecSys — Soft Interest Alignment

The "query" in RecSys is a member's interest profile — broad, stable, slowly changing.

```
Member: "Alice — ML engineer, follows AI researchers, engages with career posts"
        ↓
member_embedding captures: "likes ML content, career growth, tech news"

Any ML post embedding lands nearby → retrieved → acceptable
```

Relevance = does this member **generally** engage with this type of content?

- Interests are **broad** — many posts qualify
- The match is about **affinity**, not constraint satisfaction
- Some imprecision is tolerable — showing a slightly off-topic ML post is fine, even
  valuable (exploration, serendipity)
- The "query" (member profile) is stable — the embedding can be cached, it doesn't
  change per request

The bi-encoder gap is small here: a post embedding that's generically "ML-ish" will
land near Alice's embedding. Close enough → retrieved → re-ranker handles the rest.

→ See `llm_in_recsys.md §2` for the full retrieval flow (dual-encoder, ANN index,
hard-negative mining).

---

### Semantic Search — Hard Constraint Satisfaction

The "query" in Search is an ephemeral, specific text string — different every request.

```
Query: "senior ML engineer, remote, TensorFlow"

This has hard constraints:
  - seniority = senior      (junior ML engineer → WRONG)
  - modality  = remote      (on-site ML engineer → WRONG even if otherwise perfect)
  - skill     = TensorFlow  (PyTorch engineer → possibly WRONG)
```

Relevance = does this document satisfy **all constraints simultaneously**?

A job posting for "junior ML engineer, on-site, PyTorch" is semantically close in
embedding space (it IS an ML engineering job) but is actually irrelevant. The
bi-encoder cannot distinguish "remote" from "on-site" at retrieval time because the
document embedding was computed without knowing this specific query's constraints.

The bi-encoder gap is large here:
- One token difference ("remote" vs "on-site") flips relevance entirely
- The query is ephemeral — it changes completely between two searches
- Showing a wrong result (non-remote job when user filtered remote) is a direct failure
- No tolerance for imprecision — users have explicit intent

---

### Why Both Systems Still Use Bi-Encoder

Despite the gap, bi-encoder is used for Stage 1 retrieval in **both** systems — because
there is no faster alternative at scale:

```
RecSys (LinkedIn Feed):
  10M+ posts → bi-encoder retrieval → ~500 candidates → cross-encoder re-ranker (LiNR)

Semantic Search (LinkedIn Jobs/People):
  1B+ documents → bi-encoder retrieval → top 1000 → SLM cross-encoder ranker
```

Bi-encoder is not "the solution" in either system — it is always Stage 1. The
difference is what Stage 2 compensates for:

| | RecSys | Semantic Search |
|---|---|---|
| Stage 1 | Bi-encoder (two-tower) | Bi-encoder (LLM bi-encoder) |
| Stage 2 | Cross-encoder + feature-rich DLRM | Cross-encoder SLM ranker |
| Stage 2 compensates for | Engagement features, social signals | Query-specific token-level constraints |
| Tolerance for retrieval error | Higher — interests are broad | Lower — constraints are precise |
| "Query" nature | Stable member profile | Ephemeral specific text |

**The key insight:** bi-encoder gap is acceptable in RecSys because the retrieval task
is soft (general interest alignment), and the re-ranker compensates with feature
richness. In Semantic Search, the gap is more dangerous because constraints are hard
— a retrieved-but-wrong candidate wastes ranking budget and harms precision. The SLM
cross-encoder compensates with query-document token-level attention.

---

## 5. Pre-trained + Fine-tuning

### Why Not Train From Scratch

Training embeddings from scratch means teaching the model everything: grammar,
vocabulary, that "startup" and "founder" are related, that "ML" and "neural networks"
overlap, that professional seniority is a meaningful concept. That knowledge already
exists in models trained on internet-scale text. Relearning it wastes compute and data.

```
Training from scratch:
  Model learns: "startup" ↔ "founder" ↔ "VC" ↔ "seed round"   ← already on the internet
  Model learns: LinkedIn-specific engagement patterns            ← only you have this data
  Cost: millions of GPU hours
  Data needed: massive

Pre-trained + fine-tuned:
  Pre-trained model already knows the first part
  Fine-tuning only teaches the second part
  Cost: fraction of training from scratch
  Gets you ~80% of the way with ~10% of the compute and data
```

For most domains, you should not be training embeddings from scratch in the modern era.
Someone else already spent the GPU hours — use their weights as a starting point.

---

### What Pre-training Gives You (for Free)

Pre-trained encoders like BERT, sentence-transformers, or general LLMs already know:

```
General language understanding:
  "startup" ↔ "founder" ↔ "entrepreneur"
  "ML" ↔ "machine learning" ↔ "neural networks" ↔ "deep learning"
  "San Francisco" → inside Bay Area → South Bay, Peninsula, East Bay all nearby

Contextual meaning:
  "Java" in a programming post ≠ "Java" in a travel post
  "model" in an ML post ≠ "model" in a fashion post

Semantic similarity:
  "fired" ≈ "laid off" ≈ "let go"   (different words, same concept)
  "LLM" ≈ "large language model" ≈ "GPT-style model"
```

None of this needs to be relearned. The pre-trained weights carry it all.

---

### What Fine-tuning Teaches

Fine-tuning on your task-specific data teaches the model what matters in YOUR domain:

```
LinkedIn fine-tuning on member-post engagement data teaches:

  Professional seniority signal:
    Senior ML engineers engage with research papers, architecture posts
    Junior engineers engage with tutorials, career advice
    → embedding space warps to separate these even when topics overlap

  Domain-specific relevance:
    A "scaling laws" paper is highly relevant to an LLM researcher
    but less relevant to an ML engineer focused on recommendation systems
    → fine-tuning on engagement data teaches this distinction

  LinkedIn vocabulary:
    "open to work" = job seeking signal
    "thrilled to announce" = career milestone
    "thoughts?" at end of post = discussion-seeking
    → model learns these professional writing patterns
```

What pre-training got right stays right. Fine-tuning only adjusts weights in the
direction of your task signal — it doesn't erase general language knowledge.

---

### The Typical Flow

```
Step 1 — Start from pre-trained encoder
  Load BERT / sentence-transformer / general LLM weights
  These weights encode internet-scale language knowledge

Step 2 — Fine-tune on task-specific data (contrastive / two-tower setup)
  Positive pairs: (member, post they engaged with)
  Negatives:      in-batch negatives + mined hard negatives
  Loss:           InfoNCE
  Epochs:         few (not hundreds — you're adjusting, not relearning)

Step 3 — Deploy
  Pre-compute all post embeddings offline using fine-tuned encoder
  At query time: run member tower, dot product against post embeddings
```

**LinkedIn's approach exactly:**
LinkedIn distilled and fine-tuned an LLM using millions of member-to-post engagement
pairs from Feed data. The pre-trained LLM brought world knowledge (SF ⊂ Bay Area →
surface Bay Area jobs for SF-based profiles). Fine-tuning on engagement data taught it
LinkedIn-specific relevance — which posts senior ML engineers actually click vs scroll past.

---

### Pre-training vs Fine-tuning — What Each Layer Learns

```
Pre-trained LLM (frozen knowledge):
  Layer 1-6  (early): tokenization, syntax, basic word relationships
  Layer 7-18 (mid):   semantic concepts, entity types, topic clusters
  Layer 19-24(late):  task-specific representations ← most affected by fine-tuning

Fine-tuning adjusts mostly the later layers:
  "ML engineer engaging with this post" → what does that signal about relevance?
  Early layers stay mostly intact — grammar and basic semantics don't change
```

This is also why **parameter-efficient fine-tuning** (LoRA, adapters) works well for
embeddings: you can freeze most of the pre-trained weights and only train a small set
of added parameters. The pre-trained knowledge is preserved; only the task-specific
signal is injected. Useful when you have limited engagement data or want to fine-tune
quickly as the platform evolves.

---

## 6. Embeddings and RAG Systems

### The Problem RAG Solves

LLMs have two hard limitations:
- **Fixed context window** — you cannot paste 10,000 documents into a prompt
- **Frozen knowledge** — the LLM does not know anything that happened after its training cutoff

RAG (Retrieval-Augmented Generation) solves both. Instead of asking the LLM to know
everything, you retrieve the relevant documents at query time and give them to the LLM
as context. The LLM becomes the reasoning engine; embeddings become the retrieval engine.

```
Without RAG:
  User: "What is our vacation policy for remote employees?"
  LLM: answers from training data → may hallucinate, may be wrong, may be stale

With RAG:
  User: "What is our vacation policy for remote employees?"
  Retrieve: top-k chunks from your HR policy documents
  LLM: reads retrieved docs → generates accurate, grounded answer from YOUR documents
```

---

### The Core Flow

```
OFFLINE (pre-computed, runs once or on document update):

  Large document corpus
  (HR policies, product docs, LinkedIn posts, job descriptions, ...)
          ↓
  Chunking — split into retrievable pieces
          ↓
  Embed each chunk → store in vector store (ANN index)

─────────────────────────────────────────────────────

ONLINE (query time):

  User question
          ↓
  Embed the question (same encoder as documents)
          ↓
  ANN search → top-k most similar document chunks
          ↓
  Stuff into LLM prompt:
    "Here are relevant documents:
     [chunk 1: Remote Work Policy — remote employees accrue 15 days PTO...]
     [chunk 2: Benefits Overview — PTO applies equally to all employee types...]
     Answer this question: What is our vacation policy for remote employees?"
          ↓
  LLM reads retrieved context → generates accurate, grounded answer
```

---

### Why Retrieval Quality Is Everything

The LLM can only reason about what is in its context window. If the wrong chunks come
back from retrieval, the LLM has nothing to work with — it will hallucinate or say
"I don't know." No prompt engineering fixes bad retrieval.

```
User asks: "Can I withdraw from my 401k early?"

Bad embeddings:
  retrieve docs about "annual contribution limits"
  → LLM has no relevant context → hallucinates penalties or gives wrong amount

Good embeddings:
  understand "401k withdrawal" ≈ "retirement account distribution" ≈ "early withdrawal penalty"
  → retrieve correct IRS penalty docs and rules
  → LLM answers accurately with the 10% penalty and exceptions
```

The retrieval quality of a RAG system is almost entirely a function of how good your
embeddings are. If your embeddings don't understand domain-specific term equivalences,
no downstream fix will compensate.

---

### LinkedIn RAG Example

A LinkedIn AI assistant answering career questions:

```
User: "What skills should I learn to become an ML engineer?"

OFFLINE:
  Embed thousands of:
    - ML engineer job postings (required skills, preferred qualifications)
    - Career progression posts by ML engineers
    - Skill endorsement patterns across LinkedIn members
    - Course completion data from LinkedIn Learning

ONLINE:
  Embed question → retrieve: top job postings for ML roles,
                              popular skill paths in ML community,
                              posts by ML engineers about their learning journey
          ↓
  LLM receives real, current data from LinkedIn's corpus
          ↓
  Answer: "Based on current ML engineer job postings: Python, PyTorch, statistics,
           and system design are most frequently required. Members who transitioned
           to ML roles commonly completed courses in linear algebra and deep learning first."
```

Without RAG: LLM answers from training data — potentially 2 years stale, not grounded
in LinkedIn's actual job market data.

With RAG: LLM answers from live job postings and real member career data — specific,
current, and grounded.

---

### RAG Components in Depth

**1. Chunking — how you split documents**

```
Too small (sentence level):
  Chunk: "Remote employees receive PTO."
  Missing context: how much? under what conditions? → LLM can't answer fully

Too large (full document):
  Fills context window with irrelevant content
  → LLM attends to noise, quality degrades

Sweet spot (paragraph level, 100–300 tokens):
  Chunk: "Remote employees accrue 15 days of PTO annually, same as on-site employees.
          PTO accrual begins after 90-day probationary period. Unused PTO rolls over
          up to a maximum of 30 days."
  → Self-contained, answerable, not bloated
```

Sliding window chunking (overlapping chunks) handles the case where an answer spans
a chunk boundary — the overlap ensures no sentence is only in one chunk.

**2. Embedding — same encoder for documents and queries**

The document chunks and the user query must be embedded with the **same encoder**
into the same vector space. If you embed documents with BERT and queries with a
different model, cosine similarity is meaningless — different vector spaces.

Fine-tuning the encoder on domain-specific data (HR queries, LinkedIn career questions)
dramatically improves retrieval quality for domain-specific terminology.

**3. Vector Store — ANN index for fast similarity search**

```
Options: FAISS (open source, Meta), Pinecone, Weaviate, Qdrant, pgvector (Postgres)

At query time:
  embed query → ANN search (HNSW or IVF) → top-k nearest chunks
  millisecond retrieval over millions of document chunks
```

**4. Reranking — cross-encoder over top-k before stuffing into prompt**

Bi-encoder retrieves top-k fast but lacks query-document cross-attention.
Cross-encoder reranks the top-k with full attention before they enter the prompt:

```
Bi-encoder retrieves top-20 chunks  (~2ms, coarse)
        ↓
Cross-encoder reranks 20 → top-5    (~10ms, precise)
        ↓
Top-5 chunks stuffed into LLM prompt
```

Same bi-encoder → cross-encoder pattern as the LinkedIn Feed pipeline.
The reranking step meaningfully improves answer quality for ambiguous queries.

**5. Generation — LLM with retrieved context**

```
Prompt structure:
  [System]    You are a helpful assistant. Answer based only on the provided documents.
  [Context]   Document 1: ... Document 2: ... Document 3: ...
  [Question]  User's question
  [Instruction] If the answer is not in the documents, say "I don't have that information."
```

The "answer only from documents" instruction prevents the LLM from falling back to
its training knowledge and hallucinating — keeps answers grounded in retrieved content.

---

### Hybrid Search — Semantic + Keyword

Pure semantic search misses exact term matches. Pure keyword search misses semantic
meaning. Production RAG systems combine both:

```
Query: "FAANG SDE interview preparation"

Semantic search:
  finds: "big tech software engineering interviews", "coding interview tips"
  understands the concept even without the exact term "FAANG"

Keyword search (BM25):
  finds: documents containing literally "FAANG" or "SDE"
  catches exact terminology, acronyms, proper nouns

Hybrid score = α × semantic_score + (1-α) × BM25_score
```

| Signal type | Strength | Weakness |
|---|---|---|
| Semantic (embeddings) | Understands meaning, synonyms, concepts | May generalize away exact terms |
| Keyword (BM25) | Catches exact terms, acronyms, names | No semantic understanding |
| Hybrid | Gets both | Needs tuning of α weight |

Semantic embeddings handle meaning. Keywords handle exact terminology, product names,
acronyms, and proper nouns that embeddings might map to broader concepts.

---

### RAG Failure Modes

| Failure | Cause | Fix |
|---|---|---|
| Wrong documents retrieved | Weak embeddings, domain mismatch | Fine-tune encoder on domain data |
| Answer spans chunk boundary | Chunking too small | Sliding window / larger chunks |
| Context window overwhelmed | Too many large chunks retrieved | Reranker to reduce top-k before prompt |
| Stale answers | Documents not refreshed | CDC pipeline to re-embed on document update |
| LLM ignores retrieved docs | Prompt structure weak | Explicit "answer only from documents" instruction |
| Exact term missed | Pure semantic search | Add BM25 hybrid search |

---

## 7. Embedding Dimensionality

### Dimension Choices and What They Mean

Most production embeddings sit in the 64–1024 range:

| Dimension | Use case | Why |
|---|---|---|
| 64–128 | Huge-scale recommendation retrieval | Max QPS, fits in cache, minimal memory bandwidth |
| 256–512 | Most production tasks | Good balance of quality and cost |
| 1024+ | LLM embeddings, CLIP, high-stakes ranking | Quality matters more than serving cost |

Higher dimensions capture more nuance — more degrees of freedom to encode subtle
distinctions like seniority level, content depth, professional community. But they
cost more to train, store, and search. The QPS impact is the most concrete reason
why dimension choice matters in a system design interview.

---

### Why Dimension Directly Impacts QPS — Memory Bandwidth

At query time, scoring candidates requires reading their embedding vectors from memory
into the GPU/CPU to compute the dot product. That reading consumes **memory bandwidth**
— the pipe between RAM and compute. That pipe has a fixed capacity.

```
Per request: score 500 candidate post embeddings

Memory read per request:
  d=128:  500 × 128 × 4 bytes (FP32) =  256 KB
  d=512:  500 × 512 × 4 bytes (FP32) = 1,024 KB  (4× more)

At LinkedIn peak (~17,000 QPS):
  d=128:  17,000 × 256 KB  =  4.3 GB/s  → well within server bandwidth
  d=512:  17,000 × 1024 KB = 17.4 GB/s  → needs 4× more servers to sustain
```

At ANN index scale (millions of candidates), the index size compounds the problem:

```
1B posts in index:
  d=128: 1B × 128 × 4 bytes =  512 GB
  d=512: 1B × 512 × 4 bytes = 2,048 GB

Larger index → doesn't fit in CPU cache → cache miss on every lookup
Cache hit:  ~1ns
Cache miss: ~100ns  (100× slower)
Higher d = more cache misses = catastrophic QPS drop at scale
```

This is why small dimensions are used for high-QPS retrieval — not because quality
doesn't matter, but because the index must fit in cache to serve at production QPS.

---

### Matryoshka Embeddings — Nested Precision

Matryoshka embeddings solve the quality-vs-QPS tradeoff by training a single embedding
where the first 64 dims, first 128, first 256 are all valid embeddings on their own.

```
Full embedding: [d1 ... d64 | d65 ... d128 | d129 ... d512]
                 └─ valid 64d ─┘  └── valid 128d ──┘  └─ valid 512d ─┘
```

Named after Russian nesting dolls — smaller embeddings are nested inside the larger one.

**Training:** the loss is computed at multiple prefix lengths simultaneously:
```
loss = InfoNCE(first 64 dims) + InfoNCE(first 128 dims) + InfoNCE(full 512 dims)
```

This forces the most important semantic signal into the earliest dimensions.

**Production use — two-phase retrieval:**
```
Phase 1 — coarse retrieval (high QPS, small index):
  Use only first 64 dims → index is 8× smaller → fits in cache → max QPS
  Retrieve top-500 candidates

Phase 2 — reranking (low QPS, full precision):
  Use full 512 dims on just 500 candidates → high quality, affordable
  Return top-20
```

Same embedding stored once. Phase 1 reads only the first 64 floats per item.
No second index to maintain, no separate model to deploy.

---

### 1-Bit Sign Quantization — LinkedIn LiNR

When even reduced dimensions aren't enough (billions of items), quantize embeddings
to 1 bit per dimension by keeping only the sign.

**The geometric intuition:**

Two vectors pointing in roughly the same direction in d-dimensional space will have
the same sign on most of their dimensions. XOR gives the count of dimensions where
signs differ — fewer differences = more aligned = more similar.

```
u = [+0.9, +0.1, -0.3, +0.6]  →  1-bit: [1, 1, 0, 1]
v = [+0.8, +0.2, -0.1, +0.4]  →  1-bit: [1, 1, 0, 1]
XOR:                                      [0, 0, 0, 0]  → 0 differing bits → similar ✓

w = [-0.7, +0.3, +0.5, -0.2]  →  1-bit: [0, 1, 1, 0]
XOR with u:                               [1, 0, 1, 1]  → 3 differing bits → dissimilar ✓
```

**The precise relationship (Random Hyperplane LSH):**
```
P(sign(u_i) = sign(v_i)) = 1 - arccos(cosine_similarity(u, v)) / π

cosine_similarity = 1.0  →  100% bits agree  (same direction)
cosine_similarity = 0.0  →   50% bits agree  (perpendicular)
cosine_similarity = -1.0 →    0% bits agree  (opposite direction)
```

Fraction of matching bits is a monotonic proxy for cosine similarity.
Not exact — near-zero dimensions flip signs unreliably — but with 128 bits the
aggregate is a reliable approximation.

**Memory reduction:**
```
1B items × 64-dim × FP16 = 120 GB  → doesn't fit in A100 (80 GB HBM)
1B items × 64-dim × 1-bit =  7.5 GB → fits easily on single V100 GPU

XOR + POPCNT is a single GPU hardware instruction → much faster than FP16 dot product
```

**LiNR's three-step exhaustive search pipeline:**
```
Step 1 — ABM attribute pre-filter:
  Apply hard constraints (company, location, skills) → 240M → ~5M eligible items

Step 2 — 1-bit XOR scan:
  Compute approximate similarity via sign matching for all 5M items
  Keep top 1% (~50K items)
  Memory read: 5M × 16 bytes (128-bit packed) = 80 MB  ← fast

Step 3 — Full FP16 dot product:
  Score only the 50K survivors → top-K returned
  Memory read: 50K × 256 bytes (128-dim fp16) = 12 MB  ← tiny

vs naive exhaustive FP16: 240M × 256 bytes = 61 GB read → 400× more memory
```

**LiNR production results:**
```
Index size: 240M embeddings, 128-dim, fp16
Latency:    4.57ms avg, 4.79ms p95  (batch=1)
Live model updates: zero measurable latency impact

A/B test on LinkedIn Feed OON recommendations:
  +7%  total professional interactions
  +5%  feed viewers with 30+ secs dwell time
  -20% skipped update rate  ← users scrolling past content dropped 20%
  +3%  professional daily active users
```

The -20% skipped update rate is the most telling signal: exhaustive search with
pre-filtering retrieves items users actually read, not just embedding-close items
that fail attribute constraints and slip through post-filtering gaps.

**Why exhaustive beats ANN for filtered retrieval (the liquidity problem):**
```
ANN with post-filtering:
  Retrieve top-50K by embedding similarity
  Apply attribute filter → only 2% pass → 1K results
  The 49K retrieved items were wasted compute
  AND: the true best matching items outside top-50K are never seen → recall loss

LiNR exhaustive with pre-filtering:
  Filter first → score only eligible items → no wasted compute, no recall loss
  No liquidity problem by construction
```
