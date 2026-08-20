# LLMs in Recommendation Systems — Retrieval & Ranking

---

## 1. Why LLM Encoders Are Richer Than Traditional Encoders

### Traditional Encoder (Two-Tower / Feedforward)

```
[post_1_embedding]  ─┐
[post_2_embedding]  ─┤ → average / concatenate → user_embedding
[post_3_embedding]  ─┘
```

- Every past interaction contributes **equally and independently**
- No sense of order, recency, or relationship between interactions
- Trained from scratch on click/purchase data only
- Can only connect items users have **co-clicked** — if no user clicked both, the model never links them

### LLM Encoder (Transformer with Attention)

```
[post_1_embedding]  ─┐
[post_2_embedding]  ─┤ → attention (each token attends to all others) → user_embedding
[post_3_embedding]  ─┘
```

Two things make it richer:

#### 1. Attention Mechanism
Interactions attend to each other → the final embedding captures **relationships between past behaviors**, not just a bag of signals.

- The same post means something different depending on what surrounds it in your history
- Recency and context-dependence are learned, not hand-engineered

#### 2. Pre-Trained World Knowledge
An LLM has seen the internet. It already understands semantic and domain relationships **before seeing a single user click**.

```
Two-tower (trained from scratch):
  "power grid" ↔ "small modular reactors" → only connected if users co-clicked them

LLM encoder (pre-trained):
  "power grid" ↔ "small modular reactors" → already connected via world knowledge
```

A two-tower must learn these relationships from co-engagement data. If the data is sparse, it never learns them. The LLM already knows.

> **Summary:** Attention captures behavioral context. Pre-training provides semantic knowledge. Both together make the embedding dramatically richer.

---

## 2. The Retrieval Flow

### Goal
From 10M+ posts, fast-retrieve ~500 candidates that are semantically relevant to the user.

### Architecture: Dual Encoder (Two-Tower with LLM Backbone)

```
                    ┌──────────────────────────────────┐
                    │         OFFLINE (batch)           │
                    │                                   │
  Every post ──→ [LLM Item Encoder] ──→ post_embedding │
                    │                  stored in        │
                    │                  ANN index        │
                    └──────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │      ONLINE (per request)         │
                    │                                   │
  User X ──→ [LLM User Encoder] ──→ user_embedding     │
                    │                    ↓              │
                    │         ANN search over           │
                    │         post_embeddings           │
                    │                    ↓              │
                    │         top ~500 candidates       │
                    └──────────────────────────────────┘
```

### What goes into each encoder

**Item Encoder input:**
- Post text / title / content
- Author metadata
- Engagement signals (converted to percentile buckets — see below)
- Category, tags

**User Encoder input:**
- User's past interactions (liked, shared, commented posts)
- Profile metadata
- Recent session context

### Key Feature Engineering Detail — Percentile Buckets

Raw numbers tokenize badly in LLMs:

```
Bad:   view_count = 1,400,000          ← tokenizes as arbitrary digits
Good:  <view_percentile>71</view_percentile>  ← meaningful relative signal
```

LinkedIn found this alone improved Recall@10 by **15%** and popularity correlation by **30x**.

### Matching

```
score(user X, post P) = dot_product(user_embedding, post_embedding)
```

ANN (Approximate Nearest Neighbor) search — e.g. FAISS, ScaNN — finds top 500 without scoring all 10M.

### Training: Hard Negative Mining

| Negative type | What it is | Effect |
|---|---|---|
| Easy negatives | Randomly sampled unshown posts | Stable training signal |
| Hard negatives | Posts shown but not engaged with | Teaches nuanced distinctions |

Adding just **2 hard negatives per training example** yielded a 3.6% recall improvement (LinkedIn).

Also: filtering user history to **only positive engagements** (removing scrolled-past content) improved quality AND reduced GPU memory by **37%**.

---

## 3. The Ranking Flow

### Goal
Given ~500 retrieved candidates, score and sort them to find the best top 10 for this user **right now**.

### The Problem with Traditional Ranking

Traditional ranker builds a flat feature vector per user-item pair:

```
[user_avg_clicks, user_country, item_category, item_popularity,
 user_item_similarity, time_since_last_purchase, ...]
        ↓
  [GBDT or feedforward NN]
        ↓
  score → sort
```

- Each candidate scored **independently** — no cross-candidate awareness
- User represented as **average behavior snapshot** — no trajectory
- Recency is a hand-crafted feature (time_decay), not learned

### LLM-Based Sequential Ranking

```
User's last 1000 interactions (ordered):
  [post_A (liked, 3d ago), post_B (shared, 2d ago), post_C (commented, 1d ago), ...]
        ↓
  [Transformer — causal attention]
        ↓
  user_context_vector   ← captures trajectory, not just average
        ↓  (late fusion)
  f(user_context_vector, candidate_features)
        ↓
  score per candidate → top 10
```

### Why Sequence Changes Everything

```
User history (ordered):
  Month 1: Python basics
  Month 2: ML fundamentals
  Month 3: Model training
  Month 4: ML infra / serving
  This week: Distributed systems

Traditional ranker: user likes Python + ML + infra → scores all three equally
Sequential ranker:  user is on a trajectory → predicts they want senior infra /
                    architecture content next, NOT Python basics
```

### Late Fusion — Why It's Necessary

Running transformer over 1000 history items × 500 candidates naively is too expensive:

```
Naive (too slow):
  for each candidate:
    transformer(user_history + candidate) → score   ← 500 transformer passes

Late fusion (LinkedIn's approach):
  transformer(user_history) → user_context_vector   ← 1 transformer pass
  score = f(user_context_vector, candidate_features) ← 500 cheap dot products
```

Candidate features (post category, engagement, recency) are injected **after** the transformer.

### Advanced: Multi-gate Mixture-of-Experts (MoE) Head

Different engagement types need different scoring heads:

```
user_context_vector
        ↓
  [Gate] ──→ selects expert head per task
        ├──→ [Click head]    → P(click)
        ├──→ [Share head]    → P(share)
        └──→ [Comment head]  → P(comment)
```

Each gate learns which features matter for each engagement type.

### Traditional vs LLM Ranking — Comparison

| | Traditional Ranking | LLM Sequential Ranking |
|---|---|---|
| User representation | Flat feature vector (avg behavior) | Ordered sequence of 1000+ interactions |
| Understands trajectory | No | Yes |
| Recency sensitivity | Manual feature | Learned via attention |
| Cross-interaction patterns | No | Yes |
| New user benefit | Poor (sparse features) | Better (extracts more signal per interaction) |
| Cost | Low | High (mitigated by late fusion) |

---

## 4. Complete End-to-End Request Flow

```
User X opens feed
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL STAGE                                            │
│                                                             │
│  User X's history + profile                                 │
│        ↓                                                    │
│  [LLM User Encoder]  →  user_embedding (128-dim)           │
│        ↓                                                    │
│  ANN search over pre-computed post_embeddings               │
│  (FAISS / ScaNN index, updated nightly)                     │
│        ↓                                                    │
│  ~500 candidate posts                                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  RANKING STAGE                                              │
│                                                             │
│  User's last 1000 interactions (ordered sequence)           │
│        ↓                                                    │
│  [Transformer — causal attention]                           │
│        ↓                                                    │
│  user_context_vector                                        │
│        ↓  late fusion                                       │
│  + candidate features (category, engagement, recency)       │
│        ↓                                                    │
│  [MoE head] → score per candidate                          │
│        ↓                                                    │
│  top 10 posts, sorted by score                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
  Top 10 shown to User X
```

### What each stage cares about

| | Retrieval | Ranking |
|---|---|---|
| Question | Which posts are semantically relevant? | Which of these 500 fits the user *right now*? |
| Dominant signal | Item content + user interest embedding | User trajectory + candidate features |
| Scale | 10M → 500 | 500 → 10 |
| Latency budget | Must be fast (ANN search) | Slower OK (transformer + late fusion) |
| Item encoding | Deep (LLM encodes full post content) | Shallow (candidate features injected late) |
| User encoding | Embedding of interest profile | Sequential transformer over full history |

### Offline vs Online split

```
OFFLINE (batch, runs nightly):
  - Encode all posts → post_embeddings → build ANN index
  - Pre-compute user embeddings for known users → store in feature store
  - Retrain encoders and ranking model on new engagement data

ONLINE (per request, <100ms budget):
  - Fetch user embedding (from feature store or recompute if stale)
  - ANN search → 500 candidates
  - Run sequential ranker → score 500 → top 10
  - Return feed
```

---

## 5. Cold Start Problem — and How LLM Solves It

### What is Cold Start?

When a new user joins, there is no engagement history — no clicks, likes, shares. Traditional retrieval depends on a user embedding built from that history. No history → weak embedding → bad recommendations on day one.

```
New user: "Senior Power Systems Engineer at GE"
  Engagement history: NONE

Traditional system: no signal → fall back to popularity-based / generic content
LLM system:         job title alone → rich, domain-aware user embedding
```

### What Traditional (Non-LLM) Encoders Can Do

Non-LLM embeddings capture **shallow lexical similarity** — words that appear together frequently:

```
"electrical engineering" → similar to: power, energy, electronics, circuits
```

Useful, but surface-level:

```
"electrical engineer" → power grids?           ✅ maybe
"electrical engineer" → renewable energy?      ⚠️  weak
"electrical engineer" → small modular reactors? ❌ unlikely
```

The model only knows surface co-occurrence. It hasn't seen enough signal to connect these as deeply related domains.

### What LLM Encoders Can Do

An LLM brings **world knowledge from pre-training** — it has read papers, articles, forums. It already knows:

```
"Senior Power Systems Engineer"
        ↓  [LLM world knowledge]
  - works on power grid optimization
  - involved in renewable energy integration
  - infrastructure challenges overlap with nuclear (SMRs)
  - power systems = energy systems = grid modernization
```

So from just a job title, it constructs a rich user embedding — **without a single click**:

```
user_embedding close to:
  power grid posts        ✅
  renewable energy posts  ✅
  small modular reactors  ✅
  energy infrastructure   ✅
```

### Side by Side — Cold Start

| | Traditional Retrieval | LLM Retrieval |
|---|---|---|
| Input available | Job title only | Job title only |
| User embedding quality | Weak — surface keywords | Rich — domain concepts |
| Retrieved content | Generic engineering posts | Career-relevant, domain-specific |
| Time to good recs | Weeks (needs engagement history) | Day one |

### Beyond Cold Start — Vocabulary Mismatch

LLM world knowledge also helps **existing users** when terminology diverges:

```
User engages with:  "small modular reactors"
Post available:     "advanced nuclear power infrastructure"

Traditional:  different words → low similarity → not retrieved ❌
LLM:          same domain    → high similarity → retrieved     ✅
```

Click data alone can never fully resolve vocabulary gaps across a domain. The LLM bridges them.

### One-line summary

> Traditional encoders know what words appear together. LLM encoders know what **concepts** belong together — and that's the difference between a new user getting a generic feed vs. a relevant one on day one.

---

## 6. Tokenization in LLMs — BPE and Why Raw Numbers Fail

### What is a Token?

LLMs don't read characters or words — they read **tokens**. A token is a chunk of text, typically a subword unit. The tokenizer converts raw text into a sequence of integer IDs that the model processes.

```
"The electrical engineer works on power grids"
        ↓  tokenizer
["The", "electrical", "engineer", "works", "on", "power", "grids"]
  101     2345          891         234      56    789       1023    ← token IDs
```

### How BPE (Byte Pair Encoding) Works

BPE is the most common tokenization algorithm. It learns a vocabulary by repeatedly merging the most frequent character pairs in the training corpus.

```
Training corpus contains millions of documents.
BPE starts with characters, then merges frequent pairs:

  "e" + "n" → "en"         (frequent)
  "en" + "g" → "eng"       (frequent)
  "eng" + "ine" → "engine" (very frequent)
```

The result: common words and subwords become single tokens. Rare or unseen sequences get split into smaller chunks.

```
"engineer"  → ["engineer"]        ← 1 token (seen often in training)
"xqzplt"    → ["x", "q", "z", "p", "l", "t"]  ← 6 tokens (never seen)
```

### Why Raw Numbers Tokenize Badly

The tokenizer was trained on natural language — not numeric data. It has no concept of magnitude or order. Numbers are just character sequences to it.

```
"views: 12345"
        ↓  BPE tokenizer
["views", ":", "12", "345"]   ← arbitrary split based on character frequency
```

The split depends entirely on what digit sequences appeared frequently in pre-training. There is no guarantee consistency:

```
12345  →  ["12", "345"]
12346  →  ["12", "346"]
98000  →  ["98", "000"]
```

**The problem:** "345" and "346" are two completely unrelated tokens in the model's vocabulary. The model has no way to learn that these numbers are nearly identical, or that 98,000 is much larger than 12,345.

```
Human reads "12345" vs "98000":  instantly knows 98000 is ~8x larger
LLM reads  "12345" vs "98000":  sees two unrelated token sequences — no magnitude
```

This is why LinkedIn saw **correlation = -0.004** (essentially noise) between raw view counts and embedding similarity. The model simply could not use the number as a signal.

### Why Small Integers Tokenize Well

Numbers 1–100 appear constantly in natural language ("page 71", "95% complete", "12 items"). BPE preserves these as single tokens:

```
"71"  →  single token  ✅  stable, meaningful embedding
"95"  →  single token  ✅  stable, meaningful embedding
"12"  →  single token  ✅  stable, meaningful embedding
```

The model has seen these thousands of times in context — it already has rich embeddings for them.

### The Percentile Bucket Fix

```
Raw:     views = 12,345   →  "views:12345"  →  ["views", ":", "12", "345"]
                                                  arbitrary split, no ordinal meaning

Better:  views = 12,345   →  71st percentile  →  <view_percentile>71</view_percentile>
                                                  single token "71", stable embedding
```

The model can now learn stable representations for quantity ranges:

```
0–20   →  low popularity
40–60  →  average
80–90  →  high popularity
90+    →  viral
```

**Result at LinkedIn:** correlation jumped 30x, Recall@10 improved 15%.

---

## 7. Passing Structured Data to LLMs — Prompts, Templates, and Special Tokens

### Why Prompts Are Necessary

An LLM encoder expects **text as input**. It was pre-trained on text — articles, papers, web pages. It has no native concept of a feature vector or a database row.

So structured data must be **serialized into text** before the LLM can process it:

```
Structured data:
  job_title: "Senior Power Systems Engineer"
  skills: ["Python", "Power Systems", "SCADA"]
  view_count: 1,400,000

        ↓  serialize to text (prompt)

"Senior Power Systems Engineer. Skills: Python, Power Systems, SCADA.
 Views: <view_percentile>71</view_percentile>"

        ↓  LLM encoder

  embedding vector
```

This is not optional — it is a consequence of what LLMs are: text-in, embedding-out machines.

### Why Templates Matter

You could naively dump data as key:value strings, but inconsistent formatting confuses the model:

```
Post A:  "views=12345 author=John title=Power Grids"
Post B:  "John wrote about Power Grids. 12345 views."
Post C:  "title: Power Grids | author: John | views: 12345"
```

Three posts, three formats — the model sees different patterns and cannot generalize.

A **prompt template** enforces consistency across all items:

```
Post template:
  "Author: {name}, {headline} at {company}.
   Post: {post_text}
   Engagement: <view_percentile>{v}</view_percentile> <like_percentile>{l}</like_percentile>"

Member template:
  "Member: {name}, {headline} at {company}.
   Skills: {skills}. Education: {education}.
   Recent engagement: {chronological_post_history}"
```

Every post goes through the same template → the model always sees the same structure → it learns stable patterns.

### Why Special Tokens — Separating Signal from Prose

Without special tokens, a number inside a sentence is ambiguous:

```
"This post got 71 views"   ← is 71 a view count? a page? an age? a year?
```

Special tokens create a **namespace** — they explicitly label what the value represents:

```
"This post got <view_percentile>71</view_percentile>"
                ↑                                  ↑
          opens scope                         closes scope
          "what follows is a view percentile"
```

Now "71" is unambiguously a view percentile, not arbitrary text.

### How the Model Learns Special Token Meaning

Special tokens have **no inherent meaning out of the box**. They are just new entries added to the tokenizer vocabulary with fresh token IDs.

Their meaning is learned through **fine-tuning**:

```
Step 1: Add special tokens to tokenizer vocabulary
        <view_percentile>, </view_percentile>  →  assigned new token IDs (e.g. 50000, 50001)

Step 2: Fine-tune the LLM on your dataset
        Model repeatedly sees:
          <view_percentile>91</view_percentile>  →  high engagement post (positive signal)
          <view_percentile>12</view_percentile>  →  low engagement post (weak signal)

Step 3: Through gradient descent, model learns:
          <view_percentile> signals "what follows is a popularity bucket"
          91 inside it → "viral"
          12 inside it → "low reach"
```

There is no external config or dictionary. The meaning is baked into the model weights during fine-tuning — exactly the same way the model learned that `def` means a Python function or `<b>` means bold HTML.

```
What people assume:
  LLM + config {"<view_percentile>" = "popularity rank"} → output

What actually happens:
  Fine-tuned LLM weights (meaning learned during training) → output
```

### The Full Structured Data → Embedding Chain

```
Raw structured data (DB row, feature store)
        ↓
Prompt template serializes to consistent text
        ↓
Special tokens wrap all numerical features as percentile buckets
        ↓
Tokenizer converts text → token ID sequence
  (special tokens get their own stable IDs)
        ↓
Fine-tuned LLM processes token sequence
  (special token semantics learned during fine-tuning)
        ↓
Rich embedding vector
```

### Summary Table

| Concept | What it is | Why it matters |
|---|---|---|
| Prompt | Text serialization of structured data | LLMs only accept text input |
| Template | Consistent format for all items | Model generalizes across all posts/users |
| Special token | Reserved vocabulary entry wrapping a value | Namespaces the signal, separates from prose |
| Token meaning | Learned during fine-tuning via gradient descent | No config — meaning lives in model weights |
| Percentile bucket | Collapse raw number → 1-100 | Single stable token, model can learn ordinal meaning |

---

## Key Interview Takeaways

| Point | What to say |
|---|---|
| Why LLM in retrieval | Richer embeddings — attention captures behavioral context, pre-training gives semantic knowledge |
| Why LLM in ranking | Sequential modeling — user history as trajectory, not average snapshot |
| Feature engineering trick | Convert raw counts to percentile buckets with special tokens for LLM input |
| Scalability trick | Late fusion — run transformer once on user history, score candidates cheaply |
| Hard negatives | Shown-but-not-clicked posts as negatives teach nuanced distinctions (3.6% recall gain at LinkedIn) |
| Both stages use LLM | Retrieval for semantic embedding, Ranking for sequential user modeling |
| Cold start | LLM infers interests from profile alone (job title, headline) — no engagement history needed |
| Vocabulary mismatch | LLM connects "small modular reactors" ↔ "nuclear power infrastructure" — click data can't |
| Why raw numbers fail | BPE splits "12345" arbitrarily → no ordinal meaning → model can't use magnitude as signal |
| Special token meaning | Not from config — learned via fine-tuning, same as any other token |
