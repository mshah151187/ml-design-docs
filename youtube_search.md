# YouTube Search System

Systems: YouTube Search, YouTube Shorts search, YouTube Kids search
Technical foundation: semantic_search.md (bi-encoder retrieval, SLM cross-encoder ranking,
  distillation, inference optimizations)

---

## 1. Problem Framing

### Business Objective

A user types a query. The system shows a ranked list of videos. The business goal is not
to "show relevant videos" — that is the ML framing, not the business framing. The
business goal is:

**Maximize quality-adjusted watch time initiated through search.**

The chain from query to revenue:

```
User enters query
      ↓
System returns results
      ↓
User clicks the right video (correct result surfaced at top)
      ↓
User watches willingly — not misled by clickbait, not disappointed by irrelevant content
      ↓
User completes or returns for more
      ↓
Ad impressions accumulate during session
      ↓
User returns to YouTube tomorrow (retention)

At 2B MAU: every +1% improvement in query satisfaction
  = ~20M users finding what they wanted
  = proportional ad revenue + next-session return rate gain
```

**Why "satisfied user" is not "clicked video":**

```
Click-optimized search → surface clickbait thumbnails, misleading titles
  → user clicks, watches 5 seconds, leaves
  → click revenue in A/B looks good
  → long-run: trust erodes, user stops using search, opens TikTok instead

Quality-adjusted watch time → surface the video that actually answers the query
  → user watches 8 minutes, feels the search was worth it
  → returns next session, uses search again
  → long-run: search becomes a habit, session depth grows
```

**Business objective: quality-adjusted watch time initiated through search.**

```
quality_adjusted_watch_time = watch_time × quality_score

Quality signals for search context:
  Watch fraction (>70% completion = user found what they wanted)
  Skip rate (<5 seconds before skip = mismatch between query intent and video)
  "Not interested" signals (explicit rejection)
  Return to search results immediately (strong signal: result was wrong)
  Like / save / share from search result (strong endorsement)
```

**Guardrails — two enforcement points:**
```
At ingestion (never indexed):
  Harmful / misinformation / hate speech  ← removed from platform entirely
  Copyright violations (Content ID)       ← removed from platform entirely
  Spam / artificially promoted videos     ← removed from platform entirely

At retrieval pre-filter (indexed, filtered per query):
  Age-restricted content                  ← filtered based on user's safe search / age
  Geo-restricted content                  ← filtered based on user's region vs
                                             video's available_regions license list
```

---

### Scale

```
YouTube:
  2B MAU
  500 hours of video uploaded per minute
  Billions of queries per day across languages and devices
  p99 search result latency < 200ms
  Corpus: hundreds of millions to billions of indexed videos
  New videos: need to be indexed and searchable within minutes of upload
```

---

### Business Objective → ML Objective Mapping (Top Level)

| Layer | Business why | ML objective |
|---|---|---|
| Query Understanding | Ambiguous queries produce irrelevant results — user gives up | Intent classification, entity recognition, normalized query representation |
| Retrieval | Cannot rank a video that was never retrieved — recall determines ceiling | Recall@K: all relevant videos must appear in top K |
| Ranking | Of retrieved candidates, first page determines whether user watches | Precision@10, NDCG@10: best match at top |
| Re-ranking | Stale, repetitive, or creator-monopolized results hurt long-term engagement | Constrained optimization: quality × freshness × diversity |
| Personalization | Same query means different things for different users | Lift in query satisfaction NDCG given user history features |
| Cold Start | New videos (no engagement history) must still be retrievable | Proxy relevance signals: title, description, transcript quality |

---

## 2. System at a Glance

```
User query (text)
      ↓
┌──────────────────────────────────────────┐
│  Query Understanding Layer               │
│  Intent classification, entity NER,      │
│  spell correction, query expansion       │
└──────────────────────────────────────────┘
      ↓ structured query + signals
┌──────────────────────────────────────────┐
│  Stage 1: Retrieval                      │
│  Bi-encoder embedding search             │
│  + BM25 inverted index (hybrid)          │
│  → top-K=1000 candidate video IDs        │
└──────────────────────────────────────────┘
      ↓ 1000 candidates
┌──────────────────────────────────────────┐
│  Stage 2: Ranking                        │
│  SLM cross-encoder                       │
│  jointly scores relevance + engagement   │
│  → top-250 re-ranked                     │
└──────────────────────────────────────────┘
      ↓
┌──────────────────────────────────────────┐
│  Stage 3: Re-ranking + Business Rules    │
│  Freshness, diversity, policy enforcement│
│  → top-20 final results                  │
└──────────────────────────────────────────┘
      ↓
Final ranked video list shown to user
```

**Key scale numbers:**

| Dimension | Number |
|---|---|
| Corpus size | Hundreds of millions of indexed videos |
| Retrieval output | Top-1,000 per query |
| Ranking input | Top-250 (SLM ranks top-250 from retrieval) |
| Final results shown | Top-20 (first page) |
| End-to-end latency budget | p99 < 200ms |
| Query languages | 100+ languages |

---

## 3. Query Understanding Layer

### Objective Chain

```
Business goal: a user's typed query rarely expresses their full intent cleanly
  → "python tutorial" could mean: beginner vs advanced, scripting vs ML, Python 2 vs 3
  → surfacing wrong-intent results = user gives up on search

↓ System must do: parse the raw query into a structured, normalized representation
  that retrieval and ranking can consume reliably

↓ ML objective: minimize intent mismatch rate
  (fraction of sessions where user returns to search results and reformulates the query)
```

### What the QU Layer Does

```
Raw query: "how to make pasta carbonara easy"

QU Layer output:
{
  intent:       "how-to / tutorial",
  topic:        "pasta carbonara",
  modifier:     "easy / beginner",
  entity_type:  "recipe",
  language:     "en",
  safe_search:  "standard",
  reformulations: [
    "easy carbonara recipe",
    "carbonara pasta tutorial for beginners"
  ]
}
```

**Key operations:**

| Operation | Why it matters |
|---|---|
| Intent classification | Tutorial vs review vs music video vs news clip → routing to specialized retrievers |
| Entity recognition | Named entities (person names, brand names, song titles) get exact-match treatment in retrieval |
| Spell correction | Typos fragment inverted index lookups; correction runs before retrieval |
| Query expansion | Synonyms, related phrases added as soft signals to retrieval |
| Language detection | Routes to language-appropriate corpus shard and model |

**Why QU is a hard dependency for retrieval:**
Attribute-Based Matching (ABM) in retrieval needs structured signals — it cannot filter on raw
text. QU extracts intent and entity types that become hard pre-filter masks (e.g., for a music
query: `content_type=music_video` pre-filters non-music content before embedding scan, same
logic as LiNR's ABM from semantic_search.md).

---

## 4. Video Content Representation (Nearline / Offline)

Before retrieval can work, every video in the corpus needs a rich content representation.
This is fundamentally harder than LinkedIn's text documents (job descriptions / profiles) —
a video is multi-modal.

### Objective Chain

```
Business goal: video quality should be judged by what is in the video, not just what the
  uploader wrote in the title and tags. Titles and tags are gamed; content is not.

↓ System must do: extract a rich semantic representation from each video's actual content

↓ ML objective: document embeddings (for retrieval) that reflect actual video content,
  not just uploader-provided metadata
```

### Content Moderation Gate (Prerequisite to Indexing)

Before any video enters the content extraction and indexing pipeline, it passes through
a platform-level content moderation system that runs immediately at upload time. This is
entirely separate from the search stack.

```
Video uploaded by creator
      ↓
Content Moderation Pipeline (runs once, at upload time)
  ├── Hate speech / harassment classifier
  ├── Misinformation detector (medical, election, crisis topics)
  ├── Spam / artificially promoted content detector
  ├── Copyright check (Content ID audio/visual fingerprint matching)
  └── Age-rating classifier (produces: ALL_AGES / TEEN / ADULT)

Outcome:
  VIOLATION → video removed or restricted from platform entirely
  APPROVED  → video proceeds to indexing pipeline below
  ADULT     → video indexed but tagged adult_rating=true
```

The embedding database **only contains videos that passed this gate**. No hate speech
or misinformation video ever reaches the bi-encoder, the retrieval index, or the SLM
ranker — they were eliminated before indexing began.

**Two categories of content are indexed but filtered at retrieval, not removed globally:**

Age-restricted content: an ADULT-rated video is not removed from the index. It is tagged
(`adult_rating=true`), and the tag is consumed at retrieval pre-filter time (ABM layer)
based on the requesting user's safe search setting, which the QU layer extracts from user
context. The same video is visible to adults and filtered out for minors — that decision
depends on who is asking, not on what the video contains, so it must be query-time.

Geo-restricted content: a video licensed only for specific geographic regions (e.g., a
music video licensed only in the US, or a film available only in the EU) is also not
removed from the index. It is tagged with `available_regions: [US, CA, ...]`, and the
ABM pre-filter excludes it when the requesting user's region is not in that list. QU
extracts the user's region from IP / account locale and passes it as a hard filter
attribute. The same video returns results for a US user and is invisible to a user in
Japan — query-time filtering based on user context, not index-time removal.

Geo-restriction is a licensing constraint, not a content violation. Removing it from
the index globally would break it for all the regions where it is legitimately available.

---

### Features Used for Video Content Embedding

Three raw sources are available per video. Each has a different signal quality and token cost:

| Source | Signal quality | Token count | Problem |
|---|---|---|---|
| Title | Gameable — uploader-controlled, often clickbait | ~15 tokens | Misleading but crucial for exact-match |
| Description + Tags | Semi-reliable — uploader-controlled, keyword-stuffed | ~200 tokens | Noisy, redundant |
| ASR Transcript | Ground truth — what the video actually says | ~2000–8000 tokens | Too long, noisy (filler words, ASR errors) |

The bi-encoder document encoder takes a **single text string** as input — it cannot
consume 8000 raw transcript tokens. The pipeline below resolves this.

### How Features Are Converted to Bi-Encoder Input

```
Video uploaded
      ↓
Step 1 — Raw extraction (runs in parallel):

  Audio → ASR model (Whisper-class)
    → raw transcript, ~2000–8000 tokens, noisy
    "uh so today we're gonna uh make pasta carbonara it's really easy so
     first you want to like get your ingredients together uh we need eggs..."

  Metadata — uploader-provided:
    title:       "5-minute carbonara 🍝 | SO EASY"
    description: "Quick pasta carbonara recipe! No cream! Subscribe for more!"
    tags:        ["carbonara", "pasta", "easy recipe", "Italian", "5 minute meals"]
    category:    Food & Cooking
    channel:     CookingWithNonna
    duration:    12:34
      ↓
Step 2 — Transcript summarization (RL summarization model, 1.7B):

  Input:  raw transcript + description (~3000 tokens total)
  Output: clean content summary, ~200–300 tokens

  "Step-by-step pasta carbonara tutorial. Ingredients: eggs, pecorino romano,
   guanciale, black pepper, spaghetti. No cream. Covers common mistakes: adding
   eggs off heat to avoid scrambling. Difficulty: beginner. Cook time: ~15 min."

  Why not just use the title/description? Title is "5-minute carbonara 🍝" — it tells
  the bi-encoder almost nothing about what the video actually contains. The transcript
  summary captures the real content, which is what the query needs to match against.

  Why not feed raw transcript to bi-encoder directly? At 8000 tokens, it exceeds the
  bi-encoder's context window (typically 512–1024 tokens) and floods it with noise.
      ↓
Step 3 — Structured document construction:

  Title and tags are kept verbatim alongside the summary — NOT discarded. They carry
  exact-match signal for queries that use the uploader's exact keywords. The summary
  compresses the noisy transcript content; it does not replace metadata.

  Structured text document assembled for bi-encoder input:

  "Title: 5-minute carbonara pasta easy recipe
   Tags: carbonara pasta easy recipe Italian 5 minute meals
   Category: Food & Cooking | Duration: short (under 15 min)
   Content: Step-by-step pasta carbonara tutorial. Ingredients: eggs, pecorino romano,
   guanciale, black pepper, spaghetti. No cream. Covers common mistakes: adding eggs
   off heat to avoid scrambling. Difficulty: beginner. Cook time: ~15 min."

  Total: ~120 tokens — fits the bi-encoder context window cleanly.
      ↓
Step 4 — Bi-encoder document encoding (offline, per video):

  structured document text → document encoder → e_d (768-dim vector)
  Stored in retrieval index, keyed by video_id.
```

**What is cached per video after this pipeline:**

```
{
  video_id:          "abc123",
  document_embedding: e_d,            ← used by bi-encoder retrieval
  structured_doc:    "Title: ... Content: ...",  ← used by BM25 inverted index
  content_summary:   "Step-by-step pasta carbonara...",  ← used by SLM ranker prompt
  visual_embedding:  [v1, v2, ...],   ← CLIP keyframe vectors, used by ranking features only
  engagement_stats:  {...}            ← click rate, avg completion, etc.
}
```

**Why title and tags are kept verbatim (not replaced by summary):**

```
Query: "CookingWithNonna carbonara"   ← user searching by channel name
  → "CookingWithNonna" is in the title/channel field
  → bi-encoder over summary text alone would miss this exact-match signal
  → BM25 inverted index catches it; bi-encoder structured doc also includes it

Query: "5 minute carbonara"           ← user wants a quick recipe, uses exact tag phrase
  → "5 minute" appears in the original title/tags
  → summary says "cook time: ~15 min" (correcting the misleading title) — semantic mismatch
  → title field in structured doc preserves the exact tokens for BM25 to match
```

The summary corrects the misleading content; the title preserves the searchable keywords.
Both are needed in the bi-encoder input document for different query types.

**Multi-modal: text vs visual query path:**

Most YouTube search is text-query → video content. The bi-encoder maps both into the same space:

```
Text query: "sunset timelapse nature"
      ↓ query encoder → e_q (text-space)

Video document: structured text (title + tags + content_summary, ~120 tokens)
      ↓ document encoder → e_d (same embedding space)

cos(e_q, e_d) → retrieval score
```

Visual embeddings (CLIP keyframes) are NOT part of the bi-encoder input — they live in a
separate vector space (image-space) and cannot be compared directly against a text query
embedding. They are used as auxiliary ranking features (similarity score between query
embedding and visual embedding as one feature input to the SLM ranker), not as retrieval
vectors. Text-to-text matching is the primary retrieval mechanism for typed queries.

---

## 5. Stage 1 — Retrieval

### Objective Chain

```
Business goal: the best video in the corpus is useless if it was never retrieved.
  Retrieval sets the ceiling on result quality — nothing in ranking can recover a missed video.

↓ System must do: return a candidate set of K videos that covers all potentially relevant videos

↓ ML objective: maximize Recall@K
  Every video that could satisfy the user's intent must appear in the top-K candidates.
  Precision is secondary at this stage — ranking handles that.
```

### Hybrid Retrieval: Bi-Encoder + Inverted Index

YouTube search uses both retrieval mechanisms in parallel and merges results:

```
Query: "easy pasta carbonara tutorial"
      ↓
┌─────────────────────────┐    ┌─────────────────────────┐
│  Semantic (bi-encoder)  │    │  Lexical (BM25 / inverted│
│                         │    │  index)                  │
│  e_q = encode(query)    │    │  tokens: [easy, pasta,  │
│  scan embedding index   │    │  carbonara, tutorial]    │
│  top-500 by cos(e_q,e_d)│    │  top-500 by BM25 score  │
└─────────────────────────┘    └─────────────────────────┘
               ↓                              ↓
            Merge + deduplicate (RRF or learned fusion)
               ↓
         Top-1000 candidates → Stage 2 Ranking
```

**Why both? Why not just semantic?**

| Signal | Semantic bi-encoder | BM25 / inverted index |
|---|---|---|
| "easy pasta carbonara" (synonym-rich) | Strong — embeddings capture semantic similarity | Weak — exact token match only |
| "Adam Ragusea carbonara" (person name) | Weak — embeddings don't reliably handle rare names | Strong — exact token match |
| Song title / movie name (exact) | Risky — similar titles confuse embeddings | Strong — exact string match |
| Short ambiguous query: "pasta" | Strong — embeddings use broad semantic space | Weak — too many matches |

Hybrid retrieval dominates either alone. Entity queries and exact-match queries need BM25.
Semantic queries and intent-rich queries need the bi-encoder.

### Bi-Encoder Architecture

Same principle as LinkedIn's LiNR (semantic_search.md §11):

```
query text → encoder_q → e_q (768-dim)
                                       cos(e_q, e_d) → retrieval score
video summary text → encoder_d → e_d (pre-computed, stored in index)
```

**Training objective:** InfoNCE contrastive + pairwise margin loss (identical to semantic_search.md §11):

```
L = λ·L_InfoNCE + (1-λ)·L_pair

Positives: (query, video) pairs where video was watched with high completion rate
Negatives: in-batch negatives + hard negatives (top retrieval results rated irrelevant by oracle LLM)
```

**Hard negative mining is critical:**
Easy negatives (random videos) are trivially separable. Hard negatives are top-retrieved videos
that look semantically similar but are irrelevant for the specific query. Training on hard negatives
sharpens the decision boundary near relevance threshold — same finding as LinkedIn (semantic_search.md).

### Pre-Filtering: Attribute-Based Matching

Before the embedding scan, hard constraints from QU are applied as bitmap masks:

```
Query intent: "music video"          → mask: content_type = MUSIC_VIDEO
Query: explicit safe search off      → mask: safe_search_rating = SAFE
Query language: Spanish              → mask: primary_language IN [es, es-419]
Filter: "uploaded this week"         → mask: upload_date >= (now - 7 days)
```

This eliminates ineligible candidates before the cosine scan — no liquidity problem
(unlike ANN post-filtering), every valid candidate is considered. Identical to LiNR's ABM.

**The full set of pre-filter attributes (all extracted by QU from user context + query):**

```
Query intent: "music video"            → content_type = MUSIC_VIDEO
Filter: "uploaded this week"           → upload_date >= (now - 7 days)
User region: "JP" (Japan)              → available_regions CONTAINS "JP"
                                         (excludes US-only licensed videos)
User: minor / safe search on           → adult_rating = false
                                         (excludes age-restricted videos)
Query language: Spanish                → primary_language IN [es, es-419]
```

Geo-restriction and age-rating are enforced here — at the pre-filter step before the
embedding scan — because they are hard eligibility constraints, not ranking signals.
A US-only video shown to a Japan user is not a bad ranking; it is a licensing violation.
Pre-filtering ensures it never appears in the candidate set at all.

---

## 6. Stage 2 — Ranking

### Objective Chain

```
Business goal: the first page of results determines whether the user watches.
  A bad #1 result ends the session. A correct #1 result that matches query intent
  drives watch time, completion, and trust in search.

↓ System must do: of the 1000 retrieved candidates, score each one on how well
  it satisfies THIS specific user's query intent AND predict whether they will watch it

↓ ML objective: maximize NDCG@10 (relevance quality at top 10 positions)
  AND predict engagement probability per video (click, watch fraction, completion)
  Optimize jointly: relevance AND engagement
```

**Why joint relevance + engagement, not just relevance?**

```
Relevance only: surface the most "correct" video by topic match
  → video titled "pasta carbonara - authentic 1940s Roman recipe" is relevant
  → but it's 3 hours long and the user wanted a quick tutorial
  → user doesn't watch → quality-adjusted watch time = 0

Joint relevance + engagement: surface correct video that user will actually watch
  → "easy 10-minute pasta carbonara step by step" is both relevant and engagement-predictive
  → user watches, completes, searches again next week
```

### Architecture: Cross-Encoder SLM (Same as semantic_search.md §2)

A cross-encoder reads the full query AND the video content together — query and document
tokens see each other, enabling token-level relevance reasoning that a bi-encoder cannot do:

```
[query tokens | video summary tokens | structured features]
      ↓ decoder-only SLM (cross-encoder)
      ↓ document tokens attend to all query tokens
      → relevance_prob, click_prob, watch_completion_prob
```

Input prompt to the SLM:
```
System: "You are a search ranking model for YouTube. Given a query and a video description,
         predict whether this video is relevant to the query and whether the user will watch it."

User:   Query: "easy pasta carbonara tutorial"

        Video: "How to Make Authentic Pasta Carbonara in 12 Minutes
                Step-by-step tutorial for beginners. Ingredients: eggs, pecorino romano,
                guanciale, black pepper, spaghetti. No cream, classic Italian recipe.
                Duration: 12 minutes | Channel: CookingWithNonna | Views: 2.4M"

        Structured features:
          channel_subscriber_count: 1.2M
          avg_completion_rate_last_30d: 0.74
          query_video_click_rate_history: 0.031
          network_views_of_this_video: 142 (friends watched this)
```

Output: Yes/No first-token logit → relevance_prob + engagement_probs

### Multi-Task Distillation (Same Structure as semantic_search.md §8)

```
                 LLM Oracle (8B+)
          (offline relevance judge — grades 1-4)
                       ↓
         ┌─────────────┴──────────────┐
         ▼                            ▼
  Relevance Teacher (1.7B)    Engagement Teacher (1.7B)
  trains on oracle grades     trains on watch logs:
  {1,2,3,4} → soft labels     click, watch_fraction,
                               completion, skip, dislike
         └─────────────┬──────────────┘
                       ▼
              Student SLM (0.6B → pruned)
          jointly predicts relevance + engagement
```

**Engagement signals in YouTube search context (richer than LinkedIn):**

| Signal | What it means |
|---|---|
| Click | User chose this video from search results |
| Watch fraction (>70%) | Video matched query intent — user watched willingly |
| Watch fraction (<10%) | Mismatch — clicked but immediately left |
| Return to search results | Strong negative — result did not satisfy query |
| Skip at 5 seconds | Thumbnail / title misled user |
| Like / save from search | Strong relevance endorsement |
| Share from search | Very strong endorsement |
| "Not interested" signal | User explicitly rejected this type of content |

**Loss masking for rare signals** (same principle as semantic_search.md §7):
Likes and shares are far sparser than clicks. Without loss masking, the engagement head
for "share" collapses to near-zero predicted probabilities. Masking: treat a video as
negative for "share" only if at least one video in the same query session received a share.

### Feature Engineering

Three input types combined in the SLM prompt:

**1. Text features (SLM reads directly):**
```
Pre-computed video summary (~300 tokens) — title + description + transcript compressed
Channel name, category, duration class ("short: <5min", "long: >20min")
```

**2. Structured numerical features (encoded in prompt):**
```
avg_completion_rate_last_30d          → float, descriptive identifier
channel_subscriber_count              → log-scaled integer
click_through_rate_for_this_query     → float (query-video pair historical CTR)
days_since_upload                     → integer (freshness signal)
```

Descriptive identifiers outperform terse ones: "avg_completion_rate_last_30d" not "cr30"
— same finding as LinkedIn's feature encoding ablation (+5.8% AUC from descriptive IDs).

**3. User personalization features (search context — different from recommendation):**
```
user_watch_history_embedding          → similarity to this video's embedding
user_search_history_last_10_queries   → text of prior searches this session
user_preferred_video_length           → derived from past watch behavior
user_language_primary                 → for non-English search
```

**Critical distinction — personalization in search vs recommendation:**
In recommendation, personalization is the primary signal (show me what I'll like). In search,
the query is primary and personalization is a tie-breaker. A user who usually watches 3-hour
films still wants the "easy 10-minute pasta tutorial" if that's what they searched for.
Over-personalizing search surfaces what the user usually watches, not what they asked for.

```
Ranking score = α × relevance_score(query, video)
              + β × engagement_score(user, video)   ← weighted lower than relevance
              + γ × personalization_score(user, video)  ← tie-breaker only
```

---

## 7. Stage 3 — Re-ranking and Business Rules

### Objective Chain

```
Business goal: even a perfectly relevant, engagement-predictive ranking can fail long-term
  if it always surfaces the same creators (monopoly), stale content for time-sensitive queries,
  or violates platform trust (borderline content at top)

↓ System must do: apply soft constraints that serve long-term platform health, not just
  per-query ranking accuracy

↓ ML objective: constrained optimization
  maximize quality-adjusted watch time subject to:
    diversity(creator, topic) ≥ threshold
    freshness(time-sensitive queries) ≥ threshold
    creator_distribution ≠ monopoly on top-10
```

### Freshness Injection

For queries where recency matters, freshness re-scores candidates:

```
Time-sensitive query detection: "news", "breaking", "today", "latest", "2024", sports scores
  → detected by QU intent classifier (intent = NEWS or LIVE_EVENT)

Re-scoring:
  raw_rank_score = relevance_score × engagement_score
  freshness_boost = decay_function(days_since_upload) for time-sensitive queries only
  final_score = raw_rank_score × freshness_boost

decay_function:
  uploaded_1h_ago:   boost = 2.0
  uploaded_24h_ago:  boost = 1.5
  uploaded_7d_ago:   boost = 1.1
  uploaded_30d_ago:  boost = 1.0 (no boost)
```

**Why not always boost fresh content?**
For "how to make pasta carbonara," a 5-year-old video with 2.4M views and 74% completion rate is
far better than a video uploaded today with no watch history. Freshness only matters for queries
where the event itself is time-bound. The QU intent classifier determines this.

### Creator Diversity

```
Problem: the top-10 for "guitar tutorial" is all one channel
  → user sees no variety, may miss better content from smaller channels
  → long-term: smaller creators get no distribution, platform health degrades

Solution: max-N per creator in top-10 (configurable, default N=2)
  → if slots 1, 2, 3 would all be the same channel, demote 3 to slot 5
    and pull the next distinct-creator video up

MMR (Maximal Marginal Relevance) variant:
  score_mmr(v) = λ × relevance_score(v) - (1-λ) × max_sim(v, already_selected)
  where max_sim = maximum similarity between v and any video already in top-10
```


---

## 8. Personalization in Search

### Objective Chain

```
Business goal: the same query "python" means "beginner programming tutorial" for a high school
  student and "Python 3.11 asyncio performance" for a senior engineer.
  Serving the same results to both users is a missed opportunity.

↓ System must do: use what we know about the user to break ties in relevance-equivalent candidates

↓ ML objective: lift in query satisfaction NDCG given user history features as ranking inputs
  (A/B test: NDCG@10 with personalization vs without, for the same set of queries)
```

### What Personalization Signals Are Used

```
User's search history (this session + last 30 days):
  → "python asyncio" in history → current "python" query → lean toward advanced content

User's watch history embeddings:
  → average embedding of last 50 watched videos
  → cosine similarity with candidate video embeddings → preference alignment score

User's preferred video duration:
  → derived from median watch session duration → penalize 3-hour videos for short-session users

User's channel subscriptions:
  → subscribed channels' videos get soft boost when also relevant
  → not a hard filter — subscriptions don't override relevance

User's device and context:
  → mobile at midnight → prefer shorter videos
  → TV app → prefer longer videos
  → commute pattern (if inferable) → podcast-style audio content preferred
```

### Personalization Boundary: Search vs Recommendation

```
Search: user expressed explicit intent via query
  → query is primary, personalization is secondary
  → never surface content that ignores the query because "user usually likes this"

Recommendation: no explicit query, surface what user will like
  → personalization is primary, relevance is the signal itself

Failure mode in search over-personalization:
  User searches "vegetarian recipes"
  System knows user always watches steak videos
  → system serves "protein-rich steak alternatives to consider" (gamed by personalization)
  → correct behavior: serve vegetarian recipes — user explicitly asked for them
```

---

## 9. Cold Start — New Videos

### Objective Chain

```
Business goal: 500 hours of video uploaded per minute. A new video has no engagement history.
  Without explicit handling, it would never be retrieved (bi-encoder trained on engagement signals)
  and never ranked (SLM uses historical click/watch features that don't exist yet).
  New videos = cold start problem at industrial scale.

↓ System must do: surface new videos that are relevant before any engagement data exists

↓ ML objective: proxy relevance without engagement history
  use content-quality signals and creator authority as substitutes for watch data
```

### Cold Start Strategy

**Why cold start is different in search vs recommendation:**
In recommendation cold start, we don't know WHAT the video is about or whether users will like it.
In search, we know the user's query intent. If the content representation (transcript + title +
description embedding) is semantically aligned with the query, the video is retrievable — even
with zero views. The cold start problem in search is primarily a ranking problem, not a retrieval problem.

**Retrieval: works from day one (content-based):**
```
Video uploaded with title + description + transcript
      ↓ nearline: ASR + summarization + bi-encoder → e_d stored in index
      
At query time: cos(e_q, e_d) — this works regardless of view count
  → a brand new tutorial on "pasta carbonara" IS retrievable if the embedding aligns
  → retrieval is already content-based; zero engagement needed
```

**Ranking: where cold start hurts:**
```
SLM ranking features for a 5-year-old viral video:
  avg_completion_rate_last_30d: 0.74   ← available
  channel_subscriber_count: 1.2M      ← available
  click_through_rate_for_this_query: 0.031 ← available
  
SLM ranking features for a video uploaded 5 minutes ago:
  avg_completion_rate_last_30d: NULL  ← no data
  channel_subscriber_count: 500       ← available (creator authority proxy)
  click_through_rate_for_this_query: NULL ← no data
```

**Cold start ranking approach:**

```
Step 1 — Imputation: replace NULL engagement features with content-quality proxies

  avg_completion_rate_last_30d NULL  →  predicted from content quality signals:
    - transcript coherence score (LLM quality judge, offline per video)
    - title-description alignment (title matches transcript topic → likely quality)
    - video duration vs typical duration for this category
    - creator's historical completion rate on their past videos (channel-level, not video-level)

Step 2 — Creator authority: use channel-level signals as a floor
  channel_avg_completion_rate → proxy for this video's expected quality
  channel_topic_expertise_score → does this creator's past videos perform well in this category?

Step 3 — Temporal exploration: explicit traffic allocation for new videos
  Reserve 5% of search traffic slots for content published < 7 days ago
  This creates the feedback loop: new video gets traffic → engagement data accumulates
  → ML features become available → cold start exits after ~7 days of organic traffic
```

**The virtuous cycle:**
```
New video uploaded
  → content extracted → embedded → retrievable (day 0)
  → 5% slot allocation → first traffic
  → engagement signals accumulate (days 1–7)
  → SLM engagement features available
  → video competes on ML quality (day 7+)
  → slot allocation released
```

---

## 10. Training Pipeline

### Label Hierarchy

The training pipeline mirrors semantic_search.md but adapted for video engagement signals:

```
OFFLINE:
  Search click logs + watch logs + survey ratings
        ↓
  LLM Oracle (8B) — offline relevance judge
    Input: query + video title + description + transcript excerpt
    Output: relevance grade {1: poor, 2: fair, 3: good, 4: excellent}
    Annotation: "query asks for easy tutorial, this is a 3-hour lecture → grade 2"
        ↓
  Relevance Teacher (1.7B) — trained on oracle grades
  Engagement Teacher (1.7B) — trained on click, watch_fraction, completion, skip, dislike logs
        ↓
  Student SLM (0.6B → 375M pruned)
    distilled from both teachers
    ranks top-250 candidates per query in production

NEARLINE (per video update):
  Video uploaded / metadata updated
        ↓
  ASR: audio → transcript
  Summarizer (RL): transcript + description → 300-token summary
  Bi-encoder: summary → document embedding e_d stored in retrieval index

ONLINE:
  Per search request:
  Query → [QU → bi-encoder (query tower) + BM25] → top-1000
  → fetch pre-computed summaries by video ID
  → SLM cross-encoder scores top-250
  → re-ranking layer
  → top-20 shown to user
```

### Training Data Quality: The Oracle is the Gatekeeper

A key design decision: what counts as a "relevant" video for a given query?

**Engagement signals alone are biased:**
```
Popular video for "python tutorial": 10M views, 80% completion
  → pure engagement says: highly relevant to all python queries
  → reality: it is advanced, intermediate searchers love it, beginners hate it
  → engagement signal is correct only when you condition on user context

LLM Oracle (8B): reads query + video content → produces grade conditioned on:
  - does this video actually answer the query intent?
  - is the difficulty level appropriate for the implied audience?
  - does the title match the actual content?
  → grade that engagement history cannot reliably capture
```

The oracle produces labels that capture video-query relevance independent of how popular
the video already is. This breaks the rich-get-richer bias that pure engagement labels create.

---

## 11. Inference Optimizations

The ranking SLM (0.6B → 375M pruned) runs on top-250 candidates per query at production QPS.
The same optimization stack from semantic_search.md §12 applies directly:

### Shared Prefix Amortization

All 250 candidate prompts for one query share the same prefix:
```
[system prompt | query text | user context (personalization features)]
← same 300-400 tokens for all 250 candidates →
```

Compute prefix KV cache once per query, share across all 250 candidates.
Same FLOPs math as semantic_search.md §12 — 59% compute reduction at real prefix lengths.

### Scoring-Only Prefill

Ranking is classification, not generation. The SLM needs one logit (Yes/No relevance),
not a sequence of generated tokens. Standard LLM inference wastes:
- Autoregressive decode loop
- Log-probs over full 50k vocabulary
- KV cache maintenance for future decode steps

Scoring-only prefill path: single forward pass → grab final-position logit → return score.
Same 44% throughput gain (900 → 1,300 items/s/GPU, semantic_search.md §12).

### Context Summarization (Video-Specific)

Without summarization, the SLM prompt would include raw transcript (~3,000 tokens per video).
RL summarization (1.7B model, nearline per video) compresses to ~300 tokens.
Same ~4× throughput gain as LinkedIn context summarization (semantic_search.md §12).

### CUDA Graphs

After all other optimizations, per-kernel CPU dispatch overhead dominates.
CUDA graphs record the exact kernel sequence once per bucket size, replay with one GPU command.
Bucketing approach: pad all video summary inputs to nearest size bucket [32, 64, 128, 256 tokens].
Same +10% final-mile gain as LinkedIn (2,000 → 2,200 items/s/GPU).

---

## 12. Evaluation

### Offline Metrics (Per Layer)

| Layer | Metric | What it captures |
|---|---|---|
| Retrieval | Recall@K (K=1000) | Fraction of relevant videos that appear in top-1000 |
| Ranking | NDCG@10 | Quality of ranking in top-10 positions |
| Ranking | MRR (Mean Reciprocal Rank) | Position of first relevant result |
| Ranking | PMR@10 (Poor Match Rate) | Fraction of top-10 that are clearly wrong |
| Engagement | Watch fraction AUC | Ability to predict watch completion |
| Engagement | Click AUC | Ability to predict whether user clicks |
| Cold start | Recall@K for videos < 7 days old | Cold start coverage |

### Online Metrics (A/B Test)

| Metric | How it maps to business objective |
|---|---|
| Query satisfaction rate | % of search sessions where user watched a video willingly (no "return to results") |
| Avg watch fraction from search | Quality-adjusted watch time proxy |
| Session depth after search | Did search lead to continued watching? |
| Return-to-search rate (negative) | User found result and didn't reformulate — good |
| Search abandonment rate (negative) | User gave up without clicking — bad |
| Creator diversity (Gini coefficient) | Are top results monopolized by a few channels? |
| New video CTR | Are cold start videos getting exposure? |

### The Guardrail Metric

A ranking improvement is shipped only if it does not degrade:
```
Harmful content exposure rate (hard floor — any regression blocks ship)
Creator diversity score (Gini < threshold)
Cold start video coverage (new videos must stay discoverable)
```

---

## 13. Key Design Decisions

| Decision | Why |
|---|---|
| Two-stage retrieval + ranking | Cross-encoder too slow for billions of videos; bi-encoder gives recall, SLM gives quality |
| Hybrid retrieval (bi-encoder + BM25) | Entity queries, song titles, person names need exact-match; semantic queries need embeddings |
| Transcript as primary content signal | Titles/tags are gamed; transcript reflects actual video content |
| RL summarization before SLM ranking | Raw transcript = 3,000 tokens → 4× throughput hit; compress to 300 first |
| Relevance primary, engagement secondary | Search query expresses explicit intent — over-personalizing serves the wrong result |
| Oracle LLM for training labels | Engagement-only labels have rich-get-richer bias; oracle grades based on content-query match |
| Cold start: 5% slot allocation | Without explicit traffic budget, new videos never accumulate data → permanent cold start |
| Freshness only for time-sensitive queries | QU intent classifier detects NEWS/LIVE queries; boosting freshness universally hurts tutorial quality |
| Creator diversity in re-ranking | Prevents monopoly; protects long-term creator ecosystem and user variety |
