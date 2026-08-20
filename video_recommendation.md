# Video Recommendation System

Systems: TikTok For You Page, Instagram Reels, YouTube Shorts, YouTube homepage

---

## 1. Problem Framing

### Business Objective

**The Three-Legged Stool — balancing users, creators, and the platform:**

Any recommendation system on a two-sided marketplace must balance three stakeholders.
Over-optimizing one leg destabilizes the others.

```
User satisfaction:        users find relevant, quality content → retention, engagement
Creator sustainability:   new creators can join and grow, best creators get rewarded
Platform revenue:         watch time → ad impressions → revenue
```

**How platform revenue actually works:**

```
User watches 30 min of videos
      ↓
Platform inserts ads between videos (every ~3-5 videos)
      ↓
More watch time = more ad slots served = more revenue

Revenue ≈ (ad impressions) × (CPM — cost per 1000 impressions)
CPM is set by advertisers bidding for attention

User watches 30 min → sees ~8 ads  → 8 impressions
User watches 5 min  → sees ~1 ad   → 1 impression

At 1B DAU: every extra minute of avg watch time per user
           = 1B extra minutes consumed across platform
           = ~300M extra ad impressions per day
           = hundreds of millions in annual revenue
```

**Why naive "maximize watch time" creates the stool problem:**

```
Naive: recommend addictive content — rage bait, anxiety loops, "one more video" rabbit holes
  → user watches 3 hours today but feels worse after
  → next day opens app less
  → week 4 retention drops

Platform revenue over 1 year (burned-out user) <
Platform revenue over 1 year (healthy user at 30 min/day)
```

A user retained for 3 years at 30 min/day is worth more than a user who watches
3 hours for one week then quits. The three legs are aligned **long-term** — they
only conflict **short-term** when you over-optimize one of them. Quality-adjusted
watch time is the metric that resolves this: it filters addictive but low-quality
content that drives short-term viewing but burns users out.

This framework applies beyond video — it is universal to any recommendation system:

| Platform | User | Creator/Supplier | Platform |
|---|---|---|---|
| TikTok/YouTube | Quality content, discovery | Distribution for new creators | Watch time → ad revenue |
| LinkedIn Feed | Relevant posts | Post gets distribution | Engagement → premium subs |
| Spotify | Good music discovery | Artist streams, royalties | Subscriber retention |
| E-commerce | Find right product | Seller visibility | GMV, commission |
| Yelp | Find good business | Small business discovery | Ads, bookings |

**Why naive metrics fail — the optimization pitfall hierarchy:**

```
Optimize CTR:
  → clickbait thumbnails, misleading titles
  → users click but don't watch → leave platform
  → trust erodes, long-term retention drops

Optimize raw watch time:
  → addictive but low-quality content gets promoted
  → bias toward longer videos regardless of quality
  → short-form creators disadvantaged
  → user wellbeing harm → regulatory risk

Optimize quality-adjusted watch time:
  = watch time × satisfaction signal
  → filters clickbait: high CTR but low completion = penalized
  → filters addictive junk: watched but rated poorly = penalized
  → aligns model incentive with actual user value
```

**Business objective: maximize quality-adjusted watch time.**

```
Quality-adjusted watch time = watch time × quality_score

Quality signals:
  Completion rate (>70% = watched willingly, not just left running)
  Explicit satisfaction survey ("Was this video worth your time?" thumbs up/down)
  Reshare / save (strong endorsement, harder to game than like)
  Return session rate (did user come back the next day?)
```

This objective naturally opens up deep dives: "how do you prevent recommending a
clickbait video with lots of initial clicks?" → completion rate penalizes it.
"How do you protect new creators?" → creator sustainability leg forces exploration budget.

```
Guardrails (must not break regardless of objective):
  Harmful content exposure       ← hard filter, not a tradeoff
  Rabbit hole / addiction rate   ← regulatory/ethical constraint
  Creator diversity (Gini coef)  ← platform health metric, not user metric
```

### Scale
```
TikTok:    ~1B DAU, ~100M new videos/day, p99 feed latency < 200ms
YouTube:   ~2B MAU, ~500 hours of video uploaded per minute
Instagram: ~2B MAU, Reels competes directly with TikTok for short-form share
```

### What Makes Video Rec Hard
1. **Content freshness** — a video can go viral in hours; cold start is critical
2. **Implicit feedback only** — users don't rate videos; you infer preference from behavior
3. **Rapid preference shift** — a user's taste this week may differ from last month
4. **Session context matters** — what you recommend after a funny cat video differs from after a news clip
5. **Creator ecosystem** — system must balance user satisfaction vs creator fairness

---

## 2. High-Level Architecture

```
User opens feed
      ↓
[Retrieval Layer]             ← bi-encoder: query user embedding vs corpus
  └── Exhaustive KNN          ← user embedding vs all video embeddings (GPU 1-bit XOR + FP16 re-score)
      ↓ (~1000 candidates, maximize RECALL — don't drop good candidates)
[Ranking Layer]               ← sequential transformer + MMoE
  └── Feature store lookup    ← fetch rich features for each of ~1000 candidate IDs
  └── Sequential transformer  ← user temporal history × candidate features
  └── MMoE multi-task heads   ← predict CTR, watch time, share, skip, etc.
      ↓ (~20–50 candidates, maximize PRECISION@K — accurate relevance scores)
[Re-ranking / Value Model]    ← apply business objective, diversity, hard filters
                                 creator promotion, viral boost, policy enforcement
      ↓ (~20 shown)
User sees feed
      ↓
[Engagement signals collected] → [Online embedding update pipeline]
                               → [Training data pipeline]
```

**Three distinct layers — each with a different job:**

| Layer | Input → Output | Optimize for | Key constraint |
|---|---|---|---|
| Retrieval | Billions → ~1000 | Recall — don't drop good candidates | Speed, KNN index quality |
| Ranking | ~1000 → ~20–50 | Precision@K — accurate relevance scores | Feature richness, model capacity |
| Re-ranking | ~20–50 → 20 shown | Business objective + diversity | Business rules, not ML |

---

## 3. Retrieval System

Retrieval Deep Dive — Applying LinkedIn LiNR Techniques to Video

### Objective Chain — Why Each Technique Exists

```
Business Goal:
  Maximize quality-adjusted watch time
  = users find content they genuinely enjoy → stay on platform → long-term revenue
        ↓
  Retrieval layer translates this as:

ML Goal — maximize RECALL:
  Recall@K = (relevant videos in top-K) / (total relevant videos in corpus)
  If retrieval misses a relevant video, no downstream ranker can ever surface it.
  A ranker with 80 good candidates beats a ranker with 1000 mediocre ones —
  but a ranker with 0 good candidates (all filtered out by poor retrieval) shows
  the user something irrelevant → session ends → platform revenue drops.
        ↓
  To maximize recall, we need:

ML Technique Goal — semantically rich embeddings:
  Relevance = user interest ↔ video content aligned in embedding space.
  Better embeddings = more accurate similarity = higher recall.
  Traditional MLP two-tower only knows what users co-watched → misses concept hierarchy.
  LLM bi-encoder brings world knowledge → bridges vocabulary gaps → richer alignment.
  InfoNCE training on (user, watched video) positives + (shown, skipped) hard negatives
  directly captures user interest signal from real engagement behavior.
        ↓
  Even with rich embeddings, recall is limited by HOW we search:

System Goal — exhaustive search + pre-filtering + quantization:
  ANN (approximate) + post-filter → liquidity problem → recall drops for filtered queries.
  Exhaustive KNN + ABM pre-filter → scan all qualifying videos → guaranteed recall.
  But exhaustive over 100M FP16 embeddings = 25.6 GB → GPU memory bottleneck.
  1-bit quantization + two-stage pipeline → fits GPU, latency < 50ms.
  Counting matrix + CUDA kernel → fast clause checking without warp divergence.
```

### 3.1 Retrieval Model to achieve business goal

**Goal:** Narrow billions of videos to ~1,000 candidates fast. Focus on Relevancy and quality content

**Business → ML link:** Quality-adjusted watch time requires RELEVANT content —
content that semantically matches the user's actual interests, not just content
that happens to share keywords or co-watch patterns with what they've seen before.
Better embeddings = closer alignment = higher recall = more relevant candidates
entering the ranker = better final recommendations.

**Why the traditional two-tower is limited:**

```
Traditional two-tower:
  video_id embedding  → learned from co-watch patterns only
  user_id embedding   → learned from engagement history only

Limitation 1 — only knows what users co-watched:
  Knows:    "yoga" ↔ "pilates" (users co-watched them)
  Doesn't know: "pilates" should also match "barre" / "flexibility training"
               → vocabulary mismatch, sparse co-watch data misses the concept hierarchy

Limitation 2 — new video cold start:
  New video published 5 minutes ago → no co-watch signal → no embedding → not retrieved
  LLM: video title + description → rich embedding from world knowledge immediately

Limitation 3 — vocabulary mismatch across creators:
  Creator A: "HIIT workout"
  Creator B: "interval training"
  Creator C: "cardio blast"
  Traditional: three separate embedding clusters (no co-watch signal bridging them)
  LLM: world knowledge maps all three to the same high-intensity cardio concept
```

**The LLM bi-encoder approach:**

Both the user tower and the video tower use the **same LLM encoder** (shared weights).
All inputs — regardless of modality — are serialized to text via a prompt library,
then encoded into the same vector space. Same architecture as LinkedIn Feed GR system.

```
User Prompt → [LLM Encoder] → user_embedding ∈ R^d
                                      ↕ dot product
Video Prompt → [LLM Encoder] → video_embedding ∈ R^d
```

**Cold Start Solution**


**Example 1 — brand hierarchy (same structure as geographic containment):**

```
New user first session: watches 3 "Tesla Model 3 review" videos

Non-LLM two-tower:
  Retrieves: more Tesla Model 3 reviews, Tesla Model Y reviews
  Misses: Rivian truck review, Lucid Air review, EV charging tips, range anxiety advice
  Why: Tesla → other EV brands requires co-click data
       If not many users watched both Tesla and Rivian reviews, embedding space
       doesn't bridge them — no co-click signal, no connection

LLM bi-encoder:
  World knowledge: Tesla Model 3 ⊂ Tesla ⊂ electric vehicles ⊂ automotive
  Day 1, no co-watch data needed:
    → surfaces Rivian reviews, Lucid Air comparisons, EV charging infrastructure videos
    → user discovers the entire EV ecosystem, not just one brand

"Tesla Model 3" = "San Francisco"   (specific brand / specific city)
"electric vehicles" = "Bay Area"    (category / broader region)
LLM knows Tesla ⊂ EV just as it knows SF ⊂ Bay Area
```

**Example 2 — vocabulary mismatch (no equivalent in LinkedIn's text domain):**

```
Creator A publishes: "HIIT workout"
Creator B publishes: "interval training"
Creator C publishes: "cardio blast"

User watches Creator A's content repeatedly.

Non-LLM two-tower:
  HIIT, interval training, cardio blast → separate clusters (no co-watch bridge)
  User only gets more Creator A content

LLM bi-encoder:
  World knowledge: HIIT = High Intensity Interval Training = interval training = cardio
  → bridges all three creators into one concept cluster
  → user discovers Creator B and C on day 1 without needing other users to co-watch them
```

---

**Video Tower — Prompt Design**

Videos are multi-modal (visual + audio + text). The LLM encoder only accepts text.
So the pipeline has an intermediate step: a video understanding model converts
visual/audio content into text-representable descriptions before the LLM runs.

```
Video at upload time:
  ┌───────────────────────────────┐
  │ Video Understanding Model      │  ← pre-trained vision/audio model (e.g. CLIP, Whisper)
  │  - scene classification        │
  │  - speech-to-text transcript   │
  │  - music/audio genre labels    │
  │  - activity recognition        │
  └──────────────┬────────────────┘
                 ↓ text labels + creator metadata
  ┌───────────────────────────────┐
  │ Prompt Template                │
  └──────────────┬────────────────┘
                 ↓
  ┌───────────────────────────────┐
  │ LLM Encoder                   │  → video_embedding stored in GPU index
  └───────────────────────────────┘
```

**Video prompt example:**

```
Format: short-form video, <duration_bucket>short</duration_bucket>
Creator: fitness channel, <creator_engagement_percentile>85</creator_engagement_percentile>
Category: fitness, HIIT, home workout
Title: 15-min Full Body HIIT — No Equipment
Description: high intensity interval workout, beginner-friendly, all fitness levels
Audio: energetic music, voice coaching
<view_percentile>72</view_percentile>
<completion_rate_percentile>68</completion_rate_percentile>
<velocity_percentile>91</velocity_percentile>
```

The **percentile bucket technique** from LinkedIn applies directly:
- Raw view count (e.g., 2.4M) → tokenizes badly as arbitrary digits
- `<view_percentile>72</view_percentile>` → single stable token, model learns ordinal meaning
- Same for completion rate, like rate, velocity (views/hour normalized by video age)

**User Tower — Prompt Design**

```
User prompt example:

User profile:
  Age bucket: <age_bucket>25-34</age_bucket>
  Region: United States
  Device: mobile, evening session

Recent watches (newest first, with engagement signal):
  [15-min HIIT workout, replay, t-30m]
  [quick yoga flow, 95% completion, t-2h]
  [healthy meal prep, like+save, t-1d]
  [morning run motivation, 80% completion, t-2d]
  [pilates for beginners, skip at 20%, t-3d]   ← negative signal also included

Subscriptions: fitness, yoga, wellness category
<watch_frequency_percentile>78</watch_frequency_percentile>
```

Watch history is included chronologically (newest first) — the LLM's attention
mechanism naturally weights recent interactions more heavily via position.


### 3.2 LLM Bi-Encoder: Data and Features

These features feed directly into the prompt templates shown in 3.1. Video features
populate the video prompt; user features and engagement signals populate the user prompt.
Engagement signals serve a dual role: they appear as action tokens A_t in the user
sequence at serving time, and they determine positive/negative pairs during training.

#### Video Features

Organized by update frequency — determines caching strategy at serving time.

**Content features (compute once at upload, cache long-term):**
```
Title, description, hashtags → text embedding
Thumbnail features           → computer vision embedding
Audio features               → music, speech, tone
Video quality metrics        → resolution, stability, production quality
Topic / category embedding   → semantic content representation
Duration
```

Content features are static — same video, same content. Compute once, serve forever.

**Engagement features (update frequently — velocity matters):**
```
Historical: view count, like rate, comment rate, share rate, avg completion rate
Velocity:   views/hour, engagement rate in last 1h/6h/24h
Creator:    creator reputation, historical avg engagement rate, monetization status
```

**Normalization is critical:** a video 30 seconds old has fewer views than one live
for a day — raw counts are misleading. Compare to expected growth curve for the video's
age, or use velocity (views/hour) rather than absolute counts.

**Feedback loop risk:** engagement features create a rich-get-richer dynamic — popular
videos get shown more, get more views, rank higher. Velocity metrics partially mitigate
this (a new video with high early velocity surfaces quickly), but explicit exploration
budget in the candidate generators is still necessary.

---

#### User Features

**Profile features (static or slow-changing, cacheable):**
```
Topic preferences (inferred from behavior, not just stated)
Language preferences
Subscription list embeddings (aggregate of followed creator embeddings)
Demographics (if available)
```

**Behavioral features (dynamic, update frequently):**
```
Long-term: favorite creators/categories (last 30/90 days)
Medium-term: top topics last 7 days
Short-term: time-of-day / day-of-week patterns
Session: recently watched videos, searches in current session
Device / platform preferences
```

---

#### Engagement Signals — Implicit Feedback Hierarchy

```
Strongest positive:
  Replay                          ← watched again voluntarily
  Share / Save                    ← strong endorsement
  Follow creator after watching   ← intent to see more

Strong positive:
  Full completion (>90%)          ← didn't skip
  Like / Comment                  ← explicit positive

Weak positive:
  >50% completion
  Profile visit after watching

Negative:
  Skip within first 3 seconds     ← thumbnail mismatch
  Scroll past without clicking    ← low interest
  "Not interested" explicit       ← strong negative
  Watch <10% then leave video     ← strong negative

Strongest negative:
  Leave platform entirely         ← platform attrition signal

Label construction:
  Binary: completion > 70% = positive, < 30% = negative
  Weighted: score = w1×completion + w2×like + w3×share + w4×replay - w5×skip_rate
```

### 3.3 - LLM Bi-Encoder Training

#### Architecture

We start from a pre-trained checkpoint (e.g. BERT-base, 110M params) — not trained from scratch.
World knowledge (semantic relationships: Tesla ⊂ EV, HIIT = interval training) is already
encoded in the pre-trained weights. Our job is to adapt it to capture engagement patterns.

```
Pre-trained LLM (BERT-base, 110M params)
         ↓
  [12 Transformer layers]
  → token embeddings (seq_len × 768)
         ↓
  Mean pooling over token embeddings
  → single vector (768-dim)
         ↓
  Linear projection head  (768 → 128)
         ↓
  L2 normalize → unit vector
```

Both user and video towers share the **same** LLM backbone — one encoder used twice.
Similarity = dot product of two unit vectors = cosine similarity.

#### Does fine-tuning destroy world knowledge?

InfoNCE doesn't push embeddings randomly — it pulls together pairs where the user
actually engaged. A user watching HIIT videos engaging with a "cardio blast" video
reinforces the semantic cluster the LLM already built. Fine-tuning refines within the
right neighborhood rather than overwriting it.

To be safe in production, we use **layer freezing**:

```
Freeze bottom 6 transformer layers  → general syntax + semantics locked in
Fine-tune top 6 transformer layers  → domain adaptation (video engagement patterns)
Fine-tune projection head           → embedding geometry tuned for ANN retrieval

Learning rates:
  Projection head:       1e-4   (new layer, learns fast)
  Top transformer layers: 1e-5  (fine-tune carefully)
  Bottom 6 layers:       frozen (world knowledge preserved)
```

Lower layers encode universal language structure that took billions of tokens to learn.
Upper layers specialize to the task. We only touch the top half.

#### Training objective — InfoNCE

Goal: learn a shared vector space where user embedding and video embedding point in
the same direction if the user's interest aligns with the video's content — and
point far apart if they don't.

```
InfoNCE loss:
  Positive pair:  (user, video the user watched and completed)
  Negatives:      all other videos in the same batch (in-batch negatives)

  loss = -log [ exp(u · v⁺ / τ) / Σⱼ exp(u · vⱼ / τ) ]

  τ = temperature (e.g. 0.07) — controls how sharp the similarity distribution is
```

**Hard negatives — impressed but not engaged:**

Random in-batch negatives are easy (unrelated videos). To sharpen the model, add
hard negatives: videos that were shown to the user but scrolled past without watching.

```
Easy negative:  user interested in fitness → shown a cooking video → clearly different
Hard negative:  user interested in fitness → shown a fitness video → scrolled past
                (semantically similar but user did NOT engage → model must learn
                 that semantic similarity alone is not enough, engagement is the signal)
```

Hard negatives prevent the model from producing sloppy embeddings that cluster
any fitness content together regardless of whether users actually watch it.

#### Serving

```
Offline: pre-compute all video embeddings → store in GPU exhaustive KNN index
Online:  encode user prompt → exhaustive KNN → top-1000 candidates in ~10ms
```

Video embeddings are static after upload (content doesn't change). User embeddings
are computed fresh per request (user's recent watch history changes every session).

### 3.4 — Exhaustive GPU KNN (adapting LiNR for video)

**Business → ML link:** High recall is the ML goal. ANN with post-filtering silently
destroys recall — the user asks for English workout videos, ANN returns 1000 globally
similar videos, post-filter leaves 80. Ranker works with 80 candidates instead of 1000.
The 920 dropped candidates included relevant videos the user would have enjoyed.
Every missed relevant video is a missed engagement event = missed quality watch time.
Exhaustive KNN + pre-filter guarantees the ranker always sees K qualified candidates.

**System → latency link:** Exhaustive over 100M FP16 embeddings = 25.6 GB memory read
per query. GPU can't hold this and still meet latency SLA. Solution: 1-bit quantization
(1.6 GB) + two-stage coarse-to-fine. System technique purely in service of the recall goal.

**Why ANN breaks down for filtered video retrieval:**

```
Scenario: user wants English workout videos only
  Total video corpus: 100M videos
  English workout videos: ~8M (8% of corpus)

ANN (FAISS HNSW):
  Index clusters by semantic similarity, not by language
  ANN retrieves top-1000 globally → post-filter by language
  Best case: maybe 80 English workout videos in top-1000 (liquidity problem)
  Ranker sees 80 candidates instead of 1000 → quality degrades

LiNR exhaustive + ABM pre-filter:
  Step 1: attribute filter → 8M English workout videos
  Step 2: exhaustive KNN on 8M only → guaranteed 1000 relevant candidates
  No liquidity problem
```

**Scale and quantization:**

```
100M videos × 128-dim × FP16 = 25.6 GB
  → doesn't fit on single A100 (80GB HBM) once you add model weights
  → use 1-bit Sign-OPORP quantization (LiNR V3)

100M × 128-dim × 1-bit = 1.6 GB → fits easily on single GPU

Two-stage pipeline (V3 Mode 2):
  Stage 1: 1-bit XOR scan across all 100M videos → keep top 1% (1M candidates)
           Memory read: 100M × 16 bytes (128-bit packed) = 1.6 GB → fast
  Stage 2: FP16 dot product on 1M survivors → accurate final scores
           Memory read: 1M × 256 bytes (128-dim FP16) = 256 MB → manageable

vs naive FP16 exhaustive: 100M × 256 bytes = 25.6 GB → 100× more memory
```

**Freshness pipeline (same three-pipeline structure as LiNR):**

```
Pipeline 1 — Prompt Generation:
  Event: new video published (CDC)
  Action: run video understanding model → extract text features → build LLM prompt
  Latency: seconds (video understanding is heavier than LinkedIn post prompting)

Pipeline 2 — Embedding Generation:
  Input: prompt from Pipeline 1
  Action: run LLM video encoder forward pass
  Output: video_embedding → written to embedding store (Venice equivalent)

Pipeline 3 — GPU Index Update:
  Input: new embedding from CDC stream
  Action: upsert into pre-allocated GPU tensor (no full index rebuild)
  Output: video retrievable within minutes of upload
```

---

#### Attribute-Based Pre-Filtering (ABM) for Video

**Business → ML link:** Two separate goals here.

Goal A — User trust (business): surfacing age-restricted content to minors or
wrong-language content to a user destroys trust immediately. These are not soft
preferences — they are hard constraints. Violating them damages retention more
than any recall improvement could recover. Pre-filtering enforces them before
any embedding similarity is computed.

Goal B — Recall (ML): pre-filtering is what enables exhaustive KNN to have recall
guarantees. Without pre-filtering, exhaustive over the full 100M corpus is too
expensive. With pre-filtering (say 8M qualifying videos), exhaustive KNN over 8M
is fast and guarantees recall within the valid set. The liquidity problem that ANN
creates is solved by this ordering: filter first, then scan.

**System → latency link:** The CUDA kernel (counting matrix + attribute matrix) is
a pure system optimization. Its job is to make the attribute clause checking fast
enough to not add latency to the query path. Without this GPU-native kernel,
checking 8M rows for language + region + age rating would be
a CPU bottleneck. The kernel runs all clause checks in parallel, one GPU thread
per video, with coalesced memory access — zero warp divergence, full throughput.

**What are the hard attribute constraints in video recommendation?**

```
Language:         user.preferred_language == video.language
                  (hard constraint — surfacing French videos to English-only user = bad)

Region licensing: user.country in video.licensed_regions
                  (content rights are per-country — not a soft preference, a legal constraint)

Age rating:       video.content_rating <= user.age_appropriate_rating
                  (cannot surface age-restricted content to minors)

Already watched:  video.id NOT IN user.recent_watches (last 30 days)
                  (don't re-serve content user just saw)

Explicit genre opt-out: user said "not interested in sports"
                        → video.primary_category != "sports" (soft → can make hard)
```

**Counting matrix + attribute matrix (same structure as LiNR):**

```
Each video row in GPU memory encodes:
  [language(1), licensed_regions(N), content_rating(1), creator_id(1), ...]

CUDA kernel (one thread per video):
  Check all clauses → mask[video_id] = 1 if all pass, else 0
  Same early-exit optimization as LiNR (sorted attribute values, stop on first miss)
```

**V1 vs V2 routing for video:**

```
Language filter only (loose filter, 80% of videos pass for English users):
  → V1: full matmul + apply mask → high pass rate, masking is cheap

Strong filter (user blocked 500 creators + language + age rating → 5% pass):
  → V2: clause check first → get qualifying video IDs → slice embedding matrix
        matmul on 5M instead of 100M → much cheaper

Route per query based on estimated pass rate — same as LiNR production decision.
```

---

### 3.5 - How overall retrieval system works

```
Video corpus: 100M videos
User query arrives
        ↓
ABM pre-filter:
  language=English, licensed_region=US, age_rating OK
  100M → 20M qualifying videos
        ↓
1-bit XOR scan (V3 Stage 1):
  user 1-bit embedding vs 20M 1-bit video embeddings
  20M → 200K candidates (top 1%)
        ↓
FP16 dot product (V3 Stage 2):
  Accurate scoring on 200K only
  200K → top 1000 candidates
        ↓
Ranking Layer → Re-Ranker → Result
```

**Why this beats ANN + post-filter (objective traceability):**
```
ANN on 100M → retrieve 5000 → post-filter by language + region → maybe 200 survive
  → ranker works with 200 candidates instead of 1000
  → recall loss → relevant videos missed → user sees irrelevant content
  → quality watch time drops → business goal missed

LiNR-style exhaustive + ABM pre-filter:
  → ranker always sees 1000 qualified candidates → no recall loss
  → no liquidity problem → business goal protected

Each layer of the stack is in direct service of the same goal:
  ABM pre-filter    → protect user trust + enable exhaustive recall
  1-bit quantize    → fit GPU memory → make exhaustive feasible at 100M scale
  Counting matrix   → clause checking at GPU speed → no latency penalty
  LLM bi-encoder    → semantically rich embeddings → higher precision recall
  InfoNCE training  → embeddings capture actual user interest from engagement signal
  All of the above  → quality-adjusted watch time
```

**How a newly uploaded video goes from raw upload to retrievable in the GPU KNN index**

```
[User]
   │ upload
   ▼
[Upload Server]
   │ store video file
   ▼
[Source of Truth — Blob Store]
   │
   │ CDC: video.created
   ├──────────────────────┬──────────────────────┐
   ▼                      ▼                      ▼
[Samza/Flink:         [Samza/Flink:          [Samza/Flink:
 Content Features]     Engagement Stats]      Creator Features]
 • scene labels         • view velocity        • creator tier
 • speech transcript    • early like rate      • avg completion rate
 • audio genre          • share rate           • category affinity
 • video quality        • completion bucket    • follower count bucket
   │                      │                      │
   └──────────────────────┴──────────────────────┘
                           │ write features
                           ▼
                    [Feature Store]
                           │
                           │ CDC: features.updated  (fired by each processor on write)
                           ▼
                  [Samza/Flink: Prompt Generator]
                  + Prompt Library (templates)

                  • Listens to feature.updated events
                  • Builds video prompt incrementally as features arrive
                  • Patches existing prompt when later features complete
                  • Buffers for 30 min — waits for all feature processors
                    to settle before treating prompt as final
                           │
                           │ batch push every 30 min
                           ▼
                  [LLM Bi-Encoder Service]
                  • Runs encoder forward pass on final prompt
                  • Produces 128-dim unit-vector embedding
                  • Stores in local embedding store
                           │
                           │ push every 30 min
                           ▼
                  [GPU Exhaustive KNN Index]
                  (production retrieval — 1-bit XOR + FP16 re-score)
                  • Upserts embedding into pre-allocated GPU tensor
                  • No full index rebuild — slot update only
                  • Video becomes retrievable within ~1 hour of upload
```

**Why the 30-min buffer at Prompt Generator?**

Different feature processors complete at different speeds:
- Content features (scene labels, transcript): seconds — lightweight models
- Engagement stats (velocity, early CTR): minutes — needs some view accumulation
- Creator features: fast lookup, nearly instant

Without buffering, the first CDC fires before engagement stats exist → incomplete
prompt → low-quality embedding. The 30-min window lets all processors finish and
the prompt generator patch the prompt incrementally as each CDC arrives.
Final push to LLM only happens once per video per window — not on every feature update.

---

## 4. Ranking System

**Objective Chain**
```
Business Goal:  Quality-adjusted watch time + long-term user retention
                (not raw clicks — a click that ends in 3 seconds hurts retention)

    ↓ why retrieval isn't enough
    Retrieval maximizes Recall — it surfaces 500 relevant candidates.
    But showing all 500 is impossible. The right video at position #450
    is as good as not retrieved. Ranking loss = watch time permanently lost.

ML Goal:        Precision@K — put the best videos at the top of the list
                - "Best" = user will actually watch and complete, not just click
                - Single-metric optimization (CTR alone) → clickbait wins
                  (high click, 3-second watch, user leaves frustrated)
                - Multi-objective: watch time + completion + engagement
                  together = proxy for quality that's hard to game

    ↓ connection
    Optimizing Precision@K with multi-task signals directly protects
    quality-adjusted watch time: completion rate catches short-loop bait,
    engagement signals catch genuine interest, skip rate catches mismatch.
```

**Goal:** Score each candidate precisely. Can afford heavier model (~100ms budget).

---

#### Ranking Deep Dive — Sequential Transformer Model

**Why a standard point-wise ranker is insufficient**

A standard ranker takes `user_embedding + video_embedding → score`. The user embedding
is a fixed vector — aggregated long-term watch history compressed into one representation.
That embedding is identical regardless of which candidate is being scored.

This loses the most important signal: **what mode is the user in right now?**

A user who watched 3 cooking videos in the last 20 minutes is in a cooking session.
A 4th cooking video should score much higher than their 6-month average suggests.
Session-level temporal behavior drives relevance more than long-term averages.

---

**The sequential model: feed the raw interaction history, not a compressed embedding**

Instead of a fixed user embedding, feed the model the sequence of past T=1000 interactions
as an interleaved (content, action) sequence — same design as LinkedIn's Feed SR:

```
X_in = [X_1, A_1, X_2, A_2, ..., X_T, A_T]  +  X_candidate

where each step:
  X_t = video features (embedding, category, duration, creator)
  A_t = action taken   (watch_time, completion_rate, liked, shared, skipped)
```

When scoring a candidate, the transformer attends over all 1000 past interactions to
build a **dynamic user representation** — one that changes per candidate.

- Cooking video candidate → model attends heavily to cooking interactions in history
- News video candidate   → model attends to news interactions in history
- User representation is not fixed — recomputed per candidate via attention

This is the core advantage over a standard ranker: the model asks "given everything
this user has done, how relevant is THIS specific candidate right now?"

---

**Why transformer over RNN/LSTM**

```
RNN/LSTM:
  Processes 1000 interactions sequentially → single hidden state
  Long-range forgetting: interaction at t=1 barely survives to t=1000
  Cannot parallelize during training

Transformer:
  Full attention over all 1000 positions simultaneously
  No bottleneck — any interaction can directly attend to any other
  Parallelizable: all positions computed in one forward pass
  Long-range dependencies preserved exactly
```

For a sequence of 1000 items, transformers are the only architecture that
doesn't lose the early part of the user's session history.

---

**Why decoder-only (causal attention), not bidirectional**

LinkedIn GR uses a decoder-only transformer — same family as GPT, not BERT.

At serving time, future interactions don't exist. Interaction at t=500 should
only attend to t=1..499. Bidirectional (BERT-style) attention would let future
interactions leak into past representations — correct for understanding tasks
(retrieval bi-encoder), wrong here.

Causal attention matches the task: model a user making decisions forward in time.
"Given what happened before position t, how good is the candidate at t+1?"

---

**RoPE — encoding recency, not just position**

Standard positional embeddings encode absolute slot number (position 1, 2, ..., 1000).
But what matters is **relative recency**: how far apart are two interactions?

The gap between positions 995 and 1000 (2 minutes apart, very recent) carries
different weight than positions 1 and 6 (days apart). Absolute position doesn't
capture this — both gaps look like "5 slots."

RoPE (Rotary Position Embeddings) encodes **relative position** directly into the
attention computation. Attention between two tokens naturally reflects their distance
in the sequence. The model learns temporal decay without explicit engineering.

---

**Late fusion — Pre-LN — RescaleAndAdd**

The sequential transformer handles temporal behavior. Static features (user
demographics, device, time of day) shouldn't be inside the sequence — they'd
pollute every attention computation.

Solution: **late fusion** — run transformer on sequence independently, then fuse
output with static context using RescaleAndAdd (LinkedIn GR design):

```
final_representation = u + α · v

  u = transformer output (sequential behavior)
  v = static context features (profile, device, session)
  α = learnable scalar — model decides how much to weight each
```

Pre-LayerNorm (norm before attention, not after) is used throughout for training
stability. Deep transformers with Post-LN have gradient flow issues. Pre-LN keeps
gradients clean at any depth.

---

**Shared Context Batching at serving — key system insight**

At serving time, 500 candidates share the **exact same user context** (same 1000
interactions). A naive implementation runs the full transformer 500× with the same
context — pure waste.

LinkedIn's approach (SRMIS kernel + Shared Context Batching):
```
Step 1: Compute user context representation ONCE
        N=512 candidates share a single context forward pass

Step 2: Each candidate attends to the shared context
        Context tokens: causal attention among themselves
        Candidate tokens: attend to ALL context tokens (no mask)

Result: ~80× serving speedup vs naive per-candidate inference
```

For video ranking: same design — encode the user's 1000-interaction history once
per request, reuse for all 500 candidates. The expensive part (attention over 1000
tokens) runs once, not 500 times.

---

**Full model input at the ranking layer**

```
Context (computed once per request):
  User profile features   → demographics, long-term preferences
  Past 1000 interactions  → (video_embedding, watch_time, completion,
                             action, timestamp) per step
  Session context         → device, time of day, current session length

Candidate (per video, 500 total):
  Video embedding         → from retrieval layer (LLM bi-encoder)
  Video metadata          → category, duration, creator, upload recency
  Engagement stats        → aggregate view count, completion rate, like rate

Transformer output → late fusion with static → MTL scoring heads
```

---

**What does the transformer actually produce for each candidate?**

The candidate embedding is not permanently modified — the transformer produces a new
contextual representation at serving time, specific to each user.

```
Input to transformer:

  Position:  1        2        3     ...   T        T+1
             ──────   ──────   ────        ──────   ──────────
             [X_1,    [X_2,    [X_3,  ...  [X_T,    [X_candidate]
              A_1]     A_2]     A_3]        A_T]

  X_t = video features at step t (content embedding, category, duration)
  A_t = action at step t         (watch_time, completion, liked, skipped)
  X_candidate = the video being scored (content embedding, metadata)
```

Attention runs across all positions. At position T+1 (the candidate), attention pulls
in signals from all previous T interactions. What comes OUT at T+1 is not the original
candidate embedding — it is a new vector that started as the candidate's features but
has user history blended in via attention:

```
Before transformer:   candidate token = "this is a cooking video"

Attention pulls in:   user watched 3 cooking videos recently
                      liked 2, skipped 1
                      those were short-form (< 5 min)
                      current session is 20 min long

After transformer:    candidate token = "this is a cooking video
                       AND this user has been on a cooking binge
                       AND they prefer short-form
                       AND session is long so appetite may be high"
```

That output vector at T+1 goes to the MTL heads to predict P(complete), P(like), etc.

**Same video, different output for different users:**

The cooking video has a fixed content embedding as input. After the transformer, its
output representation is different for every user — because every user has a different
T=1000 history attending into it:

```
User A (cooking binge this session):   candidate output → P(complete) high
User B (just finished cooking, seems done): candidate output → P(complete) lower
User C (never watches cooking):        candidate output → P(complete) low
```

Same video input. Three different output vectors. Three different scores. This is what
the transformer buys over a static dot product — the model dynamically asks "which
parts of this user's past are relevant to THIS video?" and that question has a
different answer for a cooking video vs a news video vs a comedy clip.

---

**Why late fusion — not injecting static features into the transformer sequence**

On top of the transformer output, static features (time of day, device, demographics,
session length) are added via late fusion:

```
final_representation = transformer_output(T+1)  +  α · static_features

  transformer_output(T+1) = candidate + user temporal history via attention
                           = "what is this user into RIGHT NOW"
  static_features          = time of day, device, demographics, session length
  α                        = learnable scalar
```

The reason for keeping static features out of the transformer sequence is
**specialization** — each stream does what it's best at:

- Transformer is good at learning temporal patterns across 1000 interactions.
  Injecting "time of day = evening" as a sequence token would pollute attention
  computations with a signal that has no position in the temporal chain.

- Static features capture context that has nothing to do with sequence order.
  "It's 11pm on a weekday" doesn't belong at position t=500 in the history —
  it's a property of the current moment.

Late fusion lets each stream specialize, then α decides how much to weight each.
If the user's recent history is a strong signal (active session, clear intent),
α weights temporal output higher. If history is sparse (new user, cold start),
α leans more on static profile — same design as LinkedIn GR's late fusion of
Member Profile Embedding for cold-start users.

---

**Multi-Task Model — MMoE (Multi-gate Mixture of Experts)**

The naive MTL design uses one shared MLP at the bottom, then splits into task heads.
The problem: tasks conflict. Gradient from P(complete) pulls shared weights one way;
gradient from P(skip) pulls them the opposite way. One shared representation forced
to serve opposing objectives → compromised representation for every task.

MMoE (used in LinkedIn GR) solves this with K=8 independent expert networks + a
per-task gating network that learns which experts to use.

**All 8 experts see the same input — the final_representation from late fusion.**
They each process it independently through their own MLP, producing 8 different
learned transformations of the same input:

```
final_representation  (same vector forwarded to all 8)
      ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓
     E1      E2      E3      E4      E5      E6      E7      E8
    MLP     MLP     MLP     MLP     MLP     MLP     MLP     MLP
      ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓
     h1      h2      h3      h4      h5      h6      h7      h8
           8 different learned transformations of the same input
```

Experts are not explicitly assigned roles — specialization emerges from training.
One might specialize in session-level patterns, another in content-type signals,
another in creator affinity. Each learns what it needs to serve the tasks routing to it.

**Each task has its own gate that picks a weighted mixture of the 8 experts:**

```
Gate_complete = softmax([0.4, 0.05, 0.3, 0.1, 0.05, 0.03, 0.05, 0.02])
                         E1          E3                        ← routes mostly to E1, E3

task_complete_repr = 0.4·h1 + 0.05·h2 + 0.3·h3 + ...        ← weighted sum of all 8

Gate_skip     = softmax([0.05, 0.5, 0.05, 0.02, 0.3, 0.05, 0.02, 0.01])
                               E2                E5            ← routes mostly to E2, E5

task_skip_repr = 0.05·h1 + 0.5·h2 + ...                      ← completely different mixture
```

Conflicting tasks route to different experts — no shared representation forced to
serve opposing objectives. Positively correlated tasks (P(complete) and P(like))
naturally learn to share the same experts.

**Task heads on top of each task's expert mixture:**

```
Task              Output    Loss                  Signal
──────────────────────────────────────────────────────────────
P(complete > 70%) sigmoid   binary cross-entropy  main quality signal
P(like)           sigmoid   binary cross-entropy  explicit engagement
P(share)          sigmoid   binary cross-entropy  strong positive signal
P(skip < 3s)      sigmoid   binary cross-entropy  negative quality signal
watch_time        linear    MSE                   direct business metric
```

**Training — joint loss, all tasks simultaneously:**

```
L_total = w1·L_complete + w2·L_like + w3·L_share + w4·L_skip + w5·L_watchtime

  Expert networks → receive gradients from ALL tasks (learn shared structure)
  Gates           → receive gradients from their own task only (learn task routing)
  Task heads      → receive gradients from their own task only
```

**Final ranking score at serving:**

```
score = w1×P(complete) + w2×P(like) + w3×P(share) - w4×P(skip) + w5×watch_time
```

Hand-tuned linear combination outside the model — easy to adjust weights without
retraining. Completion and watch time typically get higher weight than like/share.

**Position bias correction:**
Videos shown higher in the feed get more clicks regardless of quality. At training
time, inject position as a feature with a separate bias tower and zero it out at
serving (counterfactual: predict score as if position = 0).

### Stage 2.A — Ranking System Architecture and Optimizations

#### System Architecture

The ranking inference system is disaggregated into two independent components that
scale separately — a CPU-based Inference Driver and a fleet of GPU-based PyTorch
Inference Server processes.

```
                 ┌──────────────────────────────────────────────┐
                 │           Inference Driver  (CPU)             │
                 │                                               │
                 │  Responsibilities:                            │
                 │  1. Receive ranking request (500 candidate IDs│
                 │     + member ID)                              │
                 │  2. Fetch candidate features from feature     │
                 │     store (categorical, numeric, content      │
                 │     embedding — NOT bi-encoder embeddings)    │
                 │  3. Fetch member history features (past 1000  │
                 │     interactions) from key-value store        │
                 │  4. Join + transform features (Arrow buffers, │
                 │     zero-copy) → PyTorch tensors              │
                 │  5. Route request to a GPU worker process     │
                 │  6. Manage retry / hedge if worker is slow    │
                 │  7. Return ranked scores to upstream caller   │
                 └─────────────────────┬────────────────────────┘
                                       │ gRPC (Arrow → protobuf)
               ┌───────────────────────┼───────────────────────┐
               ▼                       ▼                       ▼
          Node A                    Node B                  Node C
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │ GPU0   GPU1  ... │    │ GPU0   GPU1  ... │    │ GPU0   GPU1  ... │
   │  P0     P1   ... │    │  P0     P1   ... │    │  P0     P1   ... │
   └──────────────────┘    └──────────────────┘    └──────────────────┘
   one Python process per GPU — each has own GIL, runs independently
```

**Why disaggregated:**
CPU work (feature fetching, joining, transformation) and GPU work (transformer
forward pass) have completely different resource bottlenecks. Separating them
lets each scale independently — add CPU nodes for feature-heavy workloads,
add GPU nodes for model compute.

**Why one process per GPU:**
Python GIL allows only one thread to execute Python bytecode at a time. Multiple
threads in one process cannot parallelize. Multiple processes each have their own
GIL — 8 GPUs on a node = 8 independent Python processes, no contention.

**Request flow (one request):**
```
1. Inference Driver receives: member_id + [candidate_id_1, ..., candidate_id_500]

2. Driver fetches in parallel:
     Feature Store → candidate features for all 500 IDs
                      (content embedding 50-dim, categorical, numeric engagement)
     KV Store      → member's past 1000 interaction history
   Note: NOT the bi-encoder retrieval embeddings — those served retrieval only.
   Ranking uses its own richer feature set fetched fresh per request.

3. Driver joins features → Arrow columnar buffers → zero-copy to PyTorch tensors
   (avoids serialization cost for large feature payloads)

4. Driver picks GPU worker process from connection pool → sends via gRPC

5. GPU worker runs:
     a. Encode member history → context KV (computed ONCE, stored in HBM)
     b. Batch all 500 candidates → single forward pass attending to shared context KV
     c. MMoE heads → P(complete), P(like), P(share), P(skip), watch_time per candidate
     d. Compute final score per candidate → return ranked list

6. Driver returns ranked list to upstream caller
```

---

#### System Optimizations

**Optimization 1 — Shared Context KV Cache (context computed once)**

The member's 1000-interaction history is identical for all 500 candidates in a
request. Computing it 500 times is pure waste.

```
Naive:   500 × forward_pass(1000 context tokens + candidate tokens)
         = 500 × 1010 token-steps = 505,000 token-steps

Shared:  1 × forward_pass(1000 context tokens) → KV tensors stored in HBM
         500 × forward_pass(candidate tokens only, attend to cached KV)
         = 1000 + 500 × 10 = 6,000 token-steps
```

Context tokens attend causally among themselves. Each candidate token attends to
all context tokens (no causal mask on the candidate side). This is what LinkedIn's
SRMIS custom Flash Attention kernel implements — ~80× speedup reported in paper.

---

**Optimization 2 — Candidate Batching (score 512 in one forward pass)**

After computing shared context KV, all 500 candidates are scored in one batched
forward pass — not one at a time.

GPU is a matrix processor. Scoring 1 candidate = tiny matmul, SMs underutilized,
kernel launch overhead dominates compute time. Scoring 512 candidates in one
batched matmul = near-peak GPU utilization, kernel launch amortized across the
full batch. LinkedIn uses N=512 as the Shared Context Batching size.

---

**Optimization 3 — Feature fetching, not bi-encoder embeddings**

Ranking does NOT reuse the retrieval bi-encoder embedding. Retrieval gives top K
candidate IDs. The Inference Driver fetches those candidates' rich features from
the feature store:

```
Sequence features per candidate (X_seq):
  content embedding    → 50-dim, from a separate post embedding model
  ID embeddings        → actor/root-actor hashed IDs, learned in-model
  categorical features → post type, device OS, object type
  numeric features     → engagement counts, affinity scores (log-transformed)

Context features (late-fused after transformer):
  viewer-actor affinity, candidate popularity, bucketed dwell-time, post age
```

The bi-encoder embedding is optimized for ANN dot-product similarity over 100M
candidates. Ranking uses a richer, separate feature set — its content embedding
is 50-dim from a different model, one of many concatenated features per position.

---

**Optimization 4 — Multi-process / CPU-GPU affinity (avoid Python GIL)**

Each GPU worker is a separate Python process bound to one GPU. The Inference
Driver is a separate CPU process entirely. They run independently:

```
Inference Driver (CPU process):
  feature fetching + joining overlaps with GPU compute on previous request
  no GIL dependency on GPU workers

GPU Worker P0 (GPU 0):  handles its batch, own GIL, own memory space
GPU Worker P1 (GPU 1):  handles its batch, own GIL, own memory space
...
```

One slow GPU worker does not block any other worker or the Inference Driver.
CPU preprocessing pipelines with GPU execution across the fleet.

---

**Optimization 5 — Bucketing + Padding → CUDA Graphs**

Variable-length inputs (different history lengths per member, variable candidate
batch sizes) prevent CUDA graph capture — graphs require fixed tensor shapes.

Solution: bucket inputs into fixed shapes:
```
Context length buckets:  {128, 256, 512, 1024}  → pad to nearest
Candidate batch buckets: {128, 256, 512}         → pad to nearest
```

Pre-capture one CUDA graph per (context_bucket, candidate_bucket) combination
at warmup. Each subsequent request matching that shape = graph replay:
- No kernel launch overhead (CPU-GPU sync eliminated per op)
- Entire forward pass captured as one graph → 20-30% latency reduction
- Padding cost is offset by elimination of per-kernel overhead

---

**Optimization 6 — Two-level Caching**

**Level 1 — Context KV cache (short TTL):**
```
Key:   hash(member_id + session_hash)
Value: KV tensors for member's 1000-interaction context

If same member makes two requests in the same session (e.g. paginating feed):
  → skip context encoding entirely, reuse cached KV tensors from HBM
TTL: 1-2 minutes — invalidate when new interaction arrives
```

**Level 2 — Candidate feature cache:**
```
Key:   candidate_id
Value: precomputed feature vector (content embedding + categorical + numeric)

Video features are stable between daily model refreshes.
Driver looks up feature store (Redis/Cassandra) — already a cache layer.
TTL: aligned with feature refresh cadence (hours to days)
```

Full score caching (member × candidate) is impractical here — ranking score
depends on the member's full interaction history, which changes with every watch.
Unlike semantic search (query + candidate = stable), ranking scores are
session-dependent.

---

**Optimization 7 — Latency Budget Retry (Hedged Requests)**

The Inference Driver manages retry — GPU worker processes do not watch each other.

```
t=0ms:   Driver sends request → GPU Worker A (Node A / GPU0 / P0)
          Driver starts timer at p90 of healthy ranking latency

t=50ms:  Worker A has not responded → timer fires
          Driver sends SAME request → GPU Worker B (Node B / GPU0 / P0)
          Both workers running in parallel (hedged)

t=80ms:  Worker B responds first
          Driver returns Worker B result to caller
          Driver cancels outstanding gRPC call to Worker A

t=110ms: Worker A finishes → result discarded
```

Retry threshold T = p90 of healthy latency so normal traffic never triggers a
hedge. The Driver tracks response time per worker in its connection pool —
consistently slow workers receive fewer new requests (weighted load balancing).

---

**Optimization 8 — PID-based Dynamic Scoring Depth**

Under peak load, the Inference Driver reduces the number of candidates passed
to the GPU workers — fewer candidates per request = less GPU compute per request.

```
Off-peak:  L1 lightweight ranker passes 250 candidates to L2 heavy ranker
Peak load: L1 passes 130 candidates → 48% less GPU compute per request

PID controller (in Inference Driver):
  Error signal:   current GPU worker queue depth vs target queue depth
  Control signal: L1 pass-through threshold (how many reach L2)
  Feedback loop:  lower threshold → less GPU load → queue depth drops → error shrinks
```

PID reacts continuously — no abrupt step changes that cause oscillation. The
system self-stabilizes under arbitrary load patterns without manual re-tuning.

---

#### Observability — Distributed Tracing

Every request through the ranking stack is instrumented end-to-end so latency
spikes can be isolated to the exact layer that caused them — not guessed at.

```
Request enters Driver
   │
   ├─ [span: feature_fetch]       → p99 latency of Feature Store calls
   │    ├─ candidate features
   │    └─ member history
   │
   ├─ [span: feature_join]        → Arrow buffer assembly, zero-copy to tensors
   │
   ├─ [span: queue_wait]          → time spent waiting for a GPU worker to be free
   │
   ├─ [span: gpu_forward_pass]    → transformer + MMoE forward pass on GPU worker
   │    ├─ context_kv_encode      → member history → KV (computed once)
   │    └─ candidate_score_batch  → 500 candidates in one forward pass
   │
   └─ [span: postprocess]         → score assembly, ranked list construction
```

**How this maps to the 3x spike debug process:**

```
Spike in feature_fetch span  → Feature Store issue or more candidates per request
Spike in queue_wait span     → Traffic increase, need more GPU workers
Spike in gpu_forward_pass    → Input size grew (longer sequences → O(n²) attention)
                               OR new model deployed with larger architecture
                               OR GPU memory pressure from competing workloads
Spike in context_kv_encode   → Member history token count grew (check avg seq length)
```

Distributed tracing turns a 3x total latency spike from a mystery into a
one-lookup root cause — check the trace, the broken span is immediately visible.

---

### Stage 3 — Re-ranking and Diversity

Raw ranking maximizes individual video scores. Without diversity, a user who liked
one cooking video gets 20 cooking videos → repetitive, damages session quality.

```
Diversity rules:
  Max 2 consecutive videos from same creator
  Max 3 videos in same category before a different category
  Interleave explore candidates (forced novel content every ~5 videos)

Freshness boost:
  Multiply score by recency factor: score × (1 + α × freshness_score)
  Prevents feed being dominated by old viral videos

Hard filters:
  Remove already-watched (in last 30 days)
  Apply content safety scores (remove flagged content)
```

---

## 5. Cold Start

### New Video Cold Start

A video uploaded 5 minutes ago has zero engagement stats. The two-tower video
embedding is based on video content features only (no engagement history yet).

```
At upload time:
  1. Run video encoder → content embedding (visual + audio + text)
  2. Assign to a topic bucket based on content embedding
  3. Show to a small exploration pool (~0.1% of matching users)
  4. Collect initial engagement signal (15-30 min)
  5. If engagement rate > threshold → graduate to broader distribution
  6. Engagement stats now available → ranker uses real features

This is the "cold start evaluation window" — every new video gets a fair trial.
```

**Why this is critical for creator ecosystem:** without cold start handling, only
established creators with high follower counts get shown. New creators never break
through. Cold start exploration budget is essentially a platform investment in diversity.

### New User Cold Start

No history → no user embedding → two-tower can't personalize.

```
Fallbacks in order:
  1. Trending feed (regional + language-matched trending videos)
  2. Onboarding survey: ask user to select 3-5 interest categories
  3. Demographic-based: age/region cohort model
  4. After 5-10 interactions: fast-adapt using session context only

Fast adaptation: session embeddings update in real-time from interactions.
After 3 videos watched or skipped, the recommendation is already personalized.
```

---

## 6. Online Embedding Learning — Rapid Preference Adaptation

This is what separates TikTok from older recommendation systems.

**The problem with batch retraining:**
```
Weekly retraining cycle:
  Monday: model sees user's behavior from last week
  Friday: user discovers new interest (home improvement)
  Next Monday: model finally learns about this interest
  → 6 days of mismatched recommendations
```

**Online embedding learning — freeze weights, update embeddings:**

```
Model weights:    retrained weekly (slow, requires labeled data, GPU-expensive)
User embeddings:  updated continuously from every interaction (fast, cheap)

Every engagement event:
  User watches 90% of a cooking video
    → engagement event published to Kafka
    → nearline pipeline consumes event
    → runs frozen embedding update (lightweight gradient step on embedding layer only)
    → writes updated user_embedding to embedding store
    → next feed request uses fresh embedding

Adaptation lag: minutes, not days
```

**Why freeze model weights but update embeddings?**
- Full model retraining needs millions of labeled examples for stable gradients
- Embedding updates are a single vector change — stable with even one example
- Risk of catastrophic forgetting applies to weights (complex features), not to
  a single user's embedding vector (which represents only that user's state)

**Contrast with LinkedIn feed (same pattern, different domain):**

| | LinkedIn Feed / Search | TikTok / Reels |
|---|---|---|
| What's updated | Item embeddings (posts, profiles) | User embeddings (preference state) |
| Trigger | CDC — content change (profile edit, new post) | Engagement event — behavior change |
| Why frequent | Items change: new posts, profile updates | Users change: fast-shifting interests |
| Encoder frozen | Yes — bi-encoder weights fixed | Yes — two-tower weights fixed |

Both systems decouple model retraining cadence (slow) from representation freshness
cadence (fast). The mechanism is identical; what triggers the update differs.

---

## 7. Exploration vs Exploitation

Pure exploitation (always recommend highest-scored videos) creates filter bubbles.
Users only see content similar to what they've already liked. Discovery stops.

```
ε-greedy:
  With probability ε: show a random/novel video (explore)
  With probability 1-ε: show highest-scored video (exploit)
  Simple but wastes explore budget on truly random videos

Thompson Sampling / UCB:
  Maintain uncertainty estimate per video (or per topic bucket)
  Score = expected reward + exploration bonus (proportional to uncertainty)
  Naturally explores underexposed content without hard randomness

Multi-armed bandit on topic buckets:
  Treat each topic as an arm
  Reward = engagement signal
  Bandit allocates more feed slots to high-performing topics
  Occasional forced exploration of low-exposure topics
```

**Why exploration matters for the creator ecosystem:**
Without exploration, only already-popular creators get shown → rich-get-richer.
Exploration budget creates opportunity for new creators. Platform health depends on
a diverse creator base — single-creator dependency is a business risk.

---

## 8. Training

### Data Collection
```
Positive: (user, video, context) where watch completion > 70%
Negative: (user, video, context) where skip < 10% or explicit "not interested"
Hard negatives: videos shown but not clicked (harder to learn from than random)
```

### Loss Function
Multi-task cross-entropy with task weights:
```
L = w_complete × BCE(P(complete), y_complete)
  + w_like     × BCE(P(like), y_like)
  + w_share    × BCE(P(share), y_share)
  - w_skip     × BCE(P(skip), y_skip)   ← note: negative weight
```

Task weights tuned to align with business metric (watch time, D7 retention).

### Training Cadence
```
Two-tower retrieval model:  weekly retraining (structural features change slowly)
Ranking model:              daily retraining (engagement stats change faster)
User embeddings:            continuous (minutes, via online embedding update)
Video embeddings:           on-demand (when video content or stats change)
```

---

## 9. Evaluation

### Offline Metrics
```
Retrieval:   Recall@1000 — did the watched video appear in candidates?
Ranking:     AUC, NDCG@10 on held-out engagement labels
Watch time:  predicted vs actual watch time correlation
```

### Online Metrics (A/B test)
```
Primary:    Watch time per session, Session length
Secondary:  D7 / D30 retention
Guardrails: Harmful content rate, Creator diversity (Gini coefficient of views)
```

**Why A/B test is necessary:** offline metrics don't capture feedback loops.
A model with slightly lower offline AUC can drive better long-term retention
because it explores more — which offline evaluation can't measure.

---

## 10. Interview Summary

**Q: Why multi-task ranking instead of single objective?**
Users leave signals across multiple dimensions — completion, likes, shares, skips.
A video that completes but is never shared has different value than one that's
both completed and shared. Multi-task captures the full engagement picture and
lets you tune the business tradeoff explicitly via task weights.

**Q: How does TikTok adapt so fast to new user interests?**
Online embedding learning: model weights are frozen between weekly retraining cycles,
but user embeddings update continuously from every engagement event via a nearline
pipeline. Adaptation lag is minutes, not days.

**Q: How do you handle new videos with no engagement history?**
Cold start evaluation window: show to a small exploration pool (~0.1% of users),
collect 15-30 min of engagement signal, graduate if engagement exceeds threshold.
Uses content features (visual/audio embedding) as proxy until real engagement data arrives.

**Q: Why is exploration important beyond just model quality?**
Two reasons: (1) filter bubbles — pure exploitation stops discovery, reduces session
diversity, damages long-term retention; (2) creator ecosystem — without exploration
budget, new creators never break through, rich-get-richer collapses creator diversity
which is a platform health risk.

**Q: How is this different from LinkedIn feed/search recommendations?**
Same two-tower + ranking architecture. Key difference: what triggers embedding refresh.
LinkedIn refreshes item embeddings when content changes (CDC-driven). TikTok refreshes
user embeddings when behavior changes (engagement-driven). Both decouple model retraining
(slow) from embedding freshness (fast).

**Q: You see a 3x latency spike in your inference service. How do you debug it?**

First isolate which layer added the latency using distributed tracing — don't guess.

```
Total latency = feature fetch time (Driver)
              + queue wait time (Driver → GPU worker)
              + GPU forward pass time
              + postprocessing

3x total spike doesn't tell you where. Trace tells you which term blew up.
```

Then drill down by layer:

**Spike in feature fetch:**
- Feature store (Venice/Redis) slow — check its p99 independently
- More candidates per request → more feature lookups
- Fix: local cache on Driver, increase feature store capacity

**Spike in queue wait time:**
- Traffic increased → queue backing up → requests wait before a GPU worker picks them up
- Check: queue depth over time, request rate vs worker count
- Fix: scale out GPU workers

**Spike in GPU forward pass:**
- Input size changed — user prompt got longer (more watch history tokens)
  Transformer is O(n²) in sequence length → small input growth = large latency growth
- New model deployed with larger architecture — correlate spike with deployment timestamp
- GPU memory pressure — multiple models on same GPU competing for HBM
- Fix (input size): enforce max sequence length, truncate oldest history tokens
- Fix (deployment): rollback immediately, profile new model separately

**Spike in exhaustive KNN:**
- Video corpus grew significantly → more embeddings to scan
- 1-bit quantization pipeline failed → silently fell back to FP16 full scan (100× more data)
- Fix: monitor quantization pipeline health, alert on index size growth

**Mitigation order: mitigate first, root cause second.**
- Sudden spike after deployment → rollback
- Sudden spike from traffic event → scale out GPU workers
- Gradual spike → input size growth or corpus growth, fix upstream
