RecSys: Retrieval Challenges and GPU RaRRetrieval System — Bottlenecks and Improvements

Reference: LiNR (LinkedIn Neural Retrieval on GPUs, CIKM 2024) + LinkedIn Engineering Blog
Book: Transformers in Action: Ch 3


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. What Retrieval Is Doing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In a two-stage recommendation system:

  Stage 1 — Retrieval (Candidate Generation):
    Input:  member query (embedding)
    Index:  millions to billions of items (posts, jobs, people)
    Output: top-K candidates (K = 500 to 2000)
    Goal:   high RECALL — don't miss relevant items, even if you include some noise

  Stage 2 — Ranking:
    Input:  K candidates from retrieval
    Output: ordered top-N for the member
    Goal:   high PRECISION — the items actually shown should be relevant

Retrieval is a coarse filter. Ranking is a fine filter.
If retrieval misses a relevant item, ranking can never surface it.
This is why recall at the retrieval stage is more important than precision.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Retrieval Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Precision@K
  Of the K items retrieved, how many are actually relevant?
  Precision@K = (relevant items in top-K) / K
  Example: retrieved 100, 30 relevant → Precision@100 = 0.30

Recall@K
  Of all relevant items in the corpus, how many did retrieval find?
  Recall@K = (relevant items in top-K) / (total relevant items in corpus)
  Example: 50 relevant items exist, retrieval found 40 of them → Recall@K = 0.80

Hit Rate@K (used in LiNR)
  Binary per query: did at least one relevant item appear in top-K?
  Hit Rate@K = (queries with ≥1 relevant item in top-K) / (total queries)
  Easier to compute at scale than full Recall@K.
  LiNR reports Hit Rate@400 as the offline eval metric.

MRR — Mean Reciprocal Rank
  How high up is the first relevant item?
  MRR = average of (1 / rank_of_first_relevant_item) across queries
  MRR=1.0 → relevant item is always rank 1
  MRR=0.5 → relevant item is on average rank 2

NDCG@K — Normalized Discounted Cumulative Gain
  Measures quality of the ORDERING, not just presence.
  Items ranked higher get more credit. Graded relevance (0/1/2) supported.
  Common in search and ranking evaluation, less common in retrieval candidate gen.

Precision vs Recall Tradeoff:
  Increase K → higher recall (cast wider net) → lower precision (more noise)
  Decrease K → higher precision → lower recall (may miss relevant items)
  In retrieval: always prefer higher K (more candidates to ranker) up to latency budget.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Problem: Retrieval Quality Is Low (Wrong Items Retrieved)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Root causes:
  (a) Weak encoder — embeddings don't capture semantic meaning
  (b) Poor similarity function — dot product too simplistic
  (c) Liquidity — filtered retrieval loses valid candidates (details in Section 4)


Solution A — Replace Traditional Encoder with LLM Encoder

  Traditional two-tower encoder: trained from scratch on engagement data only.
  Three fundamental limitations:

  Limitation 1 — Only knows what users co-clicked:
    Knows: "software engineer" <-> "backend developer" (users co-clicked)
    Doesn't know: "software engineer in San Francisco" should also match "Bay Area" roles
    -> Sparse engagement data -> missing geographic and semantic equivalences

  Limitation 2 — Vocabulary mismatch:
    Member query:   "software engineer job in San Francisco"
    Available job:  "SWE role in Bay Area, California"
    Traditional: different tokens -> low similarity -> not retrieved
    LLM encoder: same meaning -> high similarity -> retrieved

  Limitation 3 — Cold start:
    New user, zero engagement history → no embedding → falls back to popularity
    LLM encoder: profile text alone → rich embedding from world knowledge

  LLM encoder fixes all three because it pre-trained on internet-scale text:
  it already knows which concepts belong together, regardless of whether
  LinkedIn users have clicked on them together.


Solution B — Better Similarity Function: Mixture of Logits (MoL)

Why Plain Dot Product Falls Short
  
  A single embedding vector tries to compress everything about a member — interest in ML, career advice, industry news — into one direction in space. When you compute a dot
  product with a post embedding, you get one blended score. This is a rank-1 approximation: one perspective, one judge. It can't capture that the same member cares about
  completely different things in different contexts.

  MoL — A Panel of Specialized Judges

  Instead of one score, MoL computes multiple similarity scores from different components of the embeddings, then combines them with learned weights that adapt per (member,
  post) pair.

  Think of it as a panel of judges:
  - Judge 1 (two-tower embedding): "Based on past engagement, does this member click on posts like this?"
  - Judge 2 (GNN embedding): "Are people in this member's network engaging with this post?"
  - Judge 3 (cluster embedding): "Does this post belong to a topic cluster this member cares about?"

  Each judge gives a score (δ_k). The gating function (π_k) dynamically decides how much to trust each judge for this specific member-post pair. A network-driven member → Judge
  2 gets more weight. A topic-driven member → Judge 3 gets more weight. The weights are not fixed — they are learned and input-dependent.

  Why This Beats Cosine Similarity

  Cosine similarity asks: "are these two vectors pointing in the same direction?"
  MoL asks: "across multiple ways of measuring relevance, how well does this post match — and which measure matters most for THIS specific pair?"

  The +23.67% hit rate gain (MoL 300 clusters vs cosine baseline) comes entirely from this adaptive, multi-perspective scoring.

  Why Hadamard MLP Was Chosen for Production Over MoL

  MoL is more accurate but slower — at LinkedIn's scale you're computing multiple scores plus a gating function per candidate pair. Hadamard MLP is the middle ground:
  element-wise multiply the member and post embeddings, then pass through an MLP. Richer than dot product (captures cross-dimension interactions), but still a single forward
  pass — production-safe latency. The tradeoff is accuracy vs inference cost; +10.21% hit rate is good enough when MoL's latency is too high.

Mathematics of MoL:

  Plain dot product: one embedding pair, rank-1 approximation.
  MoL: weighted combination of multiple embedding components.

    φ_MoL(member, post) = Σ_k  π_k(member, post) × δ_k(member, post)

    π_k = gating function (softmax) — learned per (member, post) pair
    δ_k = similarity from the k-th embedding component

  Embedding components used in LiNR:
    - Two-tower embedding     (standard engagement-trained EBR)
    - GNN embedding           (heterogeneous graph, member+post in same space)
    - Cluster ID embedding    (K-means centroids of post embeddings)

  Results (Hit Rate@400, offline):
    Cosine similarity baseline:                  —
    Hadamard MLP (single embedding):        +10.21%
    MoL, 100 non-trained clusters (single): +15.16%
    MoL, 300 non-trained clusters (multi):  +23.67%

  Hadamard MLP — simpler alternative chosen for production:
    member_emb → MLP → h_m
    post_emb   → MLP → h_p
    output     = MLP(h_m ⊙ h_p)   ← element-wise (Hadamard) product
    Lower latency than MoL, sensitive to weight init (restart if first 100 steps diverge).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Problem: Filtered Retrieval Has Low Recall (Liquidity Issue)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Context: Many retrieval systems need attribute-based filters.
  Job recommendations: filter by geo, company, job title, skills
  Feed OON: filter by language, topic, member connections

The standard approach: POST-FILTERING
  1. Retrieve top-K by embedding similarity (ANN or exhaustive)
  2. Apply attribute filter on the K results

The problem: liquidity
  15M posts in index
  Filter: geo = "San Francisco" → only 200K posts qualify
  ANN retrieves top-2000 globally (ignoring geo) → may return 0 SF posts
  Post-filter discards 95%+ of candidates → you get far fewer than K results
  → member sees an empty or low-quality feed

Why ANN makes this worse: HNSW/IVFPQ clusters the embedding space by similarity,
not by attributes. A geo filter cuts across cluster boundaries — ANN never
considers most SF posts because they're in different clusters.


Solution — PRE-FILTERING with Attribute-Based Matching (ABM)

  Filter FIRST, then compute similarity only on qualifying items.
  Guarantees K results from the correct population.

  LiNR has two implementations:



  V1 — Similarity Masking (better for high pass rate):
    1. Compute similarity: query × ALL 15M embeddings (full matmul)
    2. Build 0/1 mask from attribute clauses (CUDA kernel)
    3. Multiply scores by mask → non-qualifying items become 0
    4. Top-K on masked scores
    Cost: always runs full matmul. Filtering is cheap (bitwise).
    Best when: most items pass the filter (high pass rate → matrix slice overhead not worth it)



  
V2 — Explicit Pre-Filtering (better for low pass rate):
    1. Check attribute clauses → get set of qualifying item IDs
    2. Slice embedding matrix to qualifying items only (e.g., 200K of 15M)
    3. Compute query × 200K embeddings only
    4. Top-K on 200K scores
    Cost: matrix slice + small matmul.
    Best when: very few items pass filter → slice overhead < matmul savings

  Latency benchmarks (15.5M job index, 128-dim FP16 embeddings):
    High pass rate (1.7M pass): V1 = 4.8ms, V2 = 14.6ms  → V1 wins
    Low pass rate (thousands):  V2 = 1.9ms, V1 not reported → V2 wins

  Key: requires custom CUDA kernels for clause checking — vanilla PyTorch slice
  introduces extra copy overhead that makes V2 slower in high-pass-rate cases.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4a. How ABM Attribute Checking Works on GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The clause check (step 2 in V1, step 1 in V2) is not a hash lookup.
It runs a custom CUDA kernel on two packed data structures in GPU memory.

Mental model clarification:
  ANN + post-filtering  → liquidity issue (ANN misses qualifying items in other clusters)
  KNN + post-filtering  → liquidity issue (top-K by similarity may not contain valid items)
  KNN + ABM pre-filter  → no liquidity issue (every item in top-K is valid by construction)

  Why ANN breaks pre-filtering:
    HNSW/IVFPQ clusters the embedding space by semantic similarity.
    A geo filter (SF only) cuts across cluster boundaries — most SF items live
    in many different clusters. ANN never visits those clusters → misses qualifying items.
    Exhaustive KNN scans everything → no items skipped → pre-filtering is clean.

Two data structures stored in GPU memory:

  Counting Matrix  [n_items x n_clauses]:
    count[i][j] = number of attribute values item i has for clause j
    Example (3 clauses: location, job_type, company):
      J0: [1, 1, 1]   1 location, 1 job_type, 1 company
      J1: [1, 1, 1]
      J2: [2, 2, 1]   2 locations (NYC+SF), 2 job_types (contract+FT), 1 company
      J3: [1, 1, 2]   1 location, 1 job_type, 2 companies (Google+Meta)

  Single Attribute Matrix  [n_items x max_attrs_per_item]:
    Each row = item's attributes, all clauses concatenated, sorted within each clause.
    Rows are variable length — max_attrs_per_item pads to the longest row.
    Example row for J2 (sorted within each clause):
      [ NYC, SF  |  contract, full-time  |  Meta ]
         loc(2)       job_type(2)          company(1)

    Why sorted? Early exit: once attr > query value (integer encoding), stop scanning.
    Paper encodes all attributes as 64-bit integers before storage for fast comparison.

  CUDA Kernel — one GPU thread per item:
    For each clause c:
      offset = sum of count[i][0..c-1]   (where this clause's attrs start in row)
      scan count[i][c] attributes starting at offset
      normal clause:  pass if ANY attribute matches query attribute (OR semantics)
      reverse clause: pass if NO attribute matches  (e.g., exclude blacklisted company)
      early exit: stop scanning once sorted attr > query value
    If all clauses pass → mask[i] = 1, else mask[i] = 0

  Why not an inverted index?
    Hash lookups cause warp divergence on GPU — threads in the same warp
    take different paths → serialized execution → kills throughput.
    The packed matrix + CUDA scan is fully vectorized: all N items checked in parallel,
    same code path per thread, coalesced memory access.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Problem: Index Too Large to Fit on GPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GPU memory is the bottleneck for exhaustive KNN.

  1B posts × dim=64 × FP16 (2 bytes) = 128GB → doesn't fit on A100 (80GB)

  ANN (FAISS) sidesteps this via compression (IVFPQ) but loses pre-filtering ability.
  LiNR wants exhaustive search with pre-filtering → needs a different approach.


Solution A — 1-Bit Quantization via Sign-OPORP (V3 Quantized KNN)

  Sign-OPORP: Sign + One Permutation One Random Projection

  Step 1: Random projection to lower dimension (count-sketch variant — OPORP)
  Step 2: Take sign of each projected value → 1 bit per dimension
  Step 3: Pack bits into 64-bit integers → 256-bit vector per item (for dim=256)

  Memory reduction:
    FP16, dim=64:  128 bytes per item
    1-bit, dim=64: 8 bytes per item → 16× reduction

    1B items FP16:  1B × 128 = 120GB  → doesn't fit on V100 (32GB)
    1B items 1-bit: 1B × 8   = 7.5GB  → fits on single V100

  Similarity: bitwise XOR between query and item 1-bit vectors
    count of matched bits ≈ cosine similarity of original FP16 embeddings
    Bitwise ops are much faster than FP16 matmul

  Two ways to use V3:
    Mode 1: Replace full-precision matmul entirely (when K is large, memory is primary constraint)

 Mode 2: Two-stage pre-filter (when you want accuracy + speed):
      Stage 1: 1-bit XOR across all items → keep top 1% (200K candidates)
      Stage 2: FP16 matmul on 200K survivors → accurate final scores
      Result: ~10% latency improvement, near-parity recall vs full FP16

  Query arrives (FP16)
         │
         ▼
    Sign() → 1-bit query
         │
         ▼
    XOR + popcount vs ALL 20M items   ← cheap, 1-bit ops
         │
         ▼
    Sort by matching-bit-count, keep top 1% (200K)
         │
         ▼
    Load FP16 embeddings for only those 200K items
         │
         ▼
    Full FP16 dot product on 200K     ← accurate, but 100× smaller set
         │
         ▼
    Final ranked top-K

The 1-bit stage is a coarse filter — it's not precise enough to trust for final ranking, but it's accurate enough to not miss items that would've been in the true top-K. The FP16 stage on 200K gives you the same quality as scanning all 20M in FP16, but you only paid 1% of the compute.

  You can adjust 1-bit embedding dimension to tune memory vs accuracy tradeoff.


Solution B — Separate Storage and Compute Precision

  Storage precision and compute precision are independent choices.
  See Quantization doc for full details.

  Pattern: store 1-bit, compute FP16 on survivors (LiNR V3)
  Pattern: store INT8, compute INT32 (hardware accumulator auto-widens)
  Pattern: store FP16, compute FP8 (H100 native)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Problem: Stale Results (Freshness)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Offline batch pipeline: post embeddings recomputed every few hours or daily.
Freshly published posts are not in the index → never retrieved → member misses
the most timely, relevant content.

LiNR showed: +6% improvement from freshness alone (live index vs batch-rebuilt).


Solution A — Nearline Embedding Pipeline (Three Pipelines)

  Traditional: offline batch job → reindex every N hours
  LiNR: three chained nearline pipelines, triggered on post publish:

    Pipeline 1 — Prompt Generation:
      Event: new post published
      Action: generate LLM prompt from post metadata + content
      Output: prompt string → pushed to embedding queue

    Pipeline 2 — Embedding Generation:
      Input: prompt from Pipeline 1
      Action: run LLM encoder forward pass
      Output: post embedding → written to Venice Store (same store as offline)
      Latency: seconds after publish

    Pipeline 3 — GPU Index Update:
      Input: new embedding from Venice CDC stream
      Action: upsert embedding into live GPU tensor (no index rebuild)
      Output: post is now retrievable

  Key: offline and nearline write to the SAME Venice Store.
  Offline handles the bulk; nearline handles the delta.


Solution B — Live Updates via Change Data Capture (CDC)

  Venice Store exposes a CDC stream: every write/delete emits an event.
  LiNR's Updator subscribes to this stream:
    Upsert event → copy new embedding onto pre-allocated GPU tensor
    Delete event → mark slot as inactive (high-water mark tracking)

  Implementation details that make this work at inference time:
    Pre-allocated tensors: GPU tensors sized larger than current index (room to grow)
    High-water mark: tracks live working set, inference only scans up to this mark
    Thread-safe updates: tensor updates serialized minimally, inference path unblocked
    Benchmarked: 0 measurable latency impact at 600 updates/sec

  Bootstrapping at startup:
    Replay entire Venice CDC stream from offset 0 → rebuild full index
    Compacted snapshots stored on disk → fast warm start avoids full replay



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. Problem: Cold Start (New Users / New Content)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

New user: no engagement history → no signals → embedding is random or zero
New post: no interaction data → can't learn from clicks that don't exist yet

Traditional fallback: serve popular content (popularity bias → bad personalization)


Solution A — Cluster ID Embeddings

  K-means clustering over millions of post embeddings → C cluster centroids.
  Each cluster centroid = a "topic" (e.g., AI/ML, energy, finance).

  For a new member:
    Map profile text → cluster centroid via cosine similarity (no clicks needed)
    That cluster ID embedding feeds into MoL as one component
    → retrieval finds posts near that centroid

  Key finding from LiNR: fixed (non-trainable) cluster centroids outperform
  trainable clusters in all experiments.
  Hypothesis: cluster and model parameters converge at different rates → instability.
  Practical: run K-means once on post embeddings, freeze, use as lookup.

  Number of clusters matters: too few → coarse topics, misses nuance.
                              too many → sparse, cold start cluster itself has no posts.
  LiNR found optimal around 100-300 clusters (tuned per use case).


Solution B — LLM Encoder (Profile Text → Rich Embedding on Day 1)

  "Senior Power Systems Engineer, skills: Python, SCADA, Grid Optimization"
  → LLM encoder produces embedding already close to: grid, nuclear, renewables posts
  → No engagement history needed

  LLM encoder has pre-trained world knowledge: professional titles, skills, domains
  map directly to semantic embedding space without any clicks.

  Combined: LLM encoder for member embedding + cluster ID → MoL.
  Cold start member gets reasonable retrieval from day 1.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. Problem: High Latency
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Retrieval latency budget is tight: typically 10-50ms end-to-end.
Exhaustive GPU matmul over 15M items with 128-dim FP16 embeddings = 4-7ms on A100.
That's actually fast — but for 1B items it becomes prohibitive.


Solution A — V1 vs V2 Based on Pass Rate

  Measure the pass rate of your attribute filter in production.
  High pass rate (>10% of items pass): use V1 (similarity masking)
  Low pass rate (<1% of items pass):  use V2 (explicit pre-filtering)
  Route per query based on estimated pass rate (can compute from filter attributes).


Solution B — Quantized Pre-Filter Stage (V3)

  Add 1-bit XOR stage before full-precision matmul:
  1-bit XOR over all items → keep top p% → FP16 matmul on p% only
  p=1% → ~10% latency reduction with near-parity recall.
  p tradeoff tunable: lower p = faster but lower recall.


Solution C — ANN as Fallback (Accept Liquidity Tradeoff)

  If latency is the absolute constraint and filters are loose:
  FAISS IVFPQ / HNSW — sublinear search time.
  Accept: no clean pre-filtering, some liquidity loss, no live updates.
  Use when: corpus is static, filters are rare or loose, latency < 2ms required.

  LiNR baseline: FAISS-IVFPQ was the comparison point. LiNR (exhaustive) beat it
  on quality (Hit Rate) while still meeting latency requirements on A100.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. The Universal Pattern: Coarse-to-Fine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every solution above follows the same principle: use cheap approximate methods
to narrow the candidate set, then expensive accurate methods on the small set.

  LiNR V3:
    1-bit XOR over 1B items (cheap) → 200K candidates
    FP16 matmul over 200K (expensive) → top-K final

  Feed pipeline:
    Retrieval over 15M (cheap, embedding matmul) → 2000 candidates
    L1 ranker over 2000 (medium, lightweight model) → 200 candidates
    L2 ranker over 200 (expensive, full neural model) → top-20 shown

  Speculative decoding (LLM inference):
    Small draft model generates tokens (cheap) → candidates
    Large model verifies (expensive) → accept or reject

  ANN search:
    Quantized index for fast cluster search → candidate list
    Full-precision reranking on candidate list → final results

Core principle: quantization / approximation doesn't have to be all-or-nothing.
Use coarser precision to narrow the search space, finer precision for final decision.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. Summary — Interview Quick Reference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Bottleneck                          Diagnosis                   Solution
  ──────────────────────────────────────────────────────────────────────────────────
  Wrong items retrieved               Weak encoder                LLM encoder (world knowledge)
                                      Simple similarity           MoL (multi-embedding + gating)
  Filter kills candidates             Post-filtering              ABM pre-filtering (V1 or V2)
  (liquidity issue)                   ANN ignores filters         Custom CUDA clause kernels

  Index too large for GPU             FP16 too large              1-bit Sign-OPORP (16× reduction)
                                                                  Two-stage: 1-bit filter + FP16 rerank

  Stale results / new posts missing   Batch offline only          Nearline embedding pipeline
                                                                  Venice CDC live upsert/delete

  Cold start (new user/item)          No engagement history       Cluster ID embeddings (K-means, frozen)
                                                                  LLM encoder from profile text

  High latency                        Large index                 V1/V2 routing by pass rate
                                      Expensive matmul            V3 quantized pre-filter (1%)
                                                                  ANN as last resort (loses filtering)

Key numbers to remember (LiNR A/B results):
  +7%   total professional interactions
  +3%   daily unique high-quality interactors
  -20%  skipped update rate (users reading, not scrolling past)
  +6%   from freshness alone (live index vs batch-rebuilt)
  +23%  Hit Rate@400 from MoL vs cosine baseline (offline)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11. LiNR System Architecture and Serving Path
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What is the "Retrieval Model"?

  ABM + KNN is the retrieval process, not the model.
  The retrieval model is a two-tower neural network:

    Member tower: maps member features → member embedding
    Item tower:   maps item features   → item embedding  (run OFFLINE)

  LiNR bundles everything into ONE model binary loaded into GPU memory:
    - Member tower weights       (for live or pre-computed query embedding)
    - Pre-computed item embeddings  (output of item tower, N x D matrix in GPU)
    - ABM attribute matrix + counting matrix  (for clause checking)

  Traditional systems keep these in separate services: model server, FAISS index,
  attribute filter. LiNR co-locates all three on GPU — fewer network hops, lower latency.

End-to-End Serving Path:

  Query arrives (member_id)
         |
         v
  1. KV Lookup: fetch pre-computed member embedding from Venice/Redis  [microseconds]
     (NOT running member tower live — embedding kept fresh by nearline pipeline)
         |
         v
  2. Assemble ABM clauses from:
       - Explicit UI filters (user selected location, job type)
       - Stored job preferences (saved in member profile)
       - Profile-derived implicit attributes (current location, language)
       - Business rules (exclude applied jobs, blacklisted companies — reverse clauses)
         |
         v
  3. LiNR GPU Node (Model Cloud L0):
       Route by estimated pass rate:
         High pass rate  → V1: full matmul + mask * scores + top-K  [4.8ms on A100]
         Low pass rate   → V2: clause check → slice → small matmul  [1.9ms on A100]
         Memory pressure → V3: 1-bit XOR pre-filter → FP16 on survivors
         |
         v
  4. L1 Ranker (lightweight model): 2000 → 200 candidates
         |
         v
  5. L2 Ranker (full neural model, Feed Service): 200 → top shown

  Why pre-compute member embeddings instead of live inference?
    Running a two-tower or LLM encoder at query time: 10-100ms latency hit.
    Pre-computed: nearline pipeline updates embedding within seconds of behavior change.
    At query time: one KV lookup replaces entire member tower forward pass.

Ingestor Pipeline (Section 4.2.2):

  Item embeddings + attributes flow into GPU memory via:

  Index Store (source of truth — Venice Store):
    Offline:  Apache Beam batch job joins item features + embeddings → Venice
    Nearline: streaming pipeline writes new/updated items → same Venice Store
    Both paths write to the same store — offline handles bulk, nearline handles delta.

  Updator (live update subscriber):
    Subscribes to Venice CDC (Change-Data-Capture) stream
    Upsert event → write embedding into pre-allocated GPU tensor slot
    Delete event → mark slot inactive via high-water mark
    Thread-safe: minimal serialization, inference path unblocked
    Benchmarked: zero measurable latency impact at 600 updates/sec on A100

  Bootstrapper (cold start recovery):
    On service restart: replay Venice CDC stream from saved offset
    Compacted snapshots on disk → fast warm start, avoids full stream replay

Can an LLM Encoder Replace the Two-Tower?

  Yes — serving infrastructure is unchanged. Only the embedding source changes.

    Item tower  → LLM encoder run offline  → richer item embeddings in GPU index
    Member tower → LLM encoder run offline → richer member embeddings in KV store

  MoL (Mixture of Logits) handles multiple embedding types as parallel components:
    delta_1 = two-tower embedding similarity
    delta_2 = GNN embedding similarity
    delta_3 = LLM encoder embedding similarity   <- add here
    phi_MoL = sum_k  pi_k(member, item) * delta_k(member, item)

  Gating function pi_k is input-dependent — learns which embedding source
  matters most for each specific member-item pair at inference time.

  LLM encoder benefit: world knowledge → better cold start, vocabulary bridging,
    semantic understanding without co-click data.
  LLM encoder cost: large model → slower offline inference → higher nearline latency
    for the embedding generation pipeline (Pipeline 2).

