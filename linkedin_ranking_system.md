LinkedIn GR System

Feed SR — LinkedIn's Sequential Ranking Model

Paper: "An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking" (arXiv 2602.12354, Feb 2026)

Replaces the previous DCNv2-based ranker. Result: +2.10% time spent in A/B test.
Now the primary ranking experience for LinkedIn Feed.




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. What Feed SR Is
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Feed SR is a decoder-only transformer that reads a member's full interaction history
and scores all candidate posts in a single forward pass.

It sits at the ranking stage (L2 ranker) in the feed pipeline:

  Retrieval (LiNR) → top-2000 candidates
       ↓
  L1 ranker (lightweight)
       ↓
  Feed SR (L2 ranker, transformer) → top-N for member
       ↓
  Business logic → final feed


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Model Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input — Interleaved Sequence

  The model sees the last 1000 impressions from 1 year of history, as:
    [post_1, action_1, post_2, action_2, ..., post_1000, action_1000]
  → 2000-token sequence fed into the transformer

  Each post token = feature vector:
    actor/root-actor hashed ID embeddings (who posted/reshared)
    content embedding (50-dim from LinkedIn's post embedding model)
    categorical features (post type, device, verb type, object type)
    lightweight numeric features (actor popularity, dwell-time affinity)

  Each action token = multi-hot vector (click / like / comment / share / skip / long-dwell)
    → projected to same dimension via learnable linear layer

Transformer

  Decoder-only, causal attention mask.

  Key choices:
    Pre-LayerNorm (Pre-LN): essential — without it, training AUC collapses to 0.5
    RoPE positional embeddings: replaces learned absolute positions
      Why: absolute position embeddings are under-trained for large indices
           because member histories are long-tailed (most members have short histories)
           → embeddings at positions 800-1000 receive very few gradient updates
           → prediction scores drift during training
      RoPE: deterministic rotation applied to Q and K, no learned lookup table
           → stable training, +0.20% Long Dwell AUC
    Scaled residuals: RescaleAndAdd(u, v) = u + α*v  (α is learnable scalar per block)
      Provides stable training with minimal complexity vs vanilla residual
    SDPA with causal mask (see Section 4 for multi-item scoring optimization)

Early Fusion vs Late Fusion — Key Design Decision

  Early fusion: all features go INTO the transformer sequence (increases token dim → bigger model)
  Late fusion: only history-relevant features go into transformer; candidate features added AFTER

  Feed SR uses LATE FUSION:

    Sequence features (into transformer):
      actor/root-actor ID embeddings, content embedding, categorical features
      → things that benefit from seeing patterns ACROSS the interaction history

    Context features (concatenated to transformer OUTPUT, before prediction head):
      candidate popularity counts
      bucketed dwell-time popularity (e.g., how many people dwelled 30s+ on this post)
      viewer-actor affinity (historical engagement between this member and this author)
      post age, viewer network strength
      → things that describe the CANDIDATE at scoring time, not the history

  Why late fusion?
    Candidate popularity at position t in history tells you nothing about whether
    TODAY's candidate is popular. Injecting it into every history token adds noise.
    Late fusion reduces transformer input dimension → 12% training speedup, no accuracy loss.
    Also simplifies serving: history features stored without candidate-specific signals.

  Ablation result: moving 1/3 of features to late fusion → only 0.07% Long Dwell AUC drop,
  12% training speedup. Late fusion online A/B matched or exceeded early fusion.

Head Architecture — MMoE

  Input: transformer output (history encoding) concatenated with context features
  Architecture tested: Linear, MLP, DCNv2, MMoE
  Winner: MMoE (Multi-gate Mixture of Experts)

  Two task groups matching existing production ranker:
    Passive tasks: click, skip, long-dwell
    Active tasks: like, comment, share
  Shared experts across groups; per-task learned gating function (softmax over experts)
  Dropout on post-softmax gates during training → prevents expert collapse

  MMoE performance over linear baseline:
    Long Dwell AUC:   +0.44%
    Contributions AUC: +0.58%

Member Profile Embeddings — Cold Start Fix

  Problem: members with <10 historical actions get poor recommendations.
  Solution: Qwen3 0.6B fine-tuned on LinkedIn member profiles.
    Input: profile text (job title, company, skills, summary)
    Output: dense embedding representing professional interests
    Refreshed: daily (as member updates profile)
    Fusion: late-fused as a context feature (does not increase transformer dimension)

  Result: +2% AUC for members with <10 historical actions.
  The embedding captures domain knowledge from profile text alone — no engagement needed.

Training Techniques

  Loss: binary cross-entropy
  Negative sampling: down-sample impressions with no click at 0.1 rate

  Recency-weighted loss (two mechanisms):
    Position weighting: exponential decay within each sequence
      first position → 50% weight, last (most recent) → 100% weight
      encourages model to learn more from recent behavior
    Timestamp weighting: sample-level exponential decay by data age
      60-day half-life → 2-month-old samples get 50% weight
      keeps model fresh on recent distribution

  Incremental training:
    Cold-start: train on full 1-year interaction history
    Daily updates: compute loss only on NEW interactions (fresh data)
    Full historical sequence still provided as context during incremental steps
    Model updated daily to reflect current member behavior

  In-session leakage fix:
    Items within a 30-minute session have highly correlated labels
    (if member dwells long on item 1 in session, they likely dwell on item 2 too)
    During training, the model can exploit this — overfits session patterns not
    seen at serving time (session labels not available online)
    Fix: randomize order of items WITHIN each session during training
    (prevents model from learning session position as a proxy for label)

What Was Tried and Rejected

  HSTU layers (from Meta's generative recommender paper):
    Consistent -0.21% Long Dwell AUC vs standard transformer at matched compute.
    Also not compatible with standard FlashAttention → memory issues at scale.
    Feed SR uses standard transformer blocks.

  LLM-Ranker (LLaMA 3 fine-tuned):
    Fine-tuned to output "Yes"/"No" for "Will member click on this post?"
    Problems:
      Each post = hundreds of tokens → 1000-item history = tens of thousands of tokens
      Expensive to train and serve
      LLM can't leverage numeric features well (e.g., "# of Likes: 382" as text)
      Struggled with network-based recommendations (strength of connection hard to text-encode)
    Never beat production model online.

  TransAct (history augmentation on top of existing DCNv2 ranker):
    Good offline and online metrics.
    But: big increase in training time and inference latency for long sequences.
    Also hard to deploy in a point-wise scoring stack.
    Feed SR (sequence-centric) naturally resolves these issues via amortized computation.

Scaling Laws

  For every 10× increase in training FLOPs → +0.0093 Long Dwell AUC.
  Best single axis to scale: sequence length (consistent gains across all metrics).
  Scaling layers or ID embedding dim alone → inconsistent, sometimes non-monotonic.
  Lesson: scale multiple dimensions jointly, not one in isolation.

  Longer history: 6 months vs 1 year → 1 year gives +0.21% Long Dwell AUC, +0.15% Contribution AUC,
  +20% member coverage. Cost: 12% more training time.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. System Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Warm-Up + Compilation at Serving Startup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loading model weights into GPU VRAM is not the same as being ready to serve traffic.
Two additional steps must complete before the server opens to requests.


3.1 What Happens During Warm-Up

  After weights are loaded, the server runs N dummy inference passes with representative
  inputs before accepting any real traffic. Each pass absorbs three costs:

  1. CUDA kernel JIT compilation
     PyTorch kernels are compiled to GPU binaries on first use and cached.
     Without warm-up, the first real user request triggers this compilation --
     adding hundreds of milliseconds to that request's latency.

  2. Memory allocator warm-up
     PyTorch's CUDA caching allocator learns allocation patterns over the first few
     passes and pre-allocates memory pools. First-pass allocations are slower;
     subsequent passes reuse cached blocks in O(1).

  3. torch.compile tracing
     torch.compile defers actual kernel generation to the first forward pass per
     input shape. Warm-up triggers tracing and compilation for each shape upfront.

  Result: first real request hits the compiled, warm path -- stable, predictable latency.


3.2 Interaction with Bucketed Batching

  Feed SR uses bucketed batching: requests are padded to predefined bin sizes
  (e.g. seq_len = 64, 128, 256, 512, 1024) so the GPU always sees a finite set of shapes.

  Warm-up must cover every bucket shape, not just one:

    for shape in [64, 128, 256, 512, 1024]:
        dummy_input = torch.zeros(batch_size, shape).to("cuda")
        model(dummy_input)   # triggers compile + JIT for this shape

  If warm-up only covers shape 256 and the first real request arrives with shape 512,
  torch.compile traces a new graph at serving time -- that request pays the full
  compilation cost. Warm-up over all bucket shapes eliminates this entirely.


3.3 Compilation

  torch.compile(model) wraps the model with the Inductor backend.
  On the first forward pass per shape, it:
    - Traces the computation graph
    - Fuses multiple small ops into single CUDA kernels (fewer kernel launches)
    - Compiles fused kernels via NVCC/Triton -> GPU binary
    - Caches the compiled graph (keyed by input shape)

  Subsequent passes for the same shape hit the compiled path directly.
  Typical effect: kernel count drops ~40%, latency for hot ops decreases.


3.4 Kubernetes Readiness Probe

  The readiness probe is the enforcement mechanism. The pod reports not-ready
  until warm-up completes. Kubernetes does not route traffic to the pod until
  the probe passes -- no real request ever pays the startup cost.

  Full startup sequence inside a pod:

    Model Manager downloads checkpoint from Artifactory
    (Model Name + Model Group + Model Version from config.json)
          |
          v
    Weights loaded into GPU VRAM
          |
          v
    torch.compile(model) -- wraps all three models (Feed SR, Creator, Session)
          |
          v
    Warm-up loop over all bucket shapes for all three models
          |
          v
    Readiness probe passes -> Kubernetes routes traffic


3.5 Interview Conversation -- Staff+ Level

  Interviewer: "Walk me through how the PyTorch Inference Server handles
  the first request after deployment."

  Weak answer (L4/L5):
    "The model gets loaded from the model registry and we run inference
    on incoming requests."

  Staff+ answer:
    "Loading the model is just the beginning -- the server isn't ready yet.
    After weights are in VRAM, we go through two more steps before opening
    to traffic.

    First, compilation. We wrap the model with torch.compile which defers
    kernel generation to the first forward pass per input shape. Since we use
    bucketed batching -- say shapes 64, 128, 256, 512, 1024 -- we trigger
    compilation for all five shapes upfront during warm-up, not on the first
    real user request.

    Second, warm-up itself. Dummy passes for each bucket shape absorb CUDA
    kernel JIT compilation and get the memory allocator into steady state.
    Without this, the first user request on a fresh pod could be 10-100x
    slower than steady state.

    The Kubernetes readiness probe enforces this. The pod reports not-ready
    until warm-up completes -- Kubernetes won't route traffic until the probe
    passes."

  Follow-up: "What if you have three models in the same Scorer Orchestrator?"

  Staff+ answer:
    "Warm-up must cover all three models -- Feed SR, Creator, and Session model.
    They share the same request shapes since they all score the same candidate
    batch, but we need one pass per model per bucket shape to ensure all compiled
    graphs are cached. The readiness probe gates on all three being warm. If you
    gate on just Feed SR and the Creator model JIT-compiles on the first real
    request, you still get that latency spike."

  Follow-up: "Any risk with torch.compile in production?"

  Staff+ answer:
    "Two main risks. First, recompilation on unexpected shapes. If a request
    slips through bucketing with an out-of-range shape, torch.compile traces
    a new graph at serving time -- that request pays compilation latency.
    Mitigation: clamp inputs to the largest bucket rather than letting novel
    shapes through.

    Second, silent fallback to eager mode. torch.compile can fall back silently
    under certain input patterns. In production you instrument for this -- log
    when a fallback happens, alert on it. A silent fallback means you're serving
    at eager-mode latency without knowing it."

  Why this is Staff+ level:
    A mid-level answer stops at 'load the model and run inference.'
    A Staff+ answer covers the full startup contract: compilation, warm-up,
    readiness probe, multi-model orchestration, and failure modes.
    It connects the ML concern (compilation) to the infrastructure concern
    (Kubernetes probe) to the reliability concern (silent fallbacks).
    That end-to-end ownership is what distinguishes the level.


3.1 Disaggregated Architecture : CPU Service + GPU Service
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Previous system: scored entirely on CPU in a Java stack.
  Feed SR: split into two independently scalable services.

  Inference Driver (CPU service):
    Feature fetching from KV stores
    Join member history features + candidate document features
    CPU-bound parsing and transformation
    Sends combined features to GPU service via gRPC

  PyTorch Inference Server (GPU service):
    Receives feature payload
    Runs transformer forward pass
    Returns scores for all N candidates

  Data transfer: Apache Arrow columnar buffers inside protobuf messages
    → zero-copy conversion from Arrow to PyTorch tensors
    → eliminates expensive serialization and memory copy for large history payloads

  Operational benefit: CPU and GPU workloads scaled independently based on bottleneck.
  If feature fetching is slow → scale CPU pods.
  If scoring is slow → scale GPU pods.

  Feature storage:
    Member history features: precomputed offline, stored as Arrow columnar buffers in KV store
    Document features: fetched at serving time per candidate
    Both services implemented in Python but restricted to high-performance libraries
    (NumPy, PyTorch, Arrow) — no pure Python loops in hot paths.


3.2 GPU-Side: Shared Context Batching — The Core Serving Innovation

  Problem: N=512 candidates share the SAME member history.
  Naive approach: run 512 separate forward passes, each with [history + one candidate].
  → 512× redundant computation of history encoding.

  Feed SR approach: ONE forward pass scores all 512 candidates simultaneously.

  Sequence layout:
    [hist_1, act_1, ..., hist_1000, act_1000, cand_1, cand_2, ..., cand_512]
    |←──────────── context (2000 tokens) ──────────→|←── candidates (512) ──→|

  Attention mask — two distinct patterns in ONE mask:
    History tokens (i ≤ 2000): causal attention
      each history token attends to all PRECEDING history tokens (standard causal)
    Candidate tokens (i > 2000): each candidate attends to ALL history tokens + itself
      candidates do NOT attend to each other (no cross-candidate leakage)

  Result: single forward pass → scores for all 512 candidates.
  Speedup: 80× on transformer forward pass for typical workload
           (512 candidates, history length 1000).




3.3 Custom Flash Attention Kernel — SRMIS

  Problem with shared context batching approach above:
    PyTorch's scaled_dot_product_attention (SDPA) falls back to a NON-Flash implementation
    when a custom attention mask tensor is provided.
    → loses FlashAttention's tiled computation, streaming softmax, and O(1) memory.

  Solution: custom CUDA kernel called SRMIS
    (Sequential Recommender Multi-Item Scoring)

  Design:
    Instead of a mask tensor, SRMIS takes two scalars:
      context_length = L (number of history tokens)
      candidate_length = N (number of candidates)
    The attention pattern is fully determined by these two values.
    Masking logic is implemented INSIDE the FlashAttention tiled kernel.

  Three benefits over masked SDPA:
    1. No mask materialization: no O((L+N)²) mask tensor allocated
    2. Compute skipping: tiles entirely outside valid attention region → skipped entirely,
       not computed-then-masked
    3. Full FlashAttention memory efficiency: tiled computation, streaming softmax

  Speedup: 2× over masked SDPA (exact gain depends on L and N).

  Combined with shared context batching: 80× speedup on transformer forward pass.


3.4 CPU-Side Optimizations

  The inference driver was initially a loop-based, non-vectorized Python implementation.
  Two operations were the critical bottlenecks:

  Member history parsing:
    Before: 450ms per request  (Python loops, item-centric access)
    After:  2ms per request    (225× speedup)

    Fix: NumPy strided arrays for bulk feature processing.
    Key insight: shift from item-centric to feature-centric access.
      Item-centric:    for each item → read all F features   → O(N×F), bad cache locality
      Feature-centric: for each feature → read all N values  → O(F), SIMD-friendly
    Reading one feature for all items at once fits in CPU cache;
    reading all features for one item thrashes the cache.

  Sparse-to-dense tensor conversion:
    Before: 254ms per feature  (nested Python loops)
    After:  5ms per feature    (50× speedup)

    Fix: replace nested Python loops with PyTorch tensor indexing + batched assignments.
    Complexity: O(N×M×D) → O(N + M×D)  where D=dims, M=avg values per doc.

  Hardware-level gains from these fixes:
    66% fewer CPU cycles
    71% fewer instructions
    90% fewer cache misses
    72% fewer branch mispredictions


3.5 Training Optimizations

  Four optimizations, applied independently, with combined GPU hour savings:

  Optimization                              GPU Hour Reduction
  ─────────────────────────────────────────────────────────────
  Fused C++ data loader                         50%
  (padding + batching + packing in C++)
  fused from ~300ms/step Python overhead

  Custom CUDA AUC computation kernel            22%
  66ms/step → 0.5ms/step (132× speedup)
  Original: masked confusion matrix ops + cudaStreamSynchronize per step
  Fix: fused bucketized kernel integrates boolean mask, eliminates dynamic alloc

  Fused Adam optimizer + gradient scaling       15%
  40ms/step → 20ms/step
  Removed redundant cudaStreamSynchronize calls by integrating gradient
  scaler directly into the CUDA optimizer kernel

  Parallelized evaluation                       16%
  Save all checkpoints during training;
  run evaluation on all checkpoints simultaneously in one pass
  (evaluation uses less GPU memory than training → batch them)


3.6 Scorer Orchestrator — Multiple Models at Inference

  Online scoring is not just Feed SR. Three models combine to produce the final score:

    Feed SR (ranking model)
    + Creator model
    + Downstream session model

  Scorer class orchestrates all three:
    Runs forward passes for all three models
    Applies objective function weights to derive final ranking score
    Deployed as a single shared checkpoint
    Config stored in HuggingFace-style config.json
    (class references + objective function weights)


3.7 Energy Comparison vs Previous Production Model

  Training:  3.6× more energy  (bigger model, trained on more data)
  Inference: 0.7× (30% LESS energy)
  → GPU inference with shared context batching is cheaper per request
    than CPU inference of the previous DCNv2 model, despite the larger model.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. A/B Test Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Member Segment        Time Spent
  ──────────────────────────────────
  Overall               +2.10%
  DAU (daily active)    +2.38%
  WAU (weekly active)   +1.84%
  MAU (monthly active)  +0.82%
  New members           not stat-sig

  Note: A/B excludes incremental training — early online experiments with incremental
  training showed additional gains beyond these numbers.

  Biggest gains for most active members: the transformer benefits most from long histories.
  New members neutral: profile embeddings help but not enough on their own.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Deployment Lessons
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Lesson 1 — Avoid negative sampling in evaluation
  Down-sampling negatives biases offline AUC and inflates it.
  Evaluating on the full candidate set → offline AUC better aligned with online metrics.
  Found: negative sampling caused offline/online discrepancies that misled development.

Lesson 2 — Offline/online score comparison pipeline
  Built a pipeline to score live Feed SR sessions using the offline model.
  Compared offline scores to logged online scores for the same items.
  Discovered a large number of bugs in the online stack whose fixes had major metric impact.
  Key tool for debugging feature discrepancies.

Lesson 3 — Java → Python feature transformation migration
  Previous system: feature transformations in shared Java codebase.
  Feed SR: PyTorch/Python stack — Java transformations no longer apply.
  Had to rebuild feature transformation logic in Python.
  Many of the bugs from Lesson 2 were Java↔Python feature parity issues.
  Building a principled Python-based shared feature transformation layer is ongoing.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Summary — Key System Design Decisions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Decision                      Reason
  ──────────────────────────────────────────────────────────────────────────────
  Disaggregated CPU + GPU       Independent scaling; GPU not wasted on feature fetch
  Arrow zero-copy transfer      Large history tensors; avoid serialization overhead
  Late fusion                   Context features don't belong in history; 12% training speedup
  Shared context batching       80× GPU speedup; one forward pass for 512 candidates
  SRMIS custom kernel           SDPA drops Flash for custom masks; 2× speedup recovered
  Pre-LN transformer            Training stability; Post-LN collapses AUC to 0.5
  RoPE over learned positions   Absolute positions under-trained for long-tail short histories
  Late-fused profile embeddings Cold start fix without inflating transformer dimension
  MMoE head                     Best multi-task performance; group routing mirrors task structure
  Incremental daily training    Model freshness without full daily retraining
  Session randomization         Prevents in-session label leakage during training
  Freeze cluster centroids      Non-trainable clusters outperform trainable (convergence rate mismatch)



