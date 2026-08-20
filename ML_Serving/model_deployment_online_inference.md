# Model Deployment and Online Inference

From a trained model checkpoint to a production serving system.
Two distinct phases: **post-training preparation** (what you do to the model before
it ever sees a live request) and **online inference system design** (how you build
the system that serves requests in real time).

---

## Part 1 — Post-Training Steps

A trained model checkpoint is not production-ready. It is typically too large, too
slow, possibly unsafe, and not versioned. Post-training is the pipeline that transforms
it into something that can be deployed reliably.

```
Trained Checkpoint
       │
       ▼
  Evaluation & Benchmarking    ← is it actually good?
       │
       ▼
  Safety & Alignment           ← is it safe to ship?
       │
       ▼
  Format Conversion            ← export to format your serving engine accepts
       │
       ▼
  Model Optimization            ← quantization, compilation, kernel fusion
       │
       ▼
  Model Registry               ← version, tag, track
       │
       ▼
  Canary / Shadow Deploy       ← validate before full rollout
       │
       ▼
  Production
```

---

### Step 1 — Evaluation and Benchmarking

Before doing anything else, measure the model against the task it will serve.

**Offline metrics** (computed on held-out datasets):
- Task-specific: accuracy, F1, BLEU, ROUGE, perplexity
- Capability benchmarks: MMLU, HellaSwag, HumanEval (for code), TruthfulQA
- Regression check: compare against the previous production model

**Serving metrics** (measured on representative hardware):
- Time to First Token (TTFT) — latency from request to first generated token
- Time Per Output Token (TPOT) — latency between consecutive tokens
- Throughput — tokens/sec at target batch size
- Memory footprint — peak GPU VRAM at serving batch size

You need both. A model that scores well offline but takes 10 seconds to first token
fails the product requirement. A fast model that regresses 3% on MMLU is a silent
quality bug.

---

### Step 2 — Safety and Alignment Evaluation

For LLMs specifically, before shipping:

- **Red-teaming**: adversarial prompts designed to elicit harmful, biased, or
  policy-violating outputs. Done by a dedicated team or automated pipeline.
- **Toxicity and bias benchmarks**: measure model output distributions across
  demographic groups.
- **Policy compliance**: does the model refuse requests it should refuse? Does it
  over-refuse safe requests? Both are failure modes.
- **RLHF / DPO alignment check**: if the model was fine-tuned with human feedback,
  verify the alignment held on the evaluation set.

Output: a sign-off from a safety review, often a hard gate before any deployment.

---

### Step 3 — Format Conversion

Convert the training checkpoint into the format your serving engine accepts.
This step is a prerequisite to optimization — you cannot benchmark with
TRT-LLM until the model is in TRT-LLM's format.

See `format_conversion.md` for a full breakdown of each format.

**Decision table — what your engine needs:**

| Serving engine | Input format | Conversion needed |
|---------------|-------------|------------------|
| vLLM / SGLang | HuggingFace safetensors | None — load directly |
| TensorRT-LLM  | TRT-LLM internal format | `convert_checkpoint.py` → `trtllm-build` |
| TensorRT core | ONNX | `torch.onnx.export()` |
| ONNX Runtime  | ONNX | `torch.onnx.export()` |

The converted artifact (`.engine`, `.onnx`, or safetensors) is what flows
through the rest of the pipeline — used for benchmarking in Step 4 and stored
in the registry in Step 5.

---

### Step 4 — Model Optimization Techniques

Reduce cost per token. Apply optimizations to the converted artifact from Step 3.

**Quantization** — reduce numerical precision of weights (and optionally activations):

| Format | Bits | Memory vs FP16 | Use case |
|--------|------|----------------|----------|
| FP16 / BF16 | 16 | 1× (baseline) | Default training/serving |
| INT8 (W8A16) | 8 weights, 16 act | ~0.5× | Good quality/speed tradeoff |
| INT8 (W8A8) | 8 weights, 8 act | ~0.5× | Faster matmuls, slight quality loss |
| INT4 (AWQ, GPTQ) | 4 weights | ~0.25× | Fits larger models on fewer GPUs |
| FP8 | 8 (float) | ~0.5× | H100 native, better than INT8 for quality |

Post-training quantization (PTQ): quantize after training, no retraining needed.
Quantization-aware training (QAT): simulate quantization during training, better
quality but expensive.

See `quantization.py` for implementation of symmetric/asymmetric INT8.
See `model_optimization_techniques.md` for production tools and decision matrix.

**Pruning** — remove weights that contribute little to output (structured or
unstructured). Requires fine-tuning to recover quality. See `model_pruning_and_regularization.md`.

**Distillation** — train a smaller student model to mimic a larger teacher.
See `distillation.md`.

---

### Step 5 — Model Registry

A model registry is a versioned store of model artifacts with metadata. It answers:
"which model is in production right now, and what went into it?"

Typical metadata stored per version:
```
model_id:          llama3-8b-instruct-v2
base_model:        meta-llama/Llama-3-8B
fine_tuning_data:  hdfs://datasets/linkedin-finetune-v4
quantization:      AWQ INT4
eval_scores:       {mmlu: 0.68, task_f1: 0.84}
safety_sign_off:   approved 2026-06-15 by @safety-team
artifact_path:     hdfs://models/llama3-8b-v2/
created_by:        ml-platform-team
status:            production   # staging / shadow / production / deprecated
```

Popular tools: MLflow Model Registry, HuggingFace Hub, W&B Artifacts, internal
systems (LinkedIn uses an internal registry tied to their compliance pipeline).

---

### Step 6 — Canary and Shadow Deployment

Before routing live traffic to a new model:

**Shadow deployment**: new model receives a copy of live traffic but its responses
are discarded. Compare output distributions and latency without user impact.

**Canary deployment**: route a small percentage (1–5%) of live traffic to the new
model. Monitor quality metrics and error rates. Ramp up if healthy, roll back if not.

**A/B test**: split traffic deliberately between old and new model, measure business
metrics (click-through rate, user engagement) to confirm the model change is an
improvement, not just an offline metric improvement.

---

## Part 2 — Online Inference System Design

Online inference = serving model predictions to live user requests with latency
requirements (typically p99 < 500ms for LLMs, p99 < 50ms for embedding/classification).

### High-Level Architecture

```
Client (mobile app / web / API)
        │
        ▼
  API Gateway / Load Balancer
  - auth, rate limiting, request validation
  - routes to model serving cluster
        │
        ▼
  Inference Service (REST / gRPC)
  - request queuing
  - batching (groups requests for GPU efficiency)
  - routes to model backend
        │
        ▼
  Model Backend (vLLM / TGI / Triton / custom)
  - holds model weights in GPU VRAM
  - runs forward pass
  - manages KV cache
  - streams tokens back (for LLMs)
        │
        ▼
  Response (streamed tokens or single payload)
        │
        ▼
  Client
```

---

### Request Lifecycle (LLM text generation)

```
1. Client sends prompt → API Gateway
2. Gateway validates, rate-limits, enqueues request
3. Scheduler picks request, adds to next batch
4. Prefill phase: process all input tokens in parallel → produces KV cache
5. Decode phase: auto-regressively generate output tokens one at a time
   - each step: attend over KV cache + generate next token
   - stream each token back to client via SSE / WebSocket
6. EOS token generated or max_tokens reached → close stream
```

TTFT is dominated by the prefill phase (longer prompt = higher TTFT).
TPOT is determined by the decode phase (one forward pass per token).

---

### Batching Strategies

Batching is the single biggest lever for GPU throughput. GPU utilization is low
when processing one request at a time.

**Static batching**
Wait for B requests, process them together. Simple but introduces head-of-line
blocking — a short request waits for slow ones in the same batch.

```
Batch of 4 requests:
  req1: ██████████████████████████████ (long)
  req2: ████████ (short — done early but must wait)
  req3: ████████████████ (medium)
  req4: ████████████████████████ (medium-long)
  GPU sits idle after short requests finish until the longest finishes
```

**Dynamic batching / continuous batching** (used by vLLM, TGI)
New requests join the batch mid-decode as slots free up. No request waits for
the slowest in the batch. The GPU is always processing the maximum number of
in-flight sequences.

```
Step 1: [req1, req2, req3, req4]
Step 5: req2 finishes → slot freed → req5 joins immediately
Step 5: [req1, req3, req4, req5]   ← no idle GPU time
```

This is why continuous batching gives 2–10× higher throughput than static batching
for LLMs.

---

### KV Cache Management

During decode, every generated token must attend over all previous tokens. The
key-value matrices for all previous tokens are cached in GPU VRAM (the KV cache).

Problem: KV cache size grows with sequence length and batch size, competing with
model weights for VRAM.

**PagedAttention** (vLLM's key innovation):
- Stores KV cache in fixed-size pages (like OS virtual memory)
- Pages are allocated on demand, not pre-reserved for max sequence length
- Pages from different sequences can be interleaved in VRAM
- Near-zero VRAM waste → can fit more sequences in parallel → higher throughput

Without PagedAttention: VRAM allocated for max_seq_len even if actual sequence is short.
With PagedAttention: VRAM allocated per actual page used — significantly more efficient.

---

### Tensor Parallelism and Pipeline Parallelism

For models too large to fit on a single GPU:

**Tensor Parallelism (TP)**
Split individual weight matrices across GPUs. Each GPU holds a shard of every
layer. All GPUs participate in every forward pass, communicating via all-reduce.
- Reduces per-GPU memory proportionally
- Requires fast interconnect (NVLink) — high communication overhead over PCIe
- Best within a single node (same NVLink fabric)

**Pipeline Parallelism (PP)**
Split model layers across GPUs. GPU 0 runs layers 0–8, GPU 1 runs layers 9–16, etc.
Data flows as a pipeline — micro-batches keep all GPUs busy.
- Lower communication overhead than TP (only activations passed between stages)
- Introduces pipeline bubble (GPUs idle at start/end of pipeline)
- Best across nodes (only need to send activations, not all-reduce)

**Typical production setup:**
TP within a node (e.g. 8 GPUs with NVLink), PP across nodes.

---

### Scaling and Load Balancing

**Horizontal scaling**: run multiple replicas of the inference service.
Load balancer distributes requests across replicas.

Key consideration: LLM replicas are stateful (KV cache is in-memory).
Requests cannot be freely moved mid-generation. A request must stay on the
same replica for its entire lifetime.

Strategies:
- **Least-loaded routing**: send new requests to the replica with fewest
  in-flight sequences
- **Prefix caching**: route requests with the same system prompt prefix to
  the same replica — KV cache for the shared prefix is already computed
- **Autoscaling**: scale replicas up on high request rate, down when idle
  (Kubernetes HPA on custom metric: requests_per_second or GPU utilization)

---

### Streaming Responses

LLMs generate tokens one at a time. Users expect to see tokens stream in
as they are generated (like ChatGPT), not wait for the full response.

**Server-Sent Events (SSE)**: server pushes token chunks over a persistent
HTTP connection. Simple, unidirectional, works over HTTP/1.1.

**WebSocket**: bidirectional, lower overhead for high-frequency streaming.
Overkill for token streaming but used when the client also sends incremental data.

**gRPC streaming**: efficient binary protocol, good for service-to-service
(inference service → API gateway), not browser-facing.

---

### Observability

The four metrics that matter for online LLM serving:

| Metric | What it measures | Target |
|--------|-----------------|--------|
| TTFT (Time to First Token) | Prefill latency — how long until user sees first token | < 500ms p99 |
| TPOT (Time Per Output Token) | Decode speed — token streaming rate | < 50ms p99 |
| Throughput | Tokens/sec across all requests | maximize |
| GPU utilization | % time GPU is doing useful compute | > 80% |

Additional signals:
- Request queue depth — rising queue = system under capacity
- KV cache utilization — near 100% means OOM risk, need to shed load
- Error rate (5xx) — model crashes, OOM errors, timeout rate

---

### Caching Layers

Not every request needs a GPU forward pass:

**Prompt / prefix caching**: if many requests share the same system prompt,
pre-compute and cache its KV representation. Subsequent requests skip the
prefill for the shared prefix → lower TTFT.

**Result caching**: for deterministic use cases (classification, embedding),
cache (input → output) with a hash key. Hit rate can be very high for
repeated or similar inputs (see `ml_serving_infra_optimizations.md`).

**Semantic caching**: embed the query, find nearest cached query in vector
space, return cached result if similarity > threshold. Catches paraphrase
queries that are semantically identical but textually different.

---

### Request Routing for Multi-Model Systems

Production systems often serve multiple models (different sizes, different
fine-tunes for different tasks). Routing decides which model handles each request.

```
Request
   │
   ▼
Router
  ├── simple query      → small model (7B)   ← fast, cheap
  ├── complex reasoning → large model (70B)  ← slower, expensive
  └── code generation   → code model         ← specialized
```

Routing signals: query complexity (token count, perplexity), task type
(classification, generation, embedding), user tier (latency SLA).
See `llm_routing.md` for routing strategies in depth.

---

## Part 3 — Continuous Deployment

A model deployed to production is not static. Data distributions shift, new
training data accumulates, model quality degrades over time, and better architectures
emerge. Continuous deployment is the system that keeps the live model current without
disrupting users.

---

### When to Retrain

Two triggers:

**Scheduled retraining**: retrain on a fixed cadence (weekly, monthly) regardless
of observed quality. Simple to operate, easy to reason about. Works when data
accumulation rate is predictable.

**Drift-triggered retraining**: monitor production signals and trigger retraining
when quality degrades beyond a threshold.

```
Signals that indicate retraining is needed:
  - Data drift:    input distribution shifts (new slang, new products, seasonal topics)
  - Concept drift: relationship between input and correct output changes
                   (sentiment of the word "sick" means ill vs. excellent)
  - Performance drift: business metrics degrade (CTR, engagement, user satisfaction)
  - Label drift:   new ground-truth labels collected show model is wrong
```

---

### Incremental Training (Fine-tuning on New Data)

When the base model is good and only needs to adapt to new data — not a full
retrain from scratch.

**The core challenge: catastrophic forgetting**
Fine-tuning on new data causes the model to overwrite what it learned from old
data. After fine-tuning on week-5 data alone, the model may degrade on week-1
through week-4 patterns.

**Solutions:**

**Replay buffer / mixed dataset**: mix new data with a sampled subset of old
training data. The ratio controls the tradeoff between adapting to new data and
retaining old knowledge.

```
fine_tune_dataset = new_data (80%) + sampled_old_data (20%)
```

**Lower learning rate**: use a smaller learning rate than original training.
The model is already close to a good solution — large steps risk catastrophic
forgetting.

**LoRA / adapter fine-tuning**: freeze base model weights, train only lightweight
adapter layers on new data. Base model knowledge is preserved by construction.
Adapters are small (MBs vs GBs for full weights) and can be swapped cheaply.

```
Base model weights  →  frozen (never updated)
LoRA adapter        →  trained on new data, merged or hot-swapped at serve time
```

---

### Full Retraining

When the model architecture changes, training data distribution has fundamentally
shifted, or quality cannot be recovered by fine-tuning.

**Key operational concern**: full retraining takes hours to days. The live model
must keep serving while the new model trains in parallel.

```
t=0      Live: Model v1 serving 100% traffic
              Training: Model v2 starts training on updated dataset

t=48h    Live: Model v1 still serving
              Training: Model v2 training complete
              Evaluation: v2 offline eval running

t=50h    Live: Model v1 still serving
              Shadow: Model v2 deployed in shadow (gets copy of traffic, output discarded)
              Monitoring: latency, error rate, output distribution compared

t=54h    Live: v1 95% traffic, v2 5% (canary)
              Monitoring: business metrics, quality metrics on 5% slice

t=72h    Live: v2 100% traffic (full rollout)
              v1 kept warm for rollback window (24–48h), then decommissioned
```

---

### Rollout Strategies

**Shadow deployment**
New model receives a copy of every live request. Its outputs are never shown to
users. Compare output distributions, latency, and error rates against the live
model with zero user risk.

Use when: validating a major model change (new architecture, new training data)
before any user exposure.

**Canary deployment**
Route a small percentage of real traffic (1–5%) to the new model. Users in the
canary slice see real outputs. Monitor quality and latency metrics on this slice
before ramping.

Use when: shadow passed and you need real-world signal on user-facing quality.

**Blue-Green deployment**
Maintain two full production environments: Blue (current) and Green (new). Switch
the load balancer from Blue to Green atomically. If something goes wrong, switch back
instantly — no gradual ramp.

```
Before:  Load Balancer → Blue (v1, 100%)   Green (v2, 0%, warm and ready)
Switch:  Load Balancer → Blue (v1, 0%)     Green (v2, 100%)
Rollback: flip back in seconds
```

Use when: you need an instant cutover with instant rollback capability.
Trade-off: requires 2× the infrastructure cost while both environments are live.

**Rolling deployment**
Replace replicas of the old model one at a time with the new model. At any point
some replicas serve v1 and some serve v2.

```
t=0:  [v1, v1, v1, v1, v1, v1, v1, v1]   8 replicas
t=1:  [v2, v1, v1, v1, v1, v1, v1, v1]   1 replaced
t=2:  [v2, v2, v1, v1, v1, v1, v1, v1]   2 replaced
...
t=8:  [v2, v2, v2, v2, v2, v2, v2, v2]   full rollout
```

Use when: infrastructure cost is a constraint and gradual rollout is acceptable.
Trade-off: both versions serve traffic simultaneously — inconsistent user experience
if v1 and v2 produce meaningfully different outputs for the same user.

---

### Evaluation Gates

Every rollout must pass a sequence of gates. A gate failure stops the rollout and
may trigger automatic rollback.

```
Gate 1 — Offline eval
  ├── task metric vs baseline (must not regress beyond threshold)
  ├── safety evaluation pass
  └── latency benchmark within SLA

Gate 2 — Shadow eval (24–48h)
  ├── error rate comparable to live model
  ├── output distribution not significantly shifted
  └── latency p99 within SLA

Gate 3 — Canary eval (24–72h on 5% traffic)
  ├── business metrics (CTR, engagement) not degraded on canary slice
  ├── user-facing error rate stable
  └── no latency regression

Gate 4 — Full rollout
  ├── all above gates passed
  └── human sign-off (for major model changes)
```

---

### Automatic Rollback

Don't rely on humans to catch every production issue. Define rollback triggers
that fire automatically:

```
Triggers:
  - error rate > 1% over 5-minute window
  - p99 latency > 2× baseline for 10 minutes
  - KV cache OOM rate > 0.1%
  - output toxicity score > threshold (if online safety classifier is running)

Action:
  - immediately shift traffic back to previous model version
  - page on-call
  - new model version stays warm for post-mortem
```

The previous model version must be kept warm (not decommissioned) for the rollback
window. Decommission only after the new model has been stable in production for
a defined period (typically 24–48h).

---

### LoRA Hot-Swap (Continuous Adaptation Without Downtime)

For systems that need frequent adaptation (daily or faster) without full
redeployment, LoRA adapters enable near-zero-downtime model updates:

```
Base model:   loaded once into GPU VRAM, never unloaded
LoRA adapter: small weight delta, loaded on top of base model

Update cycle:
  t=0:  serving base + adapter_v1
  t=1:  new adapter_v2 trained on today's data
  t=2:  adapter_v2 loaded into memory (MBs, takes milliseconds)
  t=3:  serving base + adapter_v2   ← no GPU restart, no traffic interruption
```

This pattern is particularly powerful for personalization (per-user adapters)
or rapid domain adaptation (daily news, trending topics).

---

## Interview Cheat Sheet

**Post-training pipeline order:**
```
Evaluate → Safety → Convert format → Optimize (quantize + compile + CUDA graphs) → Register → Canary → Prod
```

**Key latency terms:**
- TTFT = time to first token (prefill dominated)
- TPOT = time per output token (decode dominated)
- E2E latency = TTFT + TPOT × num_output_tokens

**Throughput lever stack (most to least impactful):**
1. Continuous batching (fills GPU with multiple requests)
2. Quantization (smaller model = more sequences in VRAM)
3. PagedAttention (efficient KV cache = more concurrent sequences)
4. FlashAttention (faster attention kernel)
5. Prefix caching (skip prefill for shared prompts)

**Scaling model too large for one GPU:**
- Same node, fast interconnect → Tensor Parallelism
- Across nodes → Pipeline Parallelism
- Both → TP within node, PP across nodes

**Three caching layers:**
1. Result cache (exact match, deterministic outputs)
2. Prefix/KV cache (shared system prompt, skip prefill)
3. Semantic cache (nearest-neighbor, paraphrase queries)

**Rollout strategy decision tree:**
```
Major architecture change?  → Shadow first, then canary
Need instant rollback?      → Blue-Green
Infra cost constrained?     → Rolling deployment
Frequent small updates?     → LoRA hot-swap
```

**Catastrophic forgetting mitigations (incremental training):**
1. Replay buffer — mix new + sampled old data
2. Lower learning rate — small steps near existing solution
3. LoRA — freeze base weights, train only adapters

**Rollback requirements:**
- Keep previous version warm for 24–48h after full rollout
- Define automatic triggers (error rate, latency, OOM)
- Never decommission until stability window passes
