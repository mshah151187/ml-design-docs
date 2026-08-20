# Model Serving Contract

Before a model enters the build pipeline, the serving team sits with the model
owner (AI Modelers / Product Team) to establish a serving contract. This contract
captures everything the compiler and serving infrastructure need that the model
file itself cannot tell you.

Related: `compilation.md` (Model Bundle), `tensorrt_optimization_profiles.md`

---

## Why a Contract Is Needed

The model file (ONNX, safetensors) tells you:
- Model architecture and weights
- Which dimensions are static vs dynamic
- Input/output tensor names and dtypes

It does NOT tell you:
- What shapes will actually arrive at runtime
- Which workloads matter for performance vs just correctness
- What the latency requirements are
- Whether there are distinct client populations with different shapes
- How many concurrent requests to plan for

These gaps cannot be inferred from the model. They require a conversation.

---

## The Questions — and What Each Answer Drives

### Q1: What are your input shapes? Which dimensions are truly dynamic?

```
Why asked:
  ONNX marks dynamic dims as ? — TRT cannot proceed without shape bounds.
  Not all "dynamic" dims are equally dynamic at runtime.
  Some are fixed by the product (image always 224×224).
  Some vary per request (sequence length, batch size).

Example answers:
  "Batch is always 1 for our online API"
  → batch dim is effectively static for online — set min=max=opt=1

  "Sequence length varies between 64 and 512 depending on user input"
  → genuinely dynamic — needs a range

  "Image resolution is always 224×224"
  → mark as static even if ONNX shows it as dynamic

What this drives:
  → which dims need a profile range vs can be fixed
  → whether one profile is enough or multiple are needed
```

---

### Q2: What does your traffic distribution look like?

```
Why asked:
  opt shape in the profile is the ONLY shape tactic selection benchmarks at.
  Wrong opt = silent performance bug.
  Must be set to where most traffic actually lands, not a theoretical midpoint.

What to ask for:
  - p50, p75, p95, p99 of batch size
  - p50, p75, p95, p99 of sequence length / input size
  - Is distribution stable or does it shift (peak hours, seasonal)?

Example answers:
  "Sequence length p50=128, p75=256, p95=512, p99=768, max=1024 (rare, <0.1%)"

What this drives:
  opt = p75 shape (kernels tuned where most traffic lands)
  max = p99 or true max (correctness for tail, not tuned)

  Profile decision:
    min=(1, 64),  opt=(1, 256),  max=(1, 1024)

  If you set opt=p50 you leave performance on the table for p50-p99 traffic.
  If you set opt=p99 you over-tune for rare traffic, hurt the majority.
  p75 is usually the right balance.
```

---

### Q3: Are there distinct client populations with different shape profiles?

```
Why asked:
  If two client types have shapes far apart, one opt is a compromise for both.
  Multiple profiles let each cluster get kernels tuned for its actual shape.

Example answers:
  "We have a mobile API (batch=1, short queries) and a batch pipeline
   (batch=32, longer context) — both are production and both matter"

What this drives:
  → two optimization profiles

  Profile 0 (mobile/online):
    min=(1, 32),  opt=(1, 128),  max=(4, 256)

  Profile 1 (batch/offline):
    min=(8, 128), opt=(32, 512), max=(64, 1024)

  Serving code routes request type → profile index at runtime.

  If instead: "we only have an online API, batch size is always 1"
  → one profile is sufficient, no routing logic needed
```

---

### Q4: What are your latency requirements per client type?

```
Why asked:
  Latency SLO determines:
    - whether you need a dedicated GPU per model or can share
    - whether you need fp16/int8 (lower precision = faster but needs calibration)
    - whether CUDA graphs are worth the setup cost
    - batch size ceiling (larger batches = higher throughput but higher latency)

Example answers:
  "Online API: P99 < 150ms end-to-end, P50 < 50ms"
  "Batch pipeline: throughput > 1000 req/sec, latency flexible (< 5s acceptable)"

What this drives:
  Online SLO < 150ms:
    → fp16 or fp8 (fp32 too slow)
    → batch size capped at 4-8 to keep latency low
    → CUDA graphs worth enabling (eliminates kernel launch overhead for small batches)
    → dedicated GPU, no sharing with batch jobs

  Batch throughput 1000 req/sec:
    → large batch size (32-64) to maximize GPU utilization
    → fp8 or int8 for maximum throughput
    → sharing GPU with other batch models is fine
    → CUDA graphs less critical (kernel launch overhead amortized over large batch)
```

---

### Q5: How often does the model update? What is the acceptable rebuild time?

```
Why asked:
  TRT build with tactic selection takes 20-60 minutes per profile.
  If the model updates weekly, that's fine.
  If it updates multiple times per day, the build cost is a bottleneck.

Example answers:
  "Model is retrained weekly, we can absorb a 1-hour build window"
  → standard TRT build, timing cache carried across builds for unchanged layers

  "We hotswap model weights daily (same architecture, new weights)"
  → use timing cache aggressively — architecture unchanged, cached tactics reuse
  → build time drops to minutes on cache hit

  "We experiment with architecture changes multiple times per day"
  → TRT build is too slow for this cadence
  → consider torch.compile (faster build, less peak performance) for experimentation
  → promote to TRT only for production-stable architectures

What this drives:
  → whether TRT is the right compiler at all
  → timing cache strategy
  → whether to keep original/ weights in Model Bundle for fast fallback
```

---

### Q6: What GPU hardware will this model run on?

```
Why asked:
  TRT engine is hardware-locked — compiled for a specific SM architecture.
  sm_90 (H100) engine will not run on sm_86 (A100).
  If you serve on a mixed fleet, you need separate builds per GPU type.

Example answers:
  "We run exclusively on H100s in production"
  → one build: sm_90

  "We have H100s for online, A100s for batch (cost savings)"
  → two builds: sm_90 for online, sm_86 for batch
  → Model Bundle ships both engines:
       trtllm_engine/
         h100/rank0.engine    (sm_90)
         a100/rank0.engine    (sm_86)
  → bundle_metadata.pb maps gpu_model → engine path

  "We might add L40s in 6 months"
  → plan a third build now, or keep original/ weights in bundle
     so the build pipeline can produce sm_89 engine when needed
     without requiring the modeling team to re-export the ONNX

What this drives:
  → number of engine builds
  → Model Bundle structure
  → whether original/ weights must be preserved for future GPU migrations
```

---

### Q7: Do you need INT8 / FP8 quantization?

```
Why asked:
  Quantization changes the build process significantly:
    - requires calibration data (representative inputs from the model owner)
    - adds precision assignment and quantization lowering passes to the build
    - model owner must validate that accuracy drop is acceptable

  This is a joint decision — serving team cannot make it alone.

Example answers:
  "We need maximum throughput, accuracy drop of <1% on our eval set is acceptable"
  → proceed with INT8 or FP8
  → model owner provides calibration dataset (1000-10000 representative inputs)
  → serving team runs calibration during build, validates output on eval set
  → model owner signs off on accuracy before deploying quantized engine

  "This is a medical diagnosis model, no accuracy compromise"
  → fp16 only, no quantization
  → latency SLO must be achievable with fp16

What this drives:
  → build config: BuilderFlag.INT8 / BuilderFlag.FP8
  → calibration data pipeline
  → accuracy validation gate before deployment
  → who is responsible for accuracy regression (model owner, not serving team)
```

---

## The Contract Document

Once the conversation happens, it is formalized in a serving config that the
model owner submits alongside the model file:

```yaml
# serving_config.yaml — submitted by model owner, validated by serving team

model:
  name: "trip_eta_v3"
  onnx_path: "models/trip_eta_v3.onnx"

gpu_targets:
  - H100    # sm_90 build
  - A100    # sm_86 build

optimization_profiles:
  - name: "online"
    input_tensor: "input"
    min:  [1, 32]
    opt:  [1, 128]      # p75 of online traffic
    max:  [4, 512]
    latency_slo_ms: 150

  - name: "offline"
    input_tensor: "input"
    min:  [8, 64]
    opt:  [32, 256]     # median offline batch
    max:  [64, 1024]
    throughput_slo_rps: 1000

precision:
  mode: FP16            # FP32 / FP16 / INT8 / FP8
  # calibration_data: null  ← required if INT8 or FP8

update_frequency: weekly
timing_cache: reuse     # reuse / rebuild
preserve_original: true # keep original/ weights in bundle for fallback
```

---

## How Answers Flow Into Build Decisions

```
Q1 (which dims dynamic?)
  → which tensor names go into set_shape()
  → whether any dims can be declared static even in a "dynamic" model

Q2 (traffic distribution?)
  → opt value per profile
  → max value (p99 or true max)

Q3 (distinct client populations?)
  → number of optimization profiles
  → serving routing logic: request type → profile index

Q4 (latency SLO?)
  → precision choice (fp16 vs int8 vs fp8)
  → batch size ceiling
  → whether CUDA graphs are enabled
  → GPU allocation strategy (dedicated vs shared)

Q5 (model update frequency?)
  → whether TRT is right or torch.compile is better for the cadence
  → timing cache strategy
  → whether original/ weights stay in bundle

Q6 (GPU hardware?)
  → number of engine builds
  → Model Bundle structure
  → future GPU migration plan

Q7 (quantization acceptable?)
  → precision flag in build config
  → calibration data pipeline
  → accuracy validation ownership
```

---

## What Happens Without This Contract

```
No profile discussion:
  Serving team guesses opt → wrong tactic selection → silent perf bug
  Engine built for wrong batch size → 2× slower than it could be

No SLO discussion:
  fp32 engine deployed → latency 3× over SLO → emergency rebuild
  Quantized engine deployed without accuracy check → model owner discovers
  accuracy regression in production

No GPU discussion:
  A100 engine deployed on H100 fleet → runtime error, engine won't load
  H100 fleet expansion in 6 months → original/ not preserved →
  must re-export ONNX from model owner (weeks of coordination)

No update frequency discussion:
  Model updates 3× per day, TRT rebuild takes 45 min each →
  serving pipeline is always running stale engine
```

Every question maps to a specific failure mode if skipped.

---

## Interview Checklist

- Serving contract is a cross-team document: model owner provides use case knowledge, serving team provides infra/compiler knowledge
- opt is the most critical value to get right — it's the shape tactic selection benchmarks at; wrong opt is a silent performance bug with no error thrown
- p75 of traffic distribution is usually the right opt — tuned for the majority, max covers the tail
- Multiple profiles when client populations have shapes far apart — not just "we have dynamic shapes"
- Latency SLO drives precision choice, batch ceiling, CUDA graph decision, GPU allocation
- Model update frequency determines whether TRT is the right compiler at all
- GPU hardware drives number of builds and Model Bundle structure — engine is hardware-locked
- Quantization is a joint decision — model owner must validate accuracy, serving team runs calibration
- Without this contract: silent perf bugs, runtime errors, emergency rebuilds, stale engines in production
