# Model Compilation in the Serving Pipeline

How TRT-LLM compilation fits into the post-training serving workflow, what it writes
into the Model Bundle, and how the inference server uses the compiled artifacts.

Related: `model_bundle.md`, `format_conversion.md`, `model_optimization_techniques.md`,
`inference_server_startup.md`

For deep dives on TRT/TRT-LLM internals:
  → `ML_Compiler/knowledge_base/tensorrt.md`
  → `ML_Compiler/knowledge_base/tensorrt_llm.md`

---

## Where Compilation Fits in Online Inference Setup

Online inference setup has three steps. Compilation is entirely within Step 2.

```
Step 1 — Serving Metrics Collection
  Deploy base model (original/ safetensors) to staging via vLLM
  Benchmark TTFT p99, throughput, GPU utilization
  Write results back to bundle as eval_scores
  Push to Model Registry (baseline numbers)

Step 2 — Model Optimization               ← ALL compilation happens here
  2a. Format conversion:  convert_checkpoint.py
      reads:   inference_model/original/rank0.safetensors
      writes:  inference_model/trtllm_checkpoint/
                 rank0.safetensors  (QKV-fused, TP-sharded)
                 rank1.safetensors  (if TP > 1)
                 config.json        (TRT-LLM architecture config)

  2b. Quantization (optional): FP8 or INT4-AWQ
      modifies: trtllm_checkpoint/rank0.safetensors
      (re-runs if needed, writes quantized weights in-place)

  2c. trtllm-build:
      reads:   inference_model/trtllm_checkpoint/
      writes:  inference_model/trtllm_engine/
                 rank0.engine  (compiled binary, hardware-locked to build GPU)
                 rank1.engine  (if TP > 1)

  2d. Quality gate: MMLU drop validation
      If accuracy drop > threshold → fail the optimization job, don't write engine

Step 3 — Capacity Planning
  Calculate replicas from throughput (from Step 1 benchmarks) and peak QPS SLO
  Apply 1.3× safety buffer
  Set HPA limits
  Write Kubernetes deployment config
```

The key constraint from Step 2: **the engine must be built on the same GPU class
that production serves on**. An H100 engine built on an A100 test host won't run
in an H100 production cluster (sm_90 cubin won't execute on sm_86).

---

## Model Bundle Structure — TRT Path

The Model Bundle is the shared contract between the optimization pipeline and the
inference server. After Step 2, the bundle has both original weights AND compiled artifacts:

```
model_bundle/
├── bundle_metadata.pb          ← SINGLE entry point; fields set by optimization step
|
├── inference_model/
|  ├── model_metadata.pb
|  |
|  ├── original/               ← written by modeler at creation, never touched by optimization
|  |  ├── rank0.safetensors    (HuggingFace layout, FP32 or BF16)
|  |  └── rank1.safetensors    (if TP > 1 for original weights)
|  |
|  ├── trtllm_checkpoint/      ← written by format conversion (Step 2a)
|  |  ├── rank0.safetensors    (QKV-fused layout, TP-sharded, one file per GPU rank)
|  |  ├── rank1.safetensors
|  |  └── config.json          (TRT-LLM architecture config)
|  |
|  ├── trtllm_engine/          ← written by trtllm-build (Step 2c)
|  |  ├── rank0.engine         (compiled binary, hardware-locked)
|  |  └── rank1.engine         (one engine file per GPU rank)
|  |
|  ├── onnx/                   ← written by format conversion (ONNX path, alternative)
|  |  └── model.onnx           (compute graph + weights, hardware-agnostic)
|  |
|  └── tokenizer/              ← written by modeler, never modified
|     ├── tokenizer.json
|     └── tokenizer_config.json
|
├── preprocessing_model/       ← optional, TorchScript
|  ├── model_metadata.pb
|  └── model.pt
|
└── postprocessing_model/      ← optional, TorchScript
   ├── model_metadata.pb
   └── model.pt
```

The `original/` directory is always preserved. Even when `trtllm_engine/` exists,
the serving system can always fall back to vLLM loading from `original/` — for example
if the engine's `gpu_model` doesn't match the deployment host.

---

## bundle_metadata.pb Fields Set by Optimization

The optimization step (Step 2) updates `bundle_metadata.pb` to tell the inference
server how to use the compiled artifacts:

```protobuf
message BundleMetadata {
  // Which format to serve from — set by optimization step
  ArtifactFormat artifact_format = 1;
  // Values:
  //   SAFETENSORS   → serve from original/; vLLM loads weights directly
  //   TRTLLM_ENGINE → serve from trtllm_engine/; TRT-LLM runtime deserializes .engine

  // Hardware the engine was built on — must match deployment GPU
  string gpu_model = 2;              // "H100_80GB", "A100_40GB"
  string sm_architecture = 3;        // "sm_90", "sm_86"

  // Build parameters baked into the engine (cannot be changed without rebuild)
  TRTLLMBuildConfig trt_build_config = 4;
  message TRTLLMBuildConfig {
    int32 max_batch_size = 1;
    int32 max_input_len  = 2;
    int32 max_output_len = 3;
    int32 tp_size        = 4;
  }

  // Docker image used to build — CUDA toolkit + TRT-LLM version must match at serve time
  // Mandatory for TRTLLM_ENGINE artifacts
  string docker_image = 5;   // "nvcr.io/nvidia/trtllm:v0.12.0-cuda12.4"

  // Environment snapshot
  EnvironmentSnapshot env_snapshot = 6;
  message EnvironmentSnapshot {
    string cuda_version   = 1;   // "12.4"
    string trtllm_version = 2;   // "0.12.0"
    string torch_version  = 3;   // "2.3.0"
  }

  // Eval results written by Step 1 benchmarking
  EvalScores eval_scores = 7;
  message EvalScores {
    float ttft_p99_ms     = 1;
    float throughput_tps  = 2;
    float gpu_utilization = 3;
    float mmlu_accuracy   = 4;   // quality gate check
  }
}
```

The `docker_image` field is mandatory for `TRTLLM_ENGINE` artifacts because benchmarking
results are meaningless without locking the environment. (Incident: Python 3.10 vs 3.11
asyncio changes caused throughput numbers to differ significantly for the same engine.)

---

## Inference Server Startup: vLLM Path vs TRT-LLM Path

The inference server reads `bundle_metadata.pb.artifact_format` first and takes
a completely different startup path:

```
SAFETENSORS path (vLLM):
  1. Download bundle from blob store
  2. Read original/rank0.safetensors → load weights to GPU memory
  3. Build vLLM's cuda graph cache:
       iterate over (batch_size × seq_len) combinations
       capture CUDA graph for each combination
       This takes 2–10 minutes on first startup
  4. Ready to serve

  First-token latency: variable (CUDA graph re-capture on new shapes)
  Startup time: 2–10 min (CUDA graph build dominates)

─────────────────────────────────────────────────────────────────────

TRTLLM_ENGINE path (TRT-LLM):
  1. Download bundle from blob store
  2. Verify: gpu_model matches physical host, sm_architecture matches
     If mismatch → crash early with clear error (better than a CUDA illegal instruction)
  3. Verify: docker_image matches current container
     If mismatch → warn (soft check; hard block if major version differs)
  4. Deserialize trtllm_engine/rank{rank}.engine
       → allocate GPU memory for weights (weights already quantized+packed in engine)
       → load cubin blobs for all tactics
       → allocate activation buffers (sized at build time)
  5. Initialize TRT-LLM runtime (KV block pool, scheduler queue)
  6. Ready to serve

  First-token latency: deterministic (engine compiled for max_batch_size + max_seq_len)
  Startup time: 30–90 seconds (deserialization only, no runtime compilation)
```

The startup time difference matters operationally. vLLM's CUDA graph build time means
auto-scaling (new pod coming up) is slow. TRT-LLM's deterministic startup enables
faster scale-out events.

---

## What trtllm-build Bakes In (And Why It Matters for Interviews)

These compile-time constants are not changeable without rebuilding the engine.
Knowing them is important because interviewers will ask "what if you need to change X?"

```
max_batch_size:
  Bounds the in-flight batching scheduler's upper limit.
  Sets sizes of activation buffers (attention QKV tensors, FFN intermediates).
  "If we need more throughput" → rebuild with larger max_batch_size → different engine.

max_input_len:
  Bounds prefill phase KV cache allocation.
  "If users send longer prompts" → rebuild with larger max_input_len → different engine.
  Must be declared honestly: prompts longer than max_input_len are rejected at runtime.

max_output_len:
  Bounds per-sequence KV cache growth during decode.
  "If we need longer outputs" → rebuild.

tp_size:
  NCCL AllReduce communication topology baked into the engine.
  An engine built for tp_size=2 cannot run on 1 GPU or 4 GPUs.
  Changing GPU count → rebuild. This is why tp_size appears in bundle_metadata.pb.
```

**Interview talking point:** "We store both original/ safetensors and trtllm_engine/ in the
bundle precisely because these are compile-time constants. If product changes max output
length, we don't need to re-run training or fine-tuning — just re-run the optimization
step (Step 2) on the existing original/ weights to produce a new engine."

---

## Fallback Strategy: TRTLLM_ENGINE → SAFETENSORS

The inference server supports a graceful fallback when the engine is incompatible:

```
Startup check:
  if bundle_metadata.gpu_model != host.gpu_model:
    log WARN "Engine built for {bundle.gpu_model}, host is {host.gpu_model}"
    log WARN "Falling back to SAFETENSORS path (vLLM)"
    artifact_format = SAFETENSORS

  if bundle_metadata.sm_architecture != host.sm_architecture:
    → same fallback

  if engine deserialization throws CUDA_ERROR_INVALID_DEVICE:
    → same fallback
```

This is why `original/` is never deleted from the bundle even after the engine is built.
The engine is a performance optimization on top of the original weights, not a replacement.

---

## The ONNX Path (Alternative to TRT-LLM)

The bundle also has `onnx/model.onnx` — a hardware-agnostic representation:

```
original/ safetensors
    ↓ (via torch.onnx.export or optimum)
onnx/model.onnx      ← compute graph + weights, no GPU-specific code
    ↓ (via TRT's ONNX parser at runtime)
TRT engine (built at pod startup, not offline)
```

The ONNX path trades the offline build step for a per-pod JIT engine build on startup.
Useful for:
- Hardware fleet diversity (same ONNX can build engines on H100 and A100)
- Models where TRT-LLM plugins don't yet exist
- Smaller models where build time is seconds, not hours

The tradeoff: engine build happens on every pod start (no pre-built binary in bundle).
For large LLMs, this is impractical (60+ min build time on first request).

---

## Interview Checklist for Uber Discussion

**When you say "we add TRT engines to the Model Bundle" — here's the anticipated follow-ups:**

"What is a TRT engine?"
→ A compiled binary containing architecture-specific CUDA kernel code (cubin), packed
  weights, fused layer graph, and tactic selections. It runs the model as fast as possible
  on a specific GPU but cannot be transferred to a different GPU architecture.

"Why can't you just use the original safetensors at serving time?"
→ You can (vLLM path). But the engine has already done all the work at build time:
  kernel fusion, tactic selection, quantization packing. At inference you just execute
  the optimal kernels, no re-selection. vLLM must do CUDA graph capture on startup and
  may miss fusions that TRT finds through exhaustive tactic timing.

"What happens if your H100 fleet needs to swap to A100?"
→ The engine is hardware-locked. We'd rebuild the engine from trtllm_checkpoint/ on A100.
  The build job re-runs trtllm-build with --tp_size and capacity parameters for the new
  hardware. Original weights are untouched. The new engine goes into a new bundle version.

"What is baked into the engine?"
→ max_batch_size, max_input_len, max_output_len, tp_size, GPU architecture (sm_90 etc.),
  CUDA/TRT-LLM version, selected kernel implementations for every layer.

"How long does it take to rebuild an engine?"
→ 20–60 minutes for a 7B model. Scales with model size and number of fused layers to time.
  A timing cache from a prior build reduces repeat builds to ~5 minutes.

"What if product wants to increase max output length?"
→ Change the trtllm-build flag and re-run the optimization step. No retraining needed.
  Takes 20–60 minutes. New engine version registered in Model Registry.

"Why is docker_image mandatory in bundle metadata?"
→ Benchmarking numbers (TTFT p99, throughput) are meaningless without a locked environment.
  We learned this when a Python 3.10 → 3.11 asyncio change made the same engine appear
  20% faster in benchmarking than in prod because asyncio overhead differed.
