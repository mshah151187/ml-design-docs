# Post-Training Serving Metrics

After a model is trained, before it is deployed, you must answer:
**"Can this model serve production traffic within our latency and cost SLAs?"**

This requires systematic benchmarking — not a single measurement, but a structured
evaluation across hardware, load patterns, and optimization configurations. This doc
covers what benchmarking is done, which tools are used, and what each tool tells you.

---

## The Three Layers of Evaluation

```
Layer 1 — Quality        Does the model produce correct outputs?
                         (offline benchmarks: MMLU, task F1, ROUGE...)

Layer 2 — Serving        Can it serve requests within latency/throughput SLAs?
                         (load testing: concurrent requests, percentile latencies)

Layer 3 — GPU Internals  How does model architecture affects GPU utilization?
                         (profiling: NCU, Nsight, torch.profiler)
```

Quality is measured first — no point benchmarking serving metrics on a model that
fails quality gates. Serving metrics are measured second. GPU profiling is done only
when serving metrics are outside target and you need to diagnose why.

---

## Key Metrics Defined

Before tools, define what you are measuring:

| Metric | Definition | Who cares |
|--------|-----------|-----------|
| **TTFT** (Time to First Token) | Time from request sent to first token received. Dominated by prefill phase. | User-perceived responsiveness |
| **TPOT** (Time Per Output Token) | Time between consecutive tokens during decode. | Streaming smoothness |
| **E2E Latency** | TTFT + TPOT × num_output_tokens | Total request duration |
| **Throughput** | Tokens generated per second across all concurrent requests | Cost efficiency |
| **QPS** (Queries Per Second) | Requests the system can handle per second | Capacity planning |
| **GPU Memory (HBM)** | Peak VRAM used: model weights + KV cache + activations | Determines max batch size |
| **GPU Utilization** | % of time GPU SM is doing useful compute | Efficiency signal |
| **MFU** (Model FLOP Utilization) | Actual FLOP/s ÷ theoretical peak FLOP/s | Hardware efficiency ceiling |
| **Queue Wait Time** | Time request waits before the model sees it | Scheduler health |
| **KV Cache Utilization** | % of KV cache pages in use | OOM risk indicator |

**Always measure percentiles, not means:**
- p50 = median (typical user)
- p95 = most users
- p99 = worst 1% — this is the SLA number
- Mean hides tail latency problems completely

---

## Type 1 — Single-Request Baseline (Exploratory)

**What it is:** Send one request at a time, measure latency of each stage.
No concurrency. Used to establish a lower-bound on latency and understand
where time is being spent before adding load.

**When used:** Early in evaluation, when profiling a new model or a new
optimization (quantization, TP degree). Also used in Inference Fundamentals
to build intuition.

**What it tells you:**
- Minimum achievable TTFT (no queue wait, just model compute)
- Prefill time vs decode time breakdown
- Per-token decode cost
- Memory footprint at different sequence lengths

**Limitation:** Does not reflect production behavior. A system can look fast
under single-request load and fall apart under concurrent load (queue buildup,
KV cache contention, memory pressure).

---

## Type 2 — Systematic Sweep (Pre-Deployment Benchmarking)

**What it is:** Run benchmarks across a grid of parameters to find where the
model saturates and what configuration maximizes throughput within SLA.

**Sweep dimensions:**

```
batch_size     : [1, 2, 4, 8, 16, 32, 64]
input_seq_len  : [128, 256, 512, 1024, 2048]
output_seq_len : [64, 128, 256, 512]
tp_degree      : [1, 2, 4, 8]           # tensor parallelism
quantization   : [FP16, INT8, INT4]
```

**Output:** a table of (TTFT_p99, TPOT_p99, throughput, GPU_mem) for every
combination. You read off the configuration that maximizes throughput while
staying within your latency SLA.

```
Example sweep output (illustrative):

  batch | seq_len | TTFT_p99 | TPOT_p99 | tok/s | GPU_mem
  ------+---------+----------+----------+-------+--------
    1   |   512   |   80ms   |   12ms   |  83   | 18 GB
    8   |   512   |  120ms   |   14ms   |  580  | 22 GB
   16   |   512   |  190ms   |   18ms   | 1020  | 28 GB   ← sweet spot
   32   |   512   |  380ms   |   28ms   | 1600  | 42 GB
   64   |   512   |  750ms   |   51ms   | 1800  | OOM
```

SLA: TTFT_p99 < 400ms, TPOT_p99 < 40ms → batch=16 is the operating point.

---

## Type 3 — Concurrent Load Testing (Realistic Traffic Simulation)

**What it is:** Simulate realistic production traffic — many requests arriving
simultaneously at a target QPS, sustained over minutes. Measures how the system
behaves under real load, not synthetic single-request conditions.

**Key difference from sequential benchmarking:**

```
Sequential (Type 1):
  req1 → wait → done → req2 → wait → done
  TTFT = model compute only

Concurrent (Type 3):
  req1, req2, ..., req50 arrive simultaneously
  TTFT = model compute + queue wait time
  As QPS approaches saturation, queue wait explodes → TTFT spikes
```

**What it tells you:**
- Maximum sustainable QPS before latency SLA is violated
- How latency degrades as load increases (the "knee" of the curve)
- Whether KV cache fills up over time and starts blocking new requests
- Whether memory fragmentation grows under sustained load

**Run duration:** minimum 10 minutes at target QPS. Short runs miss:
- KV cache filling effects (takes time to reach steady state)
- Memory leak or fragmentation accumulation
- Throughput variation as system warms up

---

## Tools

### vLLM `benchmark_serving.py`

**What it is:** The standard load testing tool for vLLM-served models.
Sends concurrent requests to a running vLLM server at a configurable QPS,
using a real dataset (ShareGPT by default for realistic prompt/output lengths).

**What it measures:** TTFT, TPOT, E2E latency (p50/p95/p99), throughput
(tokens/sec), request success rate.

**How to run:**

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --tensor-parallel-size 2

# Run benchmark at 20 QPS for 60 seconds
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3-8B \
    --dataset-name sharegpt \
    --request-rate 20 \               # QPS
    --num-prompts 1000 \
    --percentile-metrics ttft,tpot,e2e_latency
```

**Output:**

```
Throughput: 18.4 requests/s
Successful requests: 1000

Time to First Token (ms):
  Mean:   142.3
  P50:    118.4
  P95:    298.7
  P99:    412.1     ← this is your SLA check

Time Per Output Token (ms):
  Mean:    18.2
  P99:     34.6

Total tokens/s: 2847.3
```

**When to use:** pre-deployment gate, comparing two model versions,
validating that a new optimization (quantization, TP change) doesn't
regress latency at target QPS.

---

### NVIDIA `genai-perf`

**What it is:** NVIDIA's benchmarking tool, built on top of Triton Inference
Server. More systematic than benchmark_serving.py — designed explicitly for
sweeping across configurations.

**What it measures:** TTFT, TPOT, throughput, GPU memory, at configurable
concurrency levels and sequence lengths.

**How to run:**

```bash
genai-perf profile \
    --model meta-llama/Llama-3-8B \
    --service-kind openai \
    --endpoint-type chat \
    --concurrency 16 \
    --input-tokens-mean 512 \
    --output-tokens-mean 128 \
    --measurement-interval 10000    # milliseconds
```

**Output:** CSV with full percentile breakdown + GPU memory usage.
Good for sweep automation — wrap in a loop over concurrency levels,
collect CSV, plot the latency-vs-throughput curve.

**When to use:** systematic pre-deployment sweep, hardware comparison
(A100 vs H100), quantization comparison (FP16 vs INT8 vs INT4).

---

### `lm-evaluation-harness` (EleutherAI)

**What it is:** Standardized quality benchmark runner. NOT a serving
latency tool — this measures model output quality on known benchmarks.

**What it measures:** Accuracy on standardized tasks:
- MMLU (knowledge across 57 subjects)
- HellaSwag (commonsense reasoning)
- HumanEval (code generation)
- TruthfulQA (hallucination rate)
- GSM8K (math reasoning)

**How to run:**

```bash
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3-8B \
    --tasks mmlu,hellaswag,gsm8k \
    --num_fewshot 5 \
    --batch_size auto
```

**Output:**

```
Task      | Metric   | Value
----------+----------+-------
mmlu      | acc      | 0.682
hellaswag | acc_norm | 0.821
gsm8k     | exact_match | 0.714
```

**When to use:** quality gate after training, regression check after
quantization (does INT4 degrade MMLU below threshold?), comparing
fine-tuned vs base model.

---

### NVIDIA Nsight Systems (`nsys`)

**What it is:** System-level GPU profiler. Captures a timeline of everything
happening on CPU and GPU — CUDA kernel launches, memory copies, CPU-GPU
synchronization, idle gaps.

**What it measures:** Timeline view showing:
- Which CUDA kernels ran and for how long
- CPU-GPU synchronization points
- Memory transfer events (H2D, D2H)
- Idle time between kernels (pipeline bubbles)

**How to run:**

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output inference_profile \
    python scripts/inference_baseline.py
```

Opens in Nsight Systems GUI — timeline view shows CPU and GPU activity side by side.

**What it tells you:**
- Is the GPU idle between batches? (scheduling overhead)
- Are memory copies overlapping with compute? (async pipeline efficiency)
- Where are the pipeline bubbles in tensor/pipeline parallel execution?

**When to use:** when throughput is lower than expected and you need to see
whether the GPU is actually busy or waiting on CPU/memory work.

---

### NVIDIA Compute Utility (`ncu`)

**What it is:** Kernel-level GPU profiler. Goes one level deeper than nsys —
instead of a system timeline, it profiles individual CUDA kernels for
roofline analysis.

**What it measures:**
- Memory bandwidth utilization (HBM reads/writes vs peak)
- Compute utilization (FLOP/s vs peak)
- L1/L2 cache hit rates
- Warp stall reasons (why warps are not executing)
- Occupancy (active warps vs maximum)

**How to run:**

```bash
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_lookup_hit.sum \
    --target-processes all \
    python scripts/inference_baseline.py
```

**What it tells you:**
- Is this kernel compute-bound or memory-bandwidth-bound?
- Are we hitting the roofline ceiling? (if yes, kernel is optimal; if no, why not)
- Is there L1/L2 reuse or is every access going to HBM?

**Prefill is compute-bound (GEMM):** ncu should show high SM utilization,
lower memory bandwidth pressure.

**Decode is memory-bandwidth-bound (GEMV):** ncu should show high HBM
bandwidth utilization, lower SM utilization. This is expected — decode reads
all model weights per token but does little compute per byte.

**When to use:** when a specific kernel is slower than expected, or when
comparing a custom attention kernel vs FlashAttention.

---

### `torch.profiler`

**What it is:** PyTorch's built-in profiler. Easier to integrate than NCU —
instrument directly in Python code, no CLI wrapping needed.

**What it measures:** Per-operator timing (CPU + CUDA), memory allocation,
stack traces. Outputs to Chrome trace format (view in `chrome://tracing`)
or TensorBoard.

**How to use:**

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./prof_logs"),
) as prof:
    model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**Output (table):**

```
Name                    CPU time    CUDA time   Calls
----------------------  ----------  ----------  -----
aten::mm                1.2ms       18.4ms      48     ← matmul (dominant)
aten::scaled_dot_product_attention  0.4ms  6.2ms  24
aten::add               0.1ms       0.8ms       96
...
```

**When to use:** identify which PyTorch operators dominate runtime, debug
unexpected CPU time, compare operator-level impact of optimizations.

---

### `locust` / `k6` (General Load Testing)

**What it is:** General-purpose HTTP load testing tools, not ML-specific.
Used when the inference endpoint is a standard REST API and you want
fine-grained control over request arrival patterns.

**What they add over benchmark_serving.py:**
- Configurable arrival distributions (Poisson, bursty, ramp-up)
- Mixed workload simulation (short prompts + long prompts simultaneously)
- Web UI for real-time monitoring during the test
- Custom scenario scripting

**When to use:** integration testing of the full serving stack (gateway +
inference service + model backend) under realistic mixed traffic patterns,
not just the model server in isolation.

---

## Production Evaluation Pipeline

How these tools fit together in a real CI/CD gate:

```
Model checkpoint ready
        │
        ▼
Step 1: lm-evaluation-harness
        Quality gate: MMLU ≥ 0.68, task_f1 ≥ 0.82
        FAIL → block deployment
        │
        ▼
Step 2: genai-perf sweep (batch × seq_len × TP grid)
        Find max throughput configuration within SLA
        FAIL (no config meets SLA) → block deployment, investigate
        │
        ▼
Step 3: vLLM benchmark_serving.py at target QPS (10 min sustained)
        Gate: TTFT_p99 < 400ms, TPOT_p99 < 40ms, error_rate < 0.1%
        FAIL → block deployment
        │
        ▼
Step 4: Shadow deployment (24h on copy of live traffic)
        Compare output distributions and latency vs current production model
        FAIL → block deployment
        │
        ▼
Step 5: Canary (5% traffic, 48h)
        Monitor business metrics on canary slice
        │
        ▼
Full rollout
```

---

## What Each Tool Answers

| Question | Tool |
|----------|------|
| Is the model's output quality acceptable? | `lm-evaluation-harness` |
| What batch size / TP degree meets our SLA? | `genai-perf` sweep |
| Does the model hold up under concurrent load? | `vllm benchmark_serving.py` |
| Where is time being spent across the system? | `nsys` (timeline) |
| Is a specific kernel compute or memory bound? | `ncu` (roofline) |
| Which PyTorch operators dominate runtime? | `torch.profiler` |
| How does the full stack behave under mixed traffic? | `locust` / `k6` |
