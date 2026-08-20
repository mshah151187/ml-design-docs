# Model Optimization Techniques

After safety evaluation passes, before the model enters the registry, you run
the full optimization stack. The goal is not just "meet the SLA" — it is
**minimize cost per token** while holding quality within an acceptable degradation
threshold.

Even if the baseline (BF16) already meets p99 latency SLA, optimized models:
- consume less GPU VRAM → fit more concurrent sequences → higher throughput
- run fewer HBM reads per token → lower TPOT
- cost less per request at scale (fewer GPUs for same QPS)

---

## The Optimization Stack

Three independent layers. Each improves a different bottleneck:

```
1. Quantization
   Reduce weight precision: BF16 → FP8 / INT8 / INT4
   Benefit: less VRAM, more sequences in flight, faster matmuls on hardware
            with native low-precision tensor cores (H100 FP8, A100 INT8)

2. Compilation + Kernel Fusion
   Fuse sequences of ops into fewer CUDA kernels
   Benefit: fewer HBM round-trips between ops, less kernel launch overhead

3. Static Graph Capture (CUDA Graphs)
   Record GPU command sequence once per input shape, replay on real requests
   Benefit: eliminates CPU-side kernel launch overhead at decode time
```

Applied in order because quantization changes the compute graph — compile
after quantizing so the compiler sees the final op types.

---

## Quantization Tools

### `llm-compressor` (Neural Magic / Red Hat)

**What it is:** The current standard library for post-training quantization of
HuggingFace models. Produces a compressed checkpoint that vLLM loads directly
with no extra conversion step.

**Supported algorithms:**

| Algorithm | Precision | What it does |
|-----------|-----------|-------------|
| **FP8** | 8-bit float | Per-tensor or per-channel scale; H100 native tensor cores; near-zero quality loss |
| **GPTQ** | INT4 / INT8 | Hessian-based PTQ; minimizes per-layer reconstruction error |
| **AWQ** | INT4 | Activation-aware; identifies and preserves outlier channels before quantizing |
| **SmoothQuant** | INT8 (W8A8) | Migrates activation outliers into weights so both are quantizable |

**How to run (FP8 example):**

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",   # per-token dynamic activation scaling
    ignore=["lm_head"],
)

oneshot(
    model="meta-llama/Llama-3-8B-Instruct",
    recipe=recipe,
    output_dir="llama3-8b-fp8",
)
```

**Load in vLLM:**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model llama3-8b-fp8 \
    --quantization fp8
```

No separate conversion step. vLLM reads the llm-compressor checkpoint natively.

**When to use:** default choice for H100. FP8 is the lowest-risk optimization —
near-zero quality degradation, ~2× memory reduction, native hardware support.

---

### `AutoAWQ`

**What it is:** AWQ (Activation-aware Weight Quantization) implementation.
Better quality than GPTQ at the same bit-width because it identifies outlier
activation channels and scales weights around them before quantizing, rather than
treating all channels equally.

**Key insight:** ~1% of weight channels cause ~80% of quantization error. AWQ
finds those channels by looking at activation magnitudes, not just weight values.

**How to run:**

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

model.quantize(tokenizer, quant_config={
    "zero_point": True,
    "q_group_size": 128,   # per-group quantization: 128 weights share one scale
    "w_bit": 4,
    "version": "GEMM",
})

model.save_quantized("llama3-8b-awq-int4")
```

**When to use:** INT4 target on A100 (no native FP8), or when INT8 memory
reduction is insufficient and you need to fit a larger model on fewer GPUs.
AWQ INT4 vs GPTQ INT4: AWQ generally loses 0.5–1% less on MMLU.

---

### `bitsandbytes`

**What it is:** NF4 and INT8 quantization, tightly integrated with HuggingFace
`transformers`. The standard library for QLoRA training. Less common in production
serving — primarily a training and research tool.

**How to use:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4: better for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
)
```

**When to use:** QLoRA fine-tuning (training on quantized base model). For
production serving, prefer llm-compressor or AutoAWQ — bitsandbytes loads
quantized weights on-the-fly rather than saving a pre-quantized checkpoint,
which adds startup overhead and is less portable.

---

### TensorRT-LLM (NVIDIA)

**What it is:** NVIDIA's full-stack LLM optimization library. Does quantization
AND compilation in one step — produces a compiled TRT engine rather than a
weight checkpoint. Highest performance ceiling on NVIDIA hardware.

**What it does:**
- INT8, FP8, INT4 weight quantization
- Ahead-of-time kernel compilation and fusion
- Hardware-specific kernel selection per GPU model
- Multi-GPU tensor parallelism baked into the engine

**How to run:**

```bash
# Convert HuggingFace checkpoint to TRT-LLM format
python convert_checkpoint.py \
    --model_dir meta-llama/Llama-3-8B \
    --output_dir trt_ckpt/ \
    --dtype float16

# Build TRT engine (this is the slow step: minutes to hours)
trtllm-build \
    --checkpoint_dir trt_ckpt/ \
    --output_dir trt_engine/ \
    --gemm_plugin float16 \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 512
```

**What it tells you:** After building, the engine is a fixed compiled artifact
tied to a specific GPU model, batch size, and sequence length. Changing any of
these requires a rebuild.

**When to use:** maximum throughput requirement on a fixed, known hardware target
(e.g. always H100 SXM5). Trade-off: build time is high, engine is not portable
across GPU generations, debugging is harder than with vLLM.

**vs llm-compressor:** llm-compressor produces a portable checkpoint (runs on
any vLLM-supported hardware); TRT-LLM produces a hardware-locked compiled engine
with higher peak performance but less flexibility.

---

## Compilation and Kernel Fusion Tools

### `torch.compile`

**What it is:** PyTorch's graph compiler. Traces the model's compute graph, fuses
operations, and generates optimized CUDA kernels via Triton. JIT — runs on first
forward pass, serves optimized kernels on all subsequent calls.

**How to use:**

```python
import torch

model = load_model()   # standard HuggingFace load

# Compile with max-autotune: slower first call, fastest steady state
model = torch.compile(model, mode="max-autotune")

# First forward pass: tracing + compilation (slow)
output = model(input_ids)

# Subsequent calls: compiled kernel (fast)
output = model(input_ids)
```

**Compilation modes:**

| Mode | Compile time | Speedup |
|------|-------------|---------|
| `default` | fast | moderate |
| `reduce-overhead` | moderate | good for small models |
| `max-autotune` | slow (minutes) | best steady-state performance |

**What it does internally:**
- TorchDynamo: traces Python bytecode to capture compute graph
- TorchInductor: lowers graph to Triton kernels (GPU) or C++ (CPU)
- Fuses pointwise ops (add, relu, layernorm) into single kernels
- Eliminates redundant memory reads between fused ops

**When to use:** first optimization to try — zero model changes, significant
speedup (10–30% on decode, higher on prefill), easy to enable/disable.

---

### FlashAttention

**What it is:** A fused attention kernel that rewrites the standard
`softmax(QK^T)V` computation to avoid materializing the full N×N attention
matrix in HBM.

**Standard attention memory access pattern:**
```
Q, K, V in HBM
→ load Q, K → compute QK^T (N×N matrix) → write to HBM
→ load QK^T → compute softmax → write to HBM
→ load softmax, V → compute output → write to HBM
Total HBM reads/writes: O(N²)
```

**FlashAttention memory access pattern:**
```
Tile Q, K, V into SRAM (on-chip, fast)
→ compute attention tile-by-tile entirely in SRAM
→ write only the final output to HBM
Total HBM reads/writes: O(N)  ← eliminates the N×N intermediate
```

Result: attention is now **memory-bandwidth-bound on the right quantity** —
loading model weights, not the attention matrix.

**Versions:**
- FlashAttention-2: better parallelism, ~2× faster than FA-1
- FlashAttention-3: H100-specific, uses async Tensor Core pipelines

**How it's used in practice:** vLLM, TGI, TRT-LLM all use FlashAttention
internally. You don't call it directly — enabling it is just a flag:

```bash
# vLLM uses FlashAttention by default when flash-attn is installed
pip install flash-attn
# vLLM auto-detects and uses it
```

**When to use:** always, for any attention-heavy LLM workload. It is the single
highest-impact kernel fusion for transformer serving.

---

### `xFormers`

**What it is:** Meta's library of memory-efficient attention and other
transformer building blocks. Includes its own fused attention implementation
and additional fused ops (layer norm + linear, etc.).

Used as an alternative to FlashAttention in some frameworks (earlier versions
of HuggingFace transformers defaulted to xFormers before FA-2 was standard).

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",  # or "eager", "sdpa"
)
```

`sdpa` (scaled dot-product attention) = PyTorch's built-in fused attention
(dispatches to FlashAttention or a fallback depending on hardware).

---

## CUDA Graph Capture

### vLLM Built-in CUDA Graph Capture

**What it is:** CUDA Graphs record the entire GPU command sequence for a fixed
input shape. On real requests, the recorded graph is replayed — skipping the
CPU-side overhead of kernel launch, argument binding, and dispatch per step.

**Why it matters for LLM decode:** decode runs one forward pass per token.
Each forward pass is a short burst of GPU kernels. Without CUDA graphs, the
CPU re-launches all kernels every step. At high token rates, CPU overhead
becomes a meaningful fraction of total TPOT.

```
Without CUDA Graphs (decode step):
  CPU: launch kernel 1 → GPU runs → CPU: launch kernel 2 → GPU runs → ...
  CPU overhead visible between GPU kernels

With CUDA Graphs:
  CPU: replay graph (single call) → GPU runs entire step from recorded commands
  CPU overhead: near zero
```

**Bucketing — the key constraint:**

CUDA Graphs require fixed input shapes. Sequence length varies per request.
Solution: capture graphs for a fixed set of shapes (buckets), pad requests to
the nearest bucket at serve time.

```
vLLM captures graphs for:
  batch_size × [1, 2, 4, 8, 16, 32, 64, 128, 256, ...]  (num tokens per step)

At serve time:
  5 tokens in decode step → pad to 8 → replay batch=8 graph
  9 tokens in decode step → pad to 16 → replay batch=16 graph
```

Padding wastes a few token slots but the speedup from graph replay far
outweighs the padding cost.

**How it's configured in vLLM:**

```bash
# CUDA graphs enabled by default in vLLM
# To disable (for debugging):
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B \
    --enforce-eager    # disables CUDA graphs → 20–40% slower decode

# Control memory reserved for captured graphs:
    --gpu-memory-utilization 0.90   # 90% of GPU VRAM for model + KV cache + graphs
```

**What it tells you:** if TPOT is higher than expected and GPU utilization
looks good during prefill but drops during decode, CUDA graph capture is likely
not active (check `--enforce-eager` flag or graph capture failure logs).

---

### `torch.cuda.graph` (PyTorch API)

**What it is:** The underlying PyTorch API that vLLM uses internally. Useful
when building a custom serving loop outside vLLM.

```python
import torch

# Warm-up run (must run before capture)
with torch.no_grad():
    output = model(static_input)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay on real request
static_input.copy_(real_input)   # fill the static input buffer
g.replay()                        # runs recorded command sequence
result = static_output.clone()    # read output before next replay
```

**When to use:** only if building custom inference code outside of vLLM/TRT-LLM.
In practice, vLLM's built-in capture handles this.

---

## Quality Gate After Optimization

Every optimization step requires a quality regression check before the model
moves to the registry. The check is identical across all optimizations:

```
Optimized model
      │
      ├── lm-evaluation-harness
      │   Compare: optimized_score vs baseline_score
      │   Gate: degradation < threshold per task
      │     FP8:      MMLU drop < 0.5%   (very tight — FP8 is near-lossless)
      │     INT8-AWQ: MMLU drop < 1.0%
      │     INT4-AWQ: MMLU drop < 2.0%
      │     INT4-GPTQ: MMLU drop < 3.0%
      │
      ├── genai-perf sweep
      │   Compare: optimized TTFT/TPOT/throughput vs baseline
      │   Gate: serving metrics improved (or at minimum not regressed)
      │
      └── PASS → model registry with optimization metadata recorded
          FAIL → either accept higher degradation threshold (needs sign-off)
                 or choose less aggressive quantization format
```

---

## Summary — What Each Tool Answers

| Question | Tool |
|----------|------|
| Quantize to FP8 with minimal quality loss? | `llm-compressor` (FP8_DYNAMIC) |
| Best INT4 quality on A100? | `AutoAWQ` |
| Quantize for QLoRA training base model? | `bitsandbytes` NF4 |
| Maximum throughput on fixed NVIDIA hardware? | `TensorRT-LLM` |
| General op fusion + kernel optimization? | `torch.compile` (max-autotune) |
| Eliminate N×N attention matrix HBM cost? | `FlashAttention-2/3` |
| Eliminate CPU kernel launch overhead at decode? | vLLM CUDA graph capture |
| Custom CUDA graph in non-vLLM serving code? | `torch.cuda.graph` |

## Format + Hardware Decision Matrix

```
H100  + LLM serving  → FP8 (llm-compressor) + FlashAttention-3 + CUDA graphs
A100  + LLM serving  → AWQ INT4 or INT8-SmoothQuant + FlashAttention-2 + CUDA graphs
Any   + max perf     → TensorRT-LLM (if fixed hardware, willing to pay build cost)
Any   + flexibility  → llm-compressor checkpoint + vLLM (portable, easier to update)
Training / fine-tune → bitsandbytes NF4 (not for serving)
```
