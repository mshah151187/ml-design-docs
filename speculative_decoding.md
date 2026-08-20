# Speculative Decoding

## The Problem It Solves

LLM decode is autoregressive — one token at a time, each step depends on the previous.
At small batch sizes this is purely memory-bandwidth-bound (GEMV): the GPU reads all model
weights from HBM to produce a single token. The weights are read, compute is tiny, and the
GPU sits mostly idle.

```
Normal decode — batch=1, 7B model:
  Step 1: read 14 GB weights → produce 1 token → 10ms
  Step 2: read 14 GB weights → produce 1 token → 10ms
  Step 3: read 14 GB weights → produce 1 token → 10ms
  ...
  100 tokens = 1000ms
  GPU is memory-bound every step — SMs mostly idle
```

The GPU is wasting compute capacity because it can only generate one token per step.
Speculative decoding exploits that idle compute to generate multiple tokens per step.

---

## Core Idea

Use a small fast **draft model** to speculatively generate K candidate tokens ahead.
Then run the large **verifier model** on all K tokens in one parallel forward pass.
If the verifier agrees with the draft, accept all K tokens in one step. If it disagrees
at position i, discard tokens i..K and keep 0..i-1.

```
Normal decode:
  large model → token 1 (10ms)
  large model → token 2 (10ms)
  large model → token 3 (10ms)
  large model → token 4 (10ms)
  large model → token 5 (10ms)
  5 tokens = 50ms

Speculative decode (K=5, draft accepts at 80%):
  draft model → [t1, t2, t3, t4, t5] in ~2ms   (tiny model, fast)
  large model → verify all 5 in one step = 10ms (parallel GEMM, not sequential GEMV)
  accepted:    4 tokens (80% of 5)
  4 tokens in ~12ms ≈ 3ms/token vs 10ms/token → 3.3× faster effective ITL
```

---

## Why Verification is Nearly Free

This is the key insight. The verifier's job is to compute the probability distribution
over the vocabulary for each of the K positions — conditioned on all prior tokens.
This is exactly what a normal forward pass does, just with K input tokens instead of 1.

```
Normal decode step:  input = [last token]         → shape (batch, 1)    → GEMV
Verification step:   input = [K draft tokens]      → shape (batch, K)    → GEMM

GEMV (batch=1, seq=1):  read 14 GB, compute 14B FLOPs  = 1 FLOP/byte   → slow
GEMM (batch=1, seq=5):  read 14 GB, compute 70B FLOPs  = 5 FLOP/byte   → faster
```

The weight read cost is the same. The compute cost scales with K. So verifying K tokens
costs only slightly more than verifying 1 token — and if all K are accepted, you've
produced K tokens for roughly the cost of 1.

---

## Acceptance Rate and Effective Speedup

Not all K draft tokens will be accepted. The draft model is smaller and less accurate.

```
acceptance_rate α = P(draft token matches verifier)

Effective tokens per step = α × K + (1 - α^K) / (1 - α)   [exact formula]

Simplified approximation:
  If α = 1.0 (perfect draft): K tokens per step → K× speedup
  If α = 0.8, K=5:            ~4 tokens per step → ~4× speedup
  If α = 0.5, K=5:            ~2 tokens per step → ~2× speedup
  If α = 0.0 (random draft):  ~1 token per step  → no speedup (but no slowdown either)
```

Crucially: **speculative decoding never makes output worse than normal decoding.**
The verifier always has the final say. Any rejected token is replaced by the verifier's
own sample — output distribution is identical to running the verifier alone.

---

## When Speculative Decoding Helps

**Best conditions:**
- Small batch (≤ 32): GPU is memory-bound on normal decode, so the draft cost is
  negligible and verification compute is free headroom
- High acceptance rate: draft and verifier agree often (same model family, similar domain)
- Latency-critical serving: user-facing, ITL SLA matters

**Hurts or is neutral when:**
- Large batch: GPU is already compute-bound during normal decode. Verification now
  competes for SMs with the existing batch. Draft overhead exceeds gain.
- Low acceptance rate: many rejections → frequent rollbacks → near-zero net speedup
- Draft model is too large: draft latency approaches verifier latency → no benefit

```
batch=1,   α=0.8, K=5 → ~3.3× ITL improvement     ← sweet spot
batch=256, α=0.8, K=5 → ~0× improvement (or worse) ← avoid
```

---

## Draft Model Options

| Draft source | How it works | Tradeoff |
|-------------|-------------|----------|
| **Smaller model (same family)** | Llama-3.1-1B drafts for Llama-3.1-70B | High α if same training; needs separate model loaded in memory |
| **Draft head on verifier** | Lightweight MLP head on top of verifier's hidden states predicts next K tokens | No separate model; slightly lower α than a full draft model |
| **Self-speculation (MEDUSA)** | Multiple decoding heads on the verifier itself, each predicts +1, +2, ..., +K tokens | No separate model; parallel heads; lower α for further positions |
| **N-gram draft** | Look up last N tokens in a table, propose continuation | Near-zero cost; only works when output repeats patterns (code, templates) |

---

## Speculative Decoding in vLLM

vLLM supports speculative decoding natively:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    speculative_model="meta-llama/Llama-3.1-1B",  # draft model
    num_speculative_tokens=5,                       # K
)

outputs = llm.generate("Tell me about LLM inference", SamplingParams(max_tokens=200))
```

vLLM handles: draft generation, batch verification, token tree sampling, rollback on rejection.

---

## Connection to Our Inference_Fundamentals Project

In our pipeline (LlamaWorker → KVStore → decode loop):

```
Normal decode step:
  read KV from store → build decode batch → model(input_ids=[last_token]) → 1 token

With speculative decoding:
  draft model:  model_small(input_ids=[last_token]) × K times → K candidate tokens
  verifier:     model_large(input_ids=[K candidates]) → K logits in 1 forward pass
  accept/reject each position → write accepted KV back to store
```

The KV store must store KV for the draft tokens speculatively and roll back on rejection.
vLLM's PagedAttention handles this via copy-on-write pages — draft tokens get tentative
pages that are freed on rollback without copying the main sequence's pages.

---

## Summary

```
What it attacks:   ITL (inter-token latency) — effective tokens per decode step
When to use:       Small batch, latency-critical, high draft acceptance rate
When to avoid:     Large batch, low acceptance rate, throughput-primary workloads
Speedup:           1.5–3× ITL reduction in practice (α ≈ 0.7–0.8, K=4–5)
Output quality:    Identical to running verifier alone — no accuracy tradeoff
Key insight:       Verification of K tokens costs ≈ verification of 1 token
                   (same weight read, slightly more compute — still memory-bound)
```
