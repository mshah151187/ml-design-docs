# GPT vs Llama Family — Architecture Differences

Both are decoder-only transformers. Same fundamental idea: attend over past tokens,
predict the next one. But Llama introduced several improvements that are now standard
across all modern LLMs (Mistral, Gemma, Qwen, Phi — all use the same Llama-style stack).

---

## 1. Scale and Vocabulary

| | GPT-2 | Llama 3.1 8B |
|---|---|---|
| Parameters | 117M | 8B |
| Vocab size | 50,257 | 128,256 |
| Embedding dim | 768 | 4096 |
| Context window | 1,024 tokens | 128,000 tokens |
| Training data | ~40 GB | ~15 TB |

GPT-2 was a 2019 research model. Llama 3.1 is a 2024 production model.
The vocab size difference matters for inference — larger vocab = fewer tokens per sentence
= fewer forward passes for the same text. Llama 3.1 uses the same tokenizer as GPT-4.

---

## 2. Attention: MHA vs GQA

### GPT-2: Multi-Head Attention (MHA)

Every query head has its own dedicated key and value head:
```
Query heads:  Q1  Q2  Q3  Q4  Q5  Q6  Q7  Q8  Q9  Q10 Q11 Q12
              |   |   |   |   |   |   |   |   |   |   |   |
Key heads:    K1  K2  K3  K4  K5  K6  K7  K8  K9  K10 K11 K12
Value heads:  V1  V2  V3  V4  V5  V6  V7  V8  V9  V10 V11 V12
```

KV cache stores K and V for every head at every token position.
KV cache size = num_layers × 2 × num_heads × head_dim × seq_len × dtype_bytes

### Llama 3.1: Grouped Query Attention (GQA)

Multiple query heads share one K and V head:
```
Query heads:  Q1  Q2  Q3  Q4 | Q5  Q6  Q7  Q8 | ... (32 total)
                    \   /           \   /
Key heads:          K1              K2            (8 total)
Value heads:        V1              V2            (8 total)
```

4 query heads share 1 KV head. KV cache is 4× smaller than MHA.
This is why in llama_worker.py NUM_KV_HEADS=8, not 32.

**Why GQA?** At inference time, the bottleneck is reading KV cache from GPU HBM
(High Bandwidth Memory). Smaller KV cache = less HBM bandwidth consumed per token
= higher throughput, especially at long sequence lengths.

**Quality impact**: near-zero. Queries still have full expressivity (32 heads).
Only K and V are shared, and in practice the model learns to encode what it needs
in 8 KV heads.

---

## 3. Position Encoding: Learned vs RoPE

### GPT-2: Learned Absolute Position Embeddings

Each position (0, 1, 2, ..., 1023) has a learned embedding vector added to the token
embedding before attention. The model sees "I am at position 5" as a fixed vector.

Problem: the model has never seen position 1024+ during training → breaks on longer
sequences. The 1024 token limit in GPT-2 is hard — not just a recommendation.

### Llama: RoPE (Rotary Position Embeddings)

**Simple explanation:**

Instead of adding a position vector to the token, RoPE encodes position by
*rotating* the query and key vectors before the dot-product attention step.

Think of it this way. Imagine Q and K as arrows in 2D space. RoPE rotates arrow Q
by angle (position × θ) and rotates arrow K by the same rule. When you compute
Q·K (dot product = how much they point in the same direction), the result naturally
depends on the *difference* in their positions, not their absolute positions.

```
Without RoPE:
  Q at position 5:  some vector
  K at position 3:  some vector
  Q·K = some number  (model has to learn what position 5 vs 3 means)

With RoPE:
  Q at position 5:  original vector rotated by 5θ
  K at position 3:  original vector rotated by 3θ
  Q·K = depends on rotation difference (5-3)θ = 2θ
        → model naturally sees RELATIVE distance between tokens
```

**Why it matters for inference:**

- Generalizes to longer sequences — rotating by a larger angle is mathematically
  well-defined. With training tricks (YaRN, rope scaling), Llama 3.1 reaches 128K tokens.
- No extra parameters — no embedding table to store. Pure math at attention time.
- Captures relative distance — "these two tokens are 10 apart" is more useful than
  "this token is at position 105 and that one is at 95".

**What changes in code:** RoPE is applied inside each attention layer to Q and K
before the dot product. No changes to the embedding table.

---

## 4. Normalization: LayerNorm vs RMSNorm

### GPT-2: LayerNorm (post-attention)

Applied AFTER the attention and FFN blocks. Normalizes using mean and variance:
```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) × γ + β
```
Requires computing both mean and variance. Has learnable scale (γ) and bias (β).
Applied at the OUTPUT of each sublayer (post-norm).

### Llama: RMSNorm (pre-attention)

Applied BEFORE the attention and FFN blocks (pre-norm). Normalizes using only
root mean square — no mean subtraction, no bias:
```
RMSNorm(x) = x / sqrt(mean(x²) + ε) × γ
```

**Simple explanation:**

LayerNorm asks: "How far is each value from the average? Rescale so they're all
within a similar range."

RMSNorm asks: "How big are these values overall? Rescale so their overall magnitude
is 1." It skips the mean subtraction step entirely.

In practice: RMSNorm is ~10-15% faster (fewer operations), equally stable,
and produces the same quality. The mean subtraction in LayerNorm turns out not
to matter much for transformer training.

**Pre-norm vs post-norm:** Llama normalizes the INPUT to each sublayer (pre-norm).
This is more numerically stable at large scale — gradients don't blow up during
training. GPT-2's post-norm can cause instability at large model sizes.

---

## 5. Feed-Forward Network: GELU vs SwiGLU

### GPT-2: Standard FFN with GELU

```
FFN(x) = Linear_down( GELU( Linear_up(x) ) )

x (768) → Linear_up → (3072) → GELU activation → Linear_down → (768)
```

GELU is a smooth activation function (similar to ReLU but curves near 0).
Two weight matrices: up-projection and down-projection.

### Llama: SwiGLU (Swish-Gated Linear Unit)

**Simple explanation of gating:**

Imagine you have information flowing through a pipe. A gate controls how much
of that information passes through. In SwiGLU, the network learns TWO things
simultaneously:
1. What information to extract (the "content" path)
2. How much of that information to let through (the "gate" path)

These are multiplied together:
```
SwiGLU(x) = (Linear_gate(x) × SiLU activation) × Linear_up(x)
              └──── gate: how much to open? ────┘   └── content ──┘
```

Then passed through a down-projection back to the original dimension.

```
FFN(x):
  gate   = SiLU( W_gate × x )    ← learns what to suppress or amplify
  content =       W_up   × x     ← learns the actual transformation
  FFN(x) = W_down( gate × content )
```

**Why gating helps:**

Without a gate: every neuron's output flows through equally. The model can only
scale and shift values (via the activation function).

With a gate: the model can selectively suppress entire dimensions of information
for a given input. "For this type of token, ignore these features entirely." This
is more expressive — the network learns not just what to amplify but what to ignore.

SiLU (Sigmoid Linear Unit, also called Swish) is the activation applied to the
gate: `SiLU(x) = x × sigmoid(x)`. It's smooth, non-monotonic (can go slightly
negative), and empirically better than GELU.

**Cost:** SwiGLU uses 3 weight matrices instead of 2. But Llama reduces the hidden
dimension so total FLOPs stay similar.

---

## 6. Full Comparison Table

| Feature | GPT-2 | Llama 3.1 8B | Why it matters |
|---|---|---|---|
| Params | 117M | 8B | quantization has measurable impact |
| Attention | MHA (12 KV heads) | GQA (8 KV heads / 32 Q heads) | 4× smaller KV cache |
| Position | Learned absolute | RoPE | generalizes to 128K context |
| Normalization | LayerNorm, post | RMSNorm, pre | faster + more stable |
| FFN activation | GELU | SwiGLU | gating = more expressive |
| Default dtype | FP32 | BF16 | half memory baseline |
| KV cache / token | ~144 bytes | ~128 KB (BF16, 32L, 8H, 128D) | GQA keeps it manageable |

---

## 7. What This Means for Inference and Quantization

**KV cache** is the bottleneck at long sequences. GQA directly addresses this.
For 64 slots × 512 seq_len on Llama 3.1 8B: ~4 GB KV cache.
Same setup with MHA (32 KV heads) would need ~16 GB.

**Quantization impact:**
- GPT-2 (117M, FP32 ~0.5GB): INT8 saves ~250MB — negligible on an A100.
- Llama 3.1 8B (BF16 ~16GB): INT8 → ~8GB, INT4 → ~4GB — measurable, real savings.

**RoPE + pre-norm:** these are now in essentially every modern LLM. If you see a
new model (Mistral, Gemma, Qwen, Phi), assume RoPE + RMSNorm + SwiGLU + GQA
unless the paper says otherwise. GPT-style MHA + learned positions + post-norm
is legacy (GPT-2, early GPT-3).
