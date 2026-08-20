# Transformer Block Architecture — ML Theory Interview Questions

---

## Q: Walk me through a transformer block. Why does each piece exist?

### 1. The two variants — post-norm vs pre-norm

**Original Transformer (Vaswani 2017) — post-norm:**
```
Input x
  │
  ├─► Multi-Head Attention ──► Dropout ──┐
  │                                      ▼
  └──────────────────────────► Add → LayerNorm → x'
                                              │
  ┌───────────────────────────────────────────┘
  │
  ├─► FFN ──► Dropout ──┐
  │                     ▼
  └──────────► Add → LayerNorm
                          │
                        Output
```

**GPT-2 style (what we implemented) — pre-norm:**
```
Input x
  │
  ├─► LayerNorm ──► Multi-Head Attention ──► Dropout ──┐
  │                                                    ▼
  └────────────────────────────────────────► Add → x'
                                                   │
  ┌────────────────────────────────────────────────┘
  │
  ├─► LayerNorm ──► FFN ──► Dropout ──┐
  │                                   ▼
  └───────────────────────► Add
                              │
                            Output
```

In code (`TransformerBlock.forward`):
```python
x = x + self.drop(self.attn(self.norm1(x), mask))   # pre-norm: norm BEFORE attn
x = x + self.drop(self.ff(self.norm2(x)))            # pre-norm: norm BEFORE ffn
```

**Pre-norm is more stable.** With post-norm, the unnormalized residual sum can be large at initialization, causing unstable early training. Pre-norm normalizes *before* the sublayer — gradients are better behaved and training doesn't need a warmup as long.

---

### 2. Residual Connections — why Add?

**What it does:**
```
output = x + sublayer(x)
```
Instead of replacing x, the sublayer learns a *delta* on top of the original signal.

**Problem it solves — vanishing gradients in depth:**

Without residuals, in a 96-layer network (GPT-3), gradients must flow through 96 matrix multiplications. Each one can shrink the gradient by a factor < 1. By layer 1, the gradient is effectively zero — lower layers learn nothing.

With residuals, the gradient can bypass any sublayer entirely:
```
∂L/∂x = ∂L/∂output · (1 + ∂sublayer/∂x)
```
The `1` term is a direct gradient highway — even if `∂sublayer/∂x` vanishes, ∂L/∂x stays ≥ ∂L/∂output. Deep networks can now be trained.

**The "skip learning" intuition:**

At initialization, `sublayer(x) ≈ 0` (small random weights). So:
```
output ≈ x + 0 = x
```
A deep network at init is approximately an identity function — a reasonable starting point. The model only needs to learn *corrections*, not reconstruct the full transformation from scratch.

**Residuals also let layers specialize.** Each block doesn't have to carry all information forward — it only learns what it needs to add. This is why you can remove individual layers from a trained GPT without catastrophic failure.

---

### 3. LayerNorm — why normalize?

**What it does:**

For each token position, normalize across the feature dimension (d_model):
```
LayerNorm(x) = (x - μ) / (σ + ε)  ·  γ  +  β
```
where μ, σ are computed over the d_model features of *that one token*, and γ, β are learned scale + shift parameters.

**Problem it solves — internal covariate shift:**

As training updates the weights of block 1, the distribution of activations it outputs changes. Block 2 was trained for the *old* distribution — it now receives inputs in a range it wasn't expecting. This instability compounds across all 96 layers.

LayerNorm re-centers and re-scales activations to a stable distribution before each sublayer sees them. Each sublayer gets a normalized input no matter what happened earlier in the network.

**Why LayerNorm and not BatchNorm?**

| | BatchNorm | LayerNorm |
|---|---|---|
| Normalizes over | batch dimension (all items in batch, one feature at a time) | feature dimension (all features, one item at a time) |
| Problem for LLMs | sequences have variable length; at inference batch size = 1, statistics are unstable | works on a single token position regardless of batch size |
| Good for | CNNs with fixed-size inputs | transformers, RNNs |

BatchNorm needs a full batch to compute meaningful statistics. At inference time (generating one token at a time), batch size = 1 → statistics are garbage. LayerNorm computes per-token statistics — works identically during training and inference.

**The γ, β parameters:**

After normalization, the network might *want* to restore some spread (e.g., if a large variance is useful for the next sublayer). γ (scale) and β (shift) are learned per-feature parameters that let the model undo normalization if needed. This makes LayerNorm expressive — it can learn to be approximately a no-op.

---

### 4. Dropout — why randomly zero activations?

**What it does:**

During training, randomly set each activation to 0 with probability `p` (typically 0.1 for transformers). Scale surviving activations by `1/(1-p)` so the expected sum is preserved.

**Problem it solves — co-adaptation:**

Without dropout, a neuron can learn to "rely" on another specific neuron always being active with a certain value. If unit A always fires when unit B fires, they co-adapt. Remove either one (e.g., during a different batch) and the network breaks.

Dropout forces each unit to be useful *on its own* — it can't count on its neighbors being there. This prevents brittle memorization and improves generalization.

**The ensemble intuition:**

Each training step uses a different random mask → a different "thinned" sub-network. Training with dropout is approximately training an exponential number (2^n where n = neurons) of models simultaneously and averaging their predictions at inference (when dropout is off). Ensemble methods generally generalize better than single models.

**Where exactly is dropout applied in our block?**

Dropout is applied at **2 levels**:

**Level 1 — attention weight dropout** (inside `MultiHeadAttention`): applied to the T×T softmax score matrix *before* multiplying by V. Zeroes out some token-to-token connections entirely — that token's V embedding doesn't contribute to the weighted sum at all. Prevents the model from over-relying on specific token pairs.

**Level 2 — sublayer output dropout** (inside `TransformerBlock`): applied to the final output of each sublayer (attention or FFN) *before* the residual add. Zeroes out some feature dimensions of the output vector. Prevents co-adaptation across the full model.

```python
# Level 1 — in MultiHeadAttention.forward():
attn = F.softmax(scores, dim=-1)   # (B, H, T, T) — weights per token pair
attn = self.attn_drop(attn)        # ← zero out some token-to-token connections
out  = attn @ V                    # dropped tokens don't contribute to the blend

# Level 2 — in TransformerBlock.forward():
x = x + self.drop(self.attn(self.norm1(x), mask))
#              ↑ zero out some feature dims of the blended output, then residual add

x = x + self.drop(self.ff(self.norm2(x)))
#              ↑ same after FFN
```

**Dropout is OFF at inference.** `model.eval()` disables it. The full network is used for generation.

---

### 5. Why does the FFN exist at all? Wasn't attention enough?

Attention is a **routing mechanism** — it decides *which* tokens to mix, then returns a weighted average of value vectors. The output is a linear combination of existing representations.

The FFN is where **new computation** happens. Each token's representation passes through:
```
FFN(x) = GELU( x W_1 ) W_2
```
- `W_1`: d_model → 4·d_model (expand — gives the network room to compute)
- GELU: nonlinearity (without this, two linear layers collapse to one)
- `W_2`: 4·d_model → d_model (project back)

This is a per-token MLP — same weights applied independently at every position. It's where the model "thinks" about each token's content after attention has mixed the context.

Research on GPT-2 shows that factual knowledge (e.g., "Paris is the capital of France") is stored in FFN weights, not attention weights. Attention retrieves context; FFN supplies knowledge.

---

### 6. Putting it all together — information flow per block

```
x (B, T, d_model)
│
├─ norm1 ──► attn  ──► dropout  ──┐     ← attention enriches each token
│                                 ▼        with context from all other tokens
└────────────────────────────► + x'   ← residual: preserve original signal
                                   │
├─ norm2 ──► ffn   ──► dropout  ──┐     ← FFN applies per-token computation
│                                 ▼        (stores/retrieves knowledge)
└────────────────────────────► + x''  ← residual: preserve signal again
                                   │
                                output (B, T, d_model) — same shape as input
```

Each block outputs the same shape it receives. N blocks are stacked — each one adds a layer of refinement to every token's representation.

---

### Summary table

| Component | Where | Why |
|---|---|---|
| Residual (Add) | Around both sublayers | Gradient highway through depth; skip learning from identity init |
| LayerNorm | Before each sublayer (pre-norm) | Stabilizes activation distribution; works at batch size 1 |
| Pre-norm vs post-norm | GPT-2 uses pre-norm | More stable at init; no warmup needed |
| Dropout (on output) | After attn and FFN, before residual | Prevents co-adaptation; ensemble effect |
| Dropout (on attn weights) | On T×T attention matrix | Prevents reliance on specific token pairs |
| FFN | After attention | Nonlinear per-token computation; where factual knowledge lives |
| GELU in FFN | Activation between two linears | Nonlinearity prevents two linears from collapsing |
