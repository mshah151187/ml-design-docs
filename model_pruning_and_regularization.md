# Model Pruning and Regularization

---

## 1. The Core Tension: Bias vs Variance

```
Model too simple → underfitting (high bias)
  Can't capture the real pattern.
  Training error high, test error high.

Model too complex → overfitting (high variance)
  Memorizes training data, fails on new data.
  Training error low, test error high.
```

**Concrete example:** 10 data points on a curve.
- Degree-9 polynomial: passes through all 10 exactly — zero training error, wildly oscillates between points.
- Degree-2 polynomial: misses some points but captures the actual trend and generalizes.

**What memorization looks like mechanically:**
The model allocates parameters to remember specific training examples rather than shared structure.
A large weight matrix can encode "if I see this exact input configuration, output this" — essentially
a lookup table. Fits training perfectly, fails on anything slightly different.

---

## 2. Regularization Techniques

All regularization techniques constrain the model's **effective capacity** below its nominal parameter count.

| Technique | Mechanism |
|---|---|
| L2 (weight decay) | Penalizes large weights — forces smooth, low-amplitude functions |
| L1 | Encourages sparsity — some weights go exactly to zero |
| Dropout | Randomly zeros neurons during training — forces redundant representations, no single neuron can memorize |
| Early stopping | Stops before the model has enough gradient steps to memorize noise |
| Data augmentation | Makes memorizing specific examples harder by perturbing them |

**Why L2 works:** adding `λ × ||W||²` to the loss means large weights are penalized at every step.
The optimizer is forced to spread signal across many small weights rather than concentrating it in a few
large ones — concentrating capacity is how memorization happens.

**Why dropout works:** at each forward pass, a random 50% of neurons are zeroed. No single neuron can
reliably encode "memorized example #47" because it might not be present. Forces the network to build
redundant, distributed representations that generalize.

---

## 3. The Modern Twist — Double Descent

Classical ML says test error is a U-curve: decreases as model grows (less bias), then increases (overfitting):

```
Error
  │   \          /
  │    \        /
  │     \      /     ← classical: more params = worse after a point
  │      \____/
  └──────────────── model size
```

But in modern deep learning, past the interpolation threshold (where the model can perfectly fit
training data), test error **decreases again**:

```
Error
  │   \       /\
  │    \      /  \
  │     \    /    \____
  │      \__/          ← second descent: very large models generalize well again
  └──────────────────── model size
       interpolation
       threshold
```

**Why?** SGD has an implicit bias — when there are many solutions that fit the training data
(overparameterized regime), gradient descent tends to find the **minimum-norm solution**. That is a
form of automatic regularization. The model finds the "simplest" interpolation of the data, which
generalizes.

This is why GPT-4 doesn't just memorize — at that scale, the optimizer naturally finds solutions
that capture structure rather than noise.

**Key insight:** "too large = memorize" is the classical view, valid in the under-overparameterized
transition zone. In the deeply overparameterized regime (modern LLMs), implicit SGD regularization
takes over and large models generalize well.

---

## 4. Model Pruning

### Unstructured Pruning (weight-level)

Set individual weights to zero based on a criterion (smallest magnitude, smallest gradient).

```
Dense:                    Sparse (unstructured):
[0.3  0.8  0.1]          [0.3   0    0  ]
[0.5  0.2  0.9]    →     [ 0   0.2  0.9 ]
[0.1  0.7  0.4]          [ 0   0.7   0  ]
```

**Why sparse matrices are bad for GPU computation — two scenarios:**

#### Scenario A — Dense storage (zeros kept in the FP16 matrix)

You zero out 50% of weights but keep the same matrix shape in memory.
The GPU still loads the entire matrix from HBM (same memory traffic) and still runs every
multiply-accumulate on tensor cores (zero × input = 0, still executed). Result: 0% latency
improvement. Not harmful, just useless — same speed, smaller checkpoint file on disk.

#### Scenario B — Sparse storage (CSR/COO format) + sparse kernels

To actually reduce memory traffic, you must store only non-zeros plus their indices.

**CSR (Compressed Sparse Row) format — concrete example:**

```
Dense matrix (stored as 9 FP16 values, row-major):
  [0.3   0    0  ]
  [ 0   0.2  0.9 ]
  [ 0   0.7   0  ]

Memory layout (dense):  [0.3, 0.0, 0.0, 0.0, 0.2, 0.9, 0.0, 0.7, 0.0]
                         ↑ contiguous, predictable, one coalesced HBM read per row

CSR layout (4 non-zeros):
  values:      [0.3,  0.2, 0.9,  0.7]   ← only non-zero values
  col_indices: [ 0,    1,   2,    1 ]   ← column of each value
  row_ptr:     [ 0,    1,   3,    4 ]   ← where each row starts in values[]
                row 0 has 1 non-zero (values[0..0])
                row 1 has 2 non-zeros (values[1..2])
                row 2 has 1 non-zero  (values[3..3])
```

**COO (Coordinate) format — same matrix:**
```
  row_indices: [0,   1,   1,   2 ]
  col_indices: [0,   1,   2,   1 ]
  values:      [0.3, 0.2, 0.9, 0.7]
```
COO stores every non-zero as a (row, col, value) triple. Simpler but larger index overhead than CSR.

**Why this creates irregular memory access — the GPU compute pattern:**

Dense GEMM multiplies matrix row i by input vector x:
```
output[i] = W[i,0]*x[0] + W[i,1]*x[1] + W[i,2]*x[2]

Memory reads: W row i is contiguous → one coalesced HBM load → entire cache line used
              x is read sequentially → prefetcher works perfectly
```

Sparse GEMM with CSR must do:
```
for each non-zero j in row i:
    output[i] += values[j] * x[col_indices[j]]
                              ↑ col_indices[j] is ARBITRARY
                              → random lookup into x at unpredictable offsets
                              → HBM cache miss per non-zero element
```

For row 1 above: reads `x[1]` then `x[2]` — fine here, but at ML scale with thousands of
non-zeros per row scattered at random column positions, each `x[col]` lookup is a cache miss.

**Three compounding problems:**

1. **Irregular memory access** — random col_index lookups destroy HBM locality.
   Dense GEMM reads x sequentially; sparse GEMM reads x at scattered offsets → 10–100× more cache misses.

2. **No tensor cores** — tensor cores do 8×4 dense tile multiply in one instruction.
   They require contiguous, regularly-laid-out data. cuSPARSE (the sparse kernel library)
   cannot use tensor cores for unstructured sparsity → falls back to scalar/SIMD ops.
   This alone is a 10–16× throughput loss vs tensor core GEMM.

3. **Warp divergence** — a GPU warp (32 threads) runs in lockstep. Different rows have
   different numbers of non-zeros. Threads processing short rows sit idle while threads
   processing long rows finish → warp serialization.

**Empirical breakeven:** cuSPARSE beats cuBLAS (dense) only above ~95% sparsity for
typical ML matrix sizes. Pruning usually reaches 50–90% → sparse kernels are slower
than just running the dense matrix with its wasted zero multiplications.

#### The one hardware exception — NVIDIA 2:4 structured sparsity

A100/H100 tensor cores have native hardware support for exactly 2 non-zeros out of every
4 consecutive weights. This pattern is regular enough to be handled natively:
```
[0.3,  0,  0.9,  0]   ← 2 non-zeros in positions {0,2} out of 4
  ↓ compressed to: values=[0.3, 0.9], indices=[0, 2]
  ↓ tensor core handles the decompression in silicon → 2× throughput vs dense
```
Unstructured 50% sparsity gets none of this. You'd need to enforce the 2:4 pattern during
training (SparseGPT, magnitude pruning with 2:4 constraint) to unlock hardware acceleration.

### Structured Pruning

Remove entire units — neurons, attention heads, or full layers. Result is a **smaller dense matrix**.

```
FFN: d_model → 4d → d_model     →    d_model → 2d → d_model
     (remove 50% neurons)
```

Smaller dense matmul = real speedup on GPU. This is what actually reduces latency.

**What can be pruned structurally:**
- **MLP neurons:** remove rows/columns from FFN weight matrices (intermediate dim shrinks)
- **Attention heads:** remove entire heads — key, query, value projections for that head
- **Layers:** remove entire transformer blocks

### Pruning Criteria

| Criterion | How |
|---|---|
| Magnitude | Remove smallest absolute-value weights/neurons — simple, no backprop needed |
| Taylor expansion | Approximate loss increase from removing a unit: `\|gradient × weight\|` |
| Activation frequency | Remove neurons that rarely activate across training data |
| Sensitivity | Measure actual loss increase when each unit is removed (expensive but accurate) |

### Post-Pruning Fine-Tuning

After removing units, the model degrades. Fine-tuning redistributes the signal across
surviving weights. With fewer parameters, the model can no longer memorize noise —
the remaining capacity is forced toward general patterns.

---

## 5. Why Pruned Models Can Exceed the Original

**The Lottery Ticket Hypothesis** (Frankle & Carlin, 2019):
Inside any large network there are smaller "winning ticket" subnetworks that, when identified
and fine-tuned, match or exceed the full network.

The dense model's capacity is split across:
```
Dense 600M:  [useful patterns] + [redundant neurons] + [memorized noise]
                    ↓ prune + fine-tune
Pruned 375M: [useful patterns] ← better generalization
```

Pruning removes redundant and noisy capacity. Fine-tuning forces the surviving weights to encode
more general patterns — there is no longer enough capacity to memorize noise.

**The regularization view:** pruning + fine-tuning is one of the strongest regularization techniques
because it hard-constrains the number of parameters, not just penalizes them.

---

## 6. LinkedIn's OSSCAR — Structured Pruning of 0.6B SLM Ranker

Applied to the 0.6B student SLM in LinkedIn's Semantic Search system.

### Two-pronged structured pruning:

**1. MLP neuron pruning — 50% of neurons removed per FFN block:**
```
Before: Linear(H → 4H) → GELU → Linear(4H → H)
After:  Linear(H → 2H) → GELU → Linear(2H → H)
```
Every FFN block halved. Dense matmuls are now half the size.

**2. Layer pruning — final 8 transformer layers removed:**
Early layers learn syntax and basic semantics. Final layers specialize for the task but are often
the most redundant for cross-encoder ranking — the model consolidates useful representations in
middle layers for classification. Removing the last 8 layers also eliminates KV cache for those layers entirely.

**Result after fine-tuning:**

| Model | P@10 | NDCG@10 |
|---|---|---|
| Dense 600M | 0.5369 | 0.8629 |
| Pruned 375M | 0.5434 | **0.8652** |

Pruned model slightly exceeds dense baseline — pruning removed redundant/noisy capacity,
fine-tuning forced remaining weights toward better generalization.

**Throughput impact (combined with context summarization):**
290 → 2,200 items/s/GPU (7.5×). Summarization and pruning are multiplicative:
shorter context = faster per-pass, smaller model = cheaper per-pass.

---

## 7. Is Pruning a Form of Regularization?

They share the same mechanism — reducing model complexity — but differ in **goal** and **timing**.

```
Regularization:   applied DURING training
                  modifies the loss function or training dynamics
                  primary goal: generalization (prevent overfitting)

Pruning:          applied AFTER training
                  directly removes parameters from a trained model
                  primary goal: compression and inference efficiency
```

**The connection — L1 is soft pruning, magnitude pruning is hard pruning:**

```
L1 regularization (soft pruning):
  Adds λ|w| penalty to loss during training
  Gradients push small weights toward zero throughout training
  Weights shrink but rarely reach exactly zero
  → model learns sparse-ish representations during training

Magnitude pruning (hard pruning):
  After training: zero out all weights below threshold τ
  Hard decision applied once, post-hoc
  → same sparsity effect but enforced after the fact
```

L1 encourages sparsity continuously through the training process.
Magnitude pruning enforces it in one shot after training is complete.
Same destination, different paths.

**Why pruning is NOT regularization in the strict sense:**
- Does not modify the training objective or loss function
- Does not affect gradient updates during training
- A pruned model can still overfit if fine-tuned without regularization
- Generalization improvement (if any) is a side effect, not the design goal

**Where the line blurs — pruning-aware training:**
Some approaches train with an iteratively updated sparsity mask, progressively
zeroing weights during training. This does shape training dynamics similarly to
regularization — the constraint is applied during, not after. But the goal
remains compression, not generalization.

**Summary:**

| | Regularization | Pruning |
|---|---|---|
| When applied | During training | After training |
| Modifies loss | Yes (L1/L2 penalty, dropout) | No |
| Primary goal | Generalization | Compression / inference speed |
| Generalization effect | Direct — by design | Indirect — side effect |
| Example | L1 → soft sparsity | Magnitude pruning → hard sparsity |

L1 ≈ regularization that causes pruning-like behavior.
Pruning ≈ compression that has regularization-like effects.

---

## 8. Interview Summary

**Q: Why does a larger model sometimes generalize worse?**
Classical bias-variance tradeoff: excess capacity gets allocated to memorizing training noise
rather than learning patterns. Regularization constrains effective capacity to prevent this.

**Q: Why do modern LLMs not just memorize despite being huge?**
Double descent: deeply overparameterized models benefit from implicit SGD regularization —
gradient descent finds the minimum-norm solution among all ways to fit the data, which generalizes.

**Q: Why can a pruned model match or beat the original?**
Lottery ticket hypothesis: large networks contain winning-ticket subnetworks. Pruning removes
redundant/harmful capacity, fine-tuning forces remaining weights to encode general patterns.
Fewer parameters = less room to memorize noise.

**Q: Why structured pruning over unstructured?**
Unstructured pruning zeros individual weights. In dense storage: same latency, just wasted
zero-multiplications. In sparse storage (CSR/COO): actually slower — random col-index lookups
destroy HBM locality, cuSPARSE cannot use tensor cores, and warp divergence serializes threads.
Sparse kernels only beat dense above ~95% sparsity; pruning lands at 50–90%.
Structured pruning (neurons/heads/layers removed) produces smaller dense matrices — tensor cores
work normally, memory access is contiguous, no index overhead. Direct proportional speedup.
