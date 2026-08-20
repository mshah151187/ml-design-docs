# Transformers — ML Theory Interview Questions

---

## Q: Help me understand the attention mechanism.

### 1. Start with the problem it solves

Before attention, sequence models (RNNs/LSTMs) processed tokens one by one and compressed the entire history into a single fixed-size hidden state vector. Two problems:

- **Bottleneck:** the whole past must fit in one vector. Long sequences lose early context.
- **Vanishing gradients:** gradients flow backwards through every time step and shrink, so the model can barely learn dependencies between tokens far apart.

Attention solves this by **letting every token look directly at every other token** — no bottleneck, no sequential dependency.

---

### 2. The core intuition: Query, Key, Value

Think of attention as a **soft dictionary lookup**.

In a real dictionary: you give an exact key, you get the exact value back.
In attention: you give a query, and instead of matching one key exactly, you **blend all values weighted by how similar your query is to each key**.

```
Q = "what information do I need?"        (the asker)
K = "what information do I offer?"       (the label on each entry)
V = "what is my actual content?"         (the value behind each entry)

For each token:
  - compute similarity of its Q against all K's
  - softmax those similarities → weights that sum to 1
  - return the weighted sum of V's
```

Each token gets a new representation = a weighted blend of all other tokens' content, where the weights reflect relevance.

---

### 3. The math

Given a sequence of tokens, we project each token's embedding into three vectors:

```
Q = X W_Q        (B, T, d_k)
K = X W_K        (B, T, d_k)
V = X W_V        (B, T, d_v)
```

Then compute attention:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V
```

Step by step:
- `Q Kᵀ` → (T, T) matrix where entry [i, j] = dot product of query_i and key_j = "how much should token i attend to token j?"
- `/ √d_k` → scale down to prevent dot products from growing too large in high dimensions (which would push softmax into near-zero gradient regions)
- `softmax(...)` → normalize each row to sum to 1 → attention weights
- `· V` → weighted sum of value vectors → new representation for each token

---

### 4. Why does scaling by √d_k matter?

With d_k dimensions, dot products have variance ≈ d_k (assuming Q and K have unit-variance entries). Large dot products push softmax into saturation — one entry gets weight ≈ 1, all others ≈ 0. The gradient of softmax near saturation is nearly zero → no learning. Dividing by √d_k brings variance back to 1.

---

### 5. Multi-Head Attention

Instead of one set of (Q, K, V) projections, run H sets in parallel:

```
head_h = Attention(X W_Q_h,  X W_K_h,  X W_V_h)     for h = 1..H
output = Concat(head_1, ..., head_H) · W_O
```

**Why multiple heads?**
One attention head can learn one type of relationship (e.g., "verb agrees with subject"). Multiple heads learn different relationships simultaneously — syntax, coreference, proximity — and the model combines them via W_O. Each head operates on a d_k = d_model / H subspace, so total compute stays the same.

---

### 6. Causal (Masked) Self-Attention — for language models

In a language model we predict the next token, so token at position t should only see positions 0..t-1. We enforce this with a **causal mask**: set all upper-triangle entries of the score matrix to -∞ before softmax → they become 0 after softmax → token t pays zero attention to future tokens.

```
scores[i, j] = -inf   if j > i     (future token — mask out)
scores[i, j] = Q_i · K_j / √d_k   if j <= i   (past/present — keep)
```

---

### 7. Complexity

Self-attention is **O(T² · d)** in time and memory — every token attends to every other token, so the attention matrix is T×T. This is the dominant cost for long sequences (why 100K+ context models need tricks like FlashAttention or sparse attention).

---

### 8. In our code

```python
# model.py — MultiHeadAttention.forward()

Q = self.W_q(x).view(B, T, H, d_h).transpose(1, 2)   # (B, H, T, d_h)
K = self.W_k(x).view(B, T, H, d_h).transpose(1, 2)
V = self.W_v(x).view(B, T, H, d_h).transpose(1, 2)

scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_h)   # (B, H, T, T)
scores = scores.masked_fill(mask == 0, float("-inf"))        # causal mask
attn   = F.softmax(scores, dim=-1)
out    = attn @ V                                            # (B, H, T, d_h)
out    = out.transpose(1, 2).contiguous().view(B, T, d_model)
return self.W_o(out)
```

Single projection per head dimension (not H separate linears) — reshape + transpose splits into heads efficiently.

---

### Summary (one sentence per concept)

| Concept | One line |
|---|---|
| Why attention | Lets every token directly see every other token — no bottleneck, no sequential dependency |
| Q / K / V | Query = what I'm looking for; Key = what I offer; Value = what I actually give |
| Score | Dot product Q·K measures relevance; softmax normalizes to weights |
| √d_k scaling | Prevents dot products from saturating softmax and killing gradients |
| Multi-head | H parallel attention heads learn different relationship types simultaneously |
| Causal mask | Mask future positions to -∞ so the model can't cheat on next-token prediction |
| Complexity | O(T² · d) — the T×T attention matrix is the bottleneck for long contexts |
