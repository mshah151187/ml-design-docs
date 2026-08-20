# Neural Network Basics — ML Theory Interview Questions

---

## Q: Why does a neural network learn more with backpropagation than gradient descent?

This is a terminology clarification worth making sharp.

**Backpropagation is not an alternative to gradient descent — it's how you compute the gradients that gradient descent needs.**

Gradient descent says: "move each weight in the direction that reduces loss." It doesn't say *how* to compute those directions. That's backprop's job.

### The two options for computing gradients

**Option 1 — Numerical gradient (no backprop):**
```
∂L/∂w ≈ (L(w + ε) - L(w)) / ε
```
Perturb one weight, run a full forward pass, measure how loss changed. Repeat for every weight. GPT-3 has 175 billion weights → 175 billion forward passes per update step. Completely infeasible.

**Option 2 — Backpropagation:**

Chain rule applied layer by layer, backwards. One forward pass to get loss, one backward pass to get ALL gradients simultaneously. O(1) passes regardless of how many parameters.

```
∂L/∂w_1 = ∂L/∂w_n · ∂w_n/∂w_{n-1} · ... · ∂w_2/∂w_1
```

Each layer's gradient depends on the next layer's gradient — backprop propagates this signal backwards through the whole network in one shot.

### Why do neural nets "learn more" then?

That's not backprop — that's **depth + nonlinearity**. Backprop just makes it *feasible* to train deep networks. The actual representational power comes from stacking nonlinear transformations:

```
pixels → edges → shapes → objects → semantics
```

Each layer learns increasingly abstract features. A linear model trained with gradient descent directly can only learn linear decision boundaries — no matter how long you train it, you can't separate XOR with a line. There's nothing backprop can do about that because there's only one layer and no depth to propagate through.

### Summary

| | Role |
|---|---|
| Gradient descent | The optimizer — move weights in the direction that reduces loss |
| Backpropagation | Efficient gradient computation — one backward pass for ALL weights via chain rule |
| Depth + nonlinearity | What gives neural nets representational power — learns hierarchical, abstract features |

**Backprop doesn't make the network smarter — it makes training deep networks computationally feasible.**
