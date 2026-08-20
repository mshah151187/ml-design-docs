# Transfer Learning — Battling Overfitting with Limited Data

## The Problem: Small Data + Large Model = Overfitting

A large neural network has millions of parameters. Training it from scratch requires
millions of examples — one to constrain each parameter. With limited data, the model
memorizes the training set instead of learning generalizable patterns.

```
10,000 training examples, 110M parameter model (BERT):
  Parameters >> Examples
  → model fits noise in training data
  → train loss: 0.01, val loss: 3.2  ← overfit
```

The naive fix — smaller model — throws away representational power. Transfer learning
keeps the power without needing the data.

---

## The Core Idea

A model pre-trained on a massive dataset has already learned general, reusable features.
You don't need to relearn them. You only need to learn what's specific to your task.

```
Pre-trained on 1B words (BERT):
  Already knows: grammar, syntax, word relationships, sentence structure,
                 common reasoning patterns, named entities

Your task: classify 5,000 customer support tickets into 8 categories
  Need to learn: what distinguishes "billing issue" from "technical problem"
                 in your specific domain vocabulary

Transfer: borrow the language understanding, learn only the classification boundary
```

The amount of data needed drops from "enough to learn everything" to "enough to learn
just the task-specific part."

---

## Why It Works: Layer Hierarchy

Neural networks learn features in a hierarchy — lower layers learn general patterns,
higher layers learn task-specific ones.

**Vision (ResNet trained on ImageNet):**
```
Layer 1–3:   edges, color gradients, textures          ← universal, any image domain
Layer 4–8:   shapes, corners, simple patterns          ← broadly reusable
Layer 9–14:  object parts (wheels, eyes, windows)      ← domain-specific
Layer 15+:   full objects, scene context               ← task-specific
```

**Language (BERT trained on Wikipedia + BooksCorpus):**
```
Lower layers:  token-level patterns, morphology, local syntax
Middle layers: phrase-level patterns, coreference, named entities
Upper layers:  sentence-level semantics, discourse, task-specific reasoning
```

The lower layers are universally useful. The upper layers need to be adapted to the
new task. Transfer learning reuses the lower layers and replaces or fine-tunes the upper.

---

## How It Is Used in Practice

### Step 1 — Start with a Pre-trained Base

Choose a base model pre-trained on a large dataset in your domain:

| Domain | Common Base Models |
|---|---|
| Vision | ResNet, ViT, EfficientNet (trained on ImageNet) |
| Language | BERT, RoBERTa, GPT (trained on web text) |
| Multimodal | CLIP, Flamingo |
| Code | CodeBERT, StarCoder |

### Step 2 — Freeze the Base

Freezing means setting `requires_grad=False` on the pre-trained weights — the optimizer
ignores them during backprop. The pre-trained knowledge is preserved, not overwritten.

```python
model = BertModel.from_pretrained("bert-base-uncased")

# Freeze all pre-trained weights
for param in model.parameters():
    param.requires_grad = False
```

Why freeze? With limited data, allowing gradients to flow into the base would overwrite
the pre-trained representations with noise from your small dataset.

### Step 3 — Add a Task-Specific Head

Replace the original output layer with one that matches your task:

```python
# Classification head on top of frozen BERT
class TextClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = model          # frozen
        self.classifier = nn.Linear(768, num_classes)   # trainable

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_embedding)
```

Only the classifier head (768 × num_classes parameters) trains. With 8 classes: 6,144
trainable parameters vs 110M total — drastically reduces overfitting risk.

### Step 4 — Optional: Gradual Unfreezing

If you have more data or need better performance, unfreeze the upper layers of the base
model and fine-tune them at a low learning rate:

```python
# Unfreeze only the last 2 transformer blocks
for layer in model.encoder.layer[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# Use a much lower learning rate for unfrozen base layers
optimizer = AdamW([
    {'params': model.encoder.layer[-2:].parameters(), 'lr': 1e-5},  # low lr for base
    {'params': classifier.parameters(), 'lr': 1e-3},                # normal lr for head
])
```

Lower learning rate prevents the pre-trained weights from moving too far from their
original values — fine-tuning rather than re-training.

---

## LoRA — Parameter-Efficient Fine-Tuning

Full fine-tuning updates all parameters. For very large models (7B+), even storing the
gradients is expensive. **LoRA (Low-Rank Adaptation)** fine-tunes a tiny fraction of
parameters while freezing the base.

### The Idea

Weight updates during fine-tuning tend to be low-rank — they live in a small subspace
of the full parameter space. LoRA exploits this: instead of updating the full weight
matrix W (d × d), it learns two small matrices A (d × r) and B (r × d) where r << d.

```
Full fine-tuning:
  W_new = W_pretrained + ΔW    ← ΔW is d×d, expensive

LoRA:
  W_new = W_pretrained + A × B   ← A is d×r, B is r×d
                                   r = 8 or 16 (rank)

For d=768, r=8:
  ΔW full:  768 × 768 = 590,592 parameters
  LoRA A+B: 768×8 + 8×768 = 12,288 parameters  ← 98% fewer
```

The base model weights are frozen. Only A and B are trained. At inference time, the
product A×B is folded back into the weight matrix — no extra computation.

```python
# Conceptually (libraries like HuggingFace PEFT handle this automatically)
class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8):
        super().__init__()
        d_in, d_out = base_layer.weight.shape
        self.base = base_layer          # frozen
        self.lora_A = nn.Linear(d_in, rank, bias=False)   # trainable
        self.lora_B = nn.Linear(rank, d_out, bias=False)  # trainable

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))
```

### Why LoRA Matters at Scale

| Approach | Trainable params (7B model) | GPU memory | Can merge back? |
|---|---|---|---|
| Full fine-tune | 7B | ~112GB (FP16 + optimizer) | n/a |
| Freeze + head | ~10M | ~14GB | yes |
| LoRA (r=16) | ~20M | ~14GB + tiny | yes — fold A×B into W |

LoRA lets you fine-tune a 7B model on a single A100 (40GB). The trained adapters are
only ~50MB, making it practical to maintain many task-specific adapters on top of one
shared base model.

---

## When to Use What

| Data size | Model size | Approach |
|---|---|---|
| < 1K examples | Any | Freeze all, train head only |
| 1K–100K examples | Small–medium | Freeze base, train head + optional unfreezing |
| 100K–1M examples | Large | LoRA or partial unfreezing |
| > 1M examples | Large | Full fine-tuning or train from scratch |

**The general rule:** the less data you have relative to model size, the more you freeze.
Freezing is regularization — it prevents the optimizer from fitting noise.

---

## Transfer Learning vs Training from Scratch

| | Transfer Learning | From Scratch |
|---|---|---|
| Data needed | Hundreds to thousands | Millions |
| Training time | Hours | Days to weeks |
| Risk of overfitting | Low (most params frozen) | High with small data |
| Domain mismatch risk | Yes — if source domain very different | None |
| Convergence | Fast — good initialization | Slow — random init |

**Domain mismatch caveat:** transfer learning works best when source and target domains
are similar. BERT fine-tuned for clinical notes works better starting from BioBERT
(pre-trained on medical text) than from vanilla BERT (pre-trained on Wikipedia). The
pre-trained features must be relevant to be transferable.
