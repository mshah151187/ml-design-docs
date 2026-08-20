# CLIP and Vision Transformers

Reference: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., OpenAI 2021)

---

## 1. What Problem CLIP Solves

### Traditional image classification — the fixed label problem

```
Classic supervised vision model:
  Training data: 1.2M labeled images, 1000 ImageNet classes
  Model output:  probability over exactly those 1000 classes
  
  Limitation: want to classify "satellite dish"?
    → not in ImageNet → collect new labeled data → retrain
    → every new concept requires new labeled data + retraining
```

The model learns visual features from pixel patterns, but it cannot generalize
to concepts it has never explicitly been trained on. The vocabulary of what the
model "understands" is frozen at training time.

### CLIP's solution — learn from natural language supervision

```
Instead of (image, fixed_label) pairs:
  use (image, text_description) pairs scraped from the internet
  400 million image-text pairs: web images + their alt-text, captions, nearby text

The model learns:
  "a photo of a dog" ↔ image of a dog
  "satellite dish on a rooftop" ↔ image of a satellite dish
  "impressionist oil painting of water lilies" ↔ that painting

Result: the model learns to align visual and language representations
        in a shared embedding space.
```

**What this unlocks:**

```
Zero-shot classification (no retraining needed):
  Want to classify "satellite dish"?
  → write "a photo of a satellite dish" → get text embedding
  → compare to image embedding
  → done — no new labeled data, no retraining

Open-vocabulary retrieval:
  "Find all images showing someone cooking outdoors"
  → encode query as text → retrieve images by embedding similarity
  → no concept had to be predefined at training time
```

---

## 2. Architecture Overview — Two Encoders, One Shared Space

```
         IMAGE                           TEXT
           ↓                               ↓
  [Image Encoder]                  [Text Encoder]
  (Vision Transformer)             (Transformer)
           ↓                               ↓
  image_embedding ∈ R^d          text_embedding ∈ R^d
           └──────────────┬────────────────┘
                   cosine similarity
                   (shared vector space)
```

Two separate encoders with different architectures — but both output into the
**same d-dimensional space**. The shared space is the whole point: it makes
cross-modal dot products meaningful.

Training objective: make (image, its_caption) similar; make (image, other_captions)
dissimilar. After training, any image and any text can be compared in this space.

---

## 3. Image Encoder — Vision Transformer (ViT)

ViT (Dosovitskiy et al., 2020) applies the transformer architecture directly to
images by treating image patches as tokens. This is the image encoder used in
CLIP's stronger variants (ViT-L/14, ViT-H/14).

### Step 1 — Split image into patches

```
Input image: H × W × C  (height × width × channels, e.g. 224 × 224 × 3)
Patch size: P × P        (e.g. 16 × 16 pixels)
Number of patches: N = (H/P) × (W/P) = (224/16) × (224/16) = 14 × 14 = 196

Each patch: P × P × C = 16 × 16 × 3 = 768 raw values (flattened)

Image of a cat:
  ┌──┬──┬──┬──┐
  │  │  │  │  │
  ├──┼──┼──┼──┤
  │  │🐱│  │  │   each cell = one 16×16 patch → one token
  ├──┼──┼──┼──┤
  │  │  │  │  │
  └──┴──┴──┴──┘
  196 patches total → sequence of 196 tokens
```

### Step 2 — Patch embedding (linear projection)

```
Each patch: 768-dim raw pixel vector
        ↓  Linear projection W_E  (768 → d_model, e.g. d_model = 1024)
        ↓
patch_embedding ∈ R^d_model

This is the "tokenization" step for images:
  Text:  word → lookup table → embedding
  Image: patch → linear projection → embedding

The linear projection is learned during training.
It learns to compress the raw patch pixels into a semantically useful vector.
```

### Step 3 — Prepend [CLS] token

```
A special learnable vector [CLS] is prepended to the sequence:
  Input sequence: [CLS, patch_1, patch_2, ..., patch_196]
  Length: 197 tokens

Why CLS?
  After the transformer runs, we need ONE vector to represent the ENTIRE image.
  [CLS] is designed exactly for this: it has no patch content of its own,
  so it is free to aggregate global information from all other patches
  through attention. After L transformer layers, [CLS] output = image embedding.
  
  Same idea as BERT's [CLS] token for sentence classification.
```

### Step 4 — Add positional embeddings

```
Patches are flattened in row-major order, but the transformer's attention
mechanism has no inherent notion of position. Without positional encoding,
the model cannot tell whether patch_42 is adjacent to patch_43 or far apart.

ViT uses learned 1D positional embeddings:
  pos_embedding[0]   = position of [CLS]
  pos_embedding[1]   = position of patch at row 0, col 0
  pos_embedding[2]   = position of patch at row 0, col 1
  ...
  pos_embedding[196] = position of patch at row 13, col 13

Final input to transformer:
  token[i] = patch_embedding[i] + pos_embedding[i]

Note: ViT uses LEARNED absolute position embeddings, not RoPE.
Why not RoPE? RoPE was designed for sequential 1D text.
Images are 2D spatial — 2D absolute embeddings or learned 1D work in practice.
Some newer models (DeiT, Swin) use 2D relative position encodings.
```

### Step 5 — Transformer encoder (BIDIRECTIONAL attention)

```
L identical transformer blocks, each containing:
  1. Layer Norm (Pre-LN in modern ViT variants)
  2. Multi-Head Self-Attention (FULL / BIDIRECTIONAL — no causal mask)
  3. Residual connection
  4. Layer Norm
  5. FFN (two linear layers with GELU activation)
  6. Residual connection

One ViT-L/14 block:
  input: sequence of 197 tokens, d_model = 1024
  attention heads: 16
  d_head = 1024 / 16 = 64
  FFN hidden dim: 4096 (4 × d_model, same ratio as text transformers)
```

### The Attention Computation — No Causal Mask

```
For a sequence of 197 tokens (1 CLS + 196 patches):

Q = X W_Q   (197 × d_model → 197 × d_model)
K = X W_K
V = X W_V

Attention scores: A = Q K^T / √d_head    shape: (197 × 197)

  patch_1  patch_2  ...  patch_196  CLS
  [ 0.8     0.1    ...    0.3      0.9 ]  ← attention from patch_1
  [ 0.2     0.9    ...    0.1      0.7 ]  ← attention from patch_2
  ...
  [ 0.5     0.4    ...    0.8      0.9 ]  ← attention from CLS

ALL 197 × 197 entries computed — no masking.

patch_3 (upper left of image) can directly attend to patch_180 (lower right).
This is full bidirectional attention — every patch "sees" every other patch.
```

### Step 6 — Extract [CLS] output as image embedding

```
After L transformer blocks:
  Output[0] = [CLS] embedding after attending to all 196 patches
  This vector has aggregated information from the entire image.

  Output[0] → Linear projection → L2 normalize → image_embedding ∈ R^d

The linear projection maps to the shared CLIP embedding space.
L2 normalization makes cosine similarity = dot product (unit sphere).
```

---

## 4. Text Encoder

```
Input: "a photo of a golden retriever"
        ↓ BPE tokenizer (same as GPT-2)
[SOS, "a", "photo", "of", "a", "golden", "retriever", EOS]
        ↓ token embedding lookup
sequence of token embeddings
        ↓ + positional embeddings
        ↓ transformer encoder (L layers)
        ↓ extract EOS token representation
        ↓ linear projection → L2 normalize
text_embedding ∈ R^d
```

**CLIP text encoder uses CAUSAL (masked) attention:**

```
Token sequence: [SOS, "a", "photo", "of", "a", "golden", "retriever", EOS]

Attention mask (lower triangular — GPT-style):
         SOS   a   photo   of   a   golden  retriever  EOS
SOS  [    1    0     0     0   0     0       0         0  ]
a    [    1    1     0     0   0     0       0         0  ]
photo[    1    1     1     0   0     0       0         0  ]
...
EOS  [    1    1     1     1   1     1       1         1  ]
              ↑ EOS attends to ALL tokens

Extract: EOS token representation → has attended to full sequence → text summary
```

Why causal here? CLIP's text encoder was designed as a GPT-2 variant.
The [EOS] token in a causal transformer has "read" all preceding tokens,
making it a valid sequence summary. The causal mask isn't strictly needed for
embedding — a bidirectional encoder (like BERT) also works and is more common
in newer contrastive text models (e.g. OpenCLIP, SigLIP use bidirectional text encoder).

---

## 5. Causal vs Bidirectional — Why Vision Doesn't Use Causal Attention

This is the fundamental difference between how transformers are used in language vs vision.

### Text — Causal is needed for GENERATION

```
Language model generating "The cat sat on the":
  When predicting next token, cannot "cheat" by looking at future tokens.
  Causal mask enforces this — token at position t only attends to positions 0..t.

  "The" → can attend to: [The]
  "cat" → can attend to: [The, cat]
  "sat" → can attend to: [The, cat, sat]
  ...

  Without causal mask: model sees the answer it's trying to predict → trivial shortcut
```

### Text — Bidirectional for UNDERSTANDING (BERT)

```
BERT task: predict masked tokens
  "The [MASK] sat on the mat"
  → model needs context from BOTH sides to predict "cat"
  → "The ___ sat" AND "sat on the mat" → context from both directions

No causal mask: every token attends to every other token.
Better for encoding/understanding tasks; cannot generate.
```

### Vision — ALWAYS bidirectional, causal makes no sense

```
Image patch sequence (row-major):
  [patch_top-left, ..., patch_center, ..., patch_bottom-right]

Is there a "future" that must be hidden?
  NO. An image has no temporal order.
  The "future" patches are just spatially to the right or below — 
  there is no reason to hide them.

Every patch needs context from ALL directions:
  patch_center (eye of the cat) needs to see:
    - patches above it (forehead)
    - patches below it (nose, mouth)
    - patches to the left and right (other eye, ears)

  If we used causal masking:
    patch_center could only see patches above-left of it.
    Completely breaks the ability to understand spatial context.

ViT uses full bidirectional attention by design.
Causal attention in a ViT would be architecturally wrong, not just suboptimal.
```

### Summary table

| Model | Architecture | Attention | Why |
|---|---|---|---|
| GPT / LLaMA | Decoder-only | Causal (lower-triangular mask) | Autoregressive text generation |
| BERT / RoBERTa | Encoder-only | Bidirectional (full) | Text understanding, masked prediction |
| ViT (image encoder) | Encoder-only | Bidirectional (full) | Images have no temporal direction |
| CLIP text encoder | Encoder-only (GPT-style) | Causal | Historical choice, GPT-2 based |
| CLIP image encoder | Encoder-only (ViT) | Bidirectional | Images are 2D spatial, not sequential |
| T5, BART | Encoder-Decoder | Encoder: bidirectional, Decoder: causal | Understand input, generate output |

---

## 6. What Attention Learns in ViT

After training, ViT attention heads learn to attend to semantically meaningful regions,
not just spatial neighbors.

```
Early layers (1-4):
  Local attention — each patch attends mostly to nearby patches.
  Similar to CNN early layers detecting edges, textures, local patterns.
  Patch attention radius ≈ 2-3 patches away.

Middle layers (5-8):
  Expanding attention — patches start attending to distant but related patches.
  Eye patch begins attending to nose patch (part-of-face awareness).

Late layers (9-12):
  Global semantic attention — [CLS] attends to the most semantically important patches.
  For a dog image: [CLS] attends strongly to the face/eyes, weakly to background.
  Patches of the same object cluster their attention together.
  
  This is "attention visualization" — shows WHERE the model is looking.
  ViT naturally learns object saliency without any saliency labels.
```

```
[CLS] attention over patches in a dog image (late layer):

┌──┬──┬──┬──┬──┬──┬──┐
│░░│░░│░░│░░│░░│░░│░░│  ← background, low attention
├──┼──┼──┼──┼──┼──┼──┤
│░░│░░│▓▓│██│▓▓│░░│░░│  ← face region, HIGH attention
├──┼──┼──┼──┼──┼──┼──┤
│░░│▓▓│██│██│██│▓▓│░░│  ← body, medium attention
├──┼──┼──┼──┼──┼──┼──┤
│░░│░░│▓▓│▓▓│▓▓│░░│░░│  ← legs, low-medium
└──┴──┴──┴──┴──┴──┴──┘

██ = very high attention weight from [CLS]
▓▓ = medium
░░ = low
```

---

## 7. CLIP Training — Contrastive on N×N Matrix

```
Batch of N image-text pairs:
  (image_1, "a dog playing in the snow")
  (image_2, "golden gate bridge at sunset")
  (image_3, "person eating sushi")
  ...
  (image_N, ...)

Compute N image embeddings: I₁, I₂, ..., Iₙ  (each L2-normalized)
Compute N text embeddings:  T₁, T₂, ..., Tₙ  (each L2-normalized)

N×N cosine similarity matrix:
           T₁    T₂    T₃  ...  Tₙ
  I₁  [ 0.91  0.02  0.01 ...  0.03 ]  ← diagonal = positive pair
  I₂  [ 0.01  0.89  0.02 ...  0.01 ]
  I₃  [ 0.02  0.03  0.92 ...  0.02 ]
  ...
  Iₙ  [ 0.01  0.02  0.01 ...  0.88 ]

Positive pairs: diagonal entries (image_i, text_i)
Negative pairs: all off-diagonal entries (image_i, text_j where j≠i)

Loss: symmetric cross-entropy
  Row loss:    for each image, text_i should rank highest among all N texts
  Column loss: for each text, image_i should rank highest among all N images
  Total loss = (row_loss + column_loss) / 2

Temperature τ (learned scalar):
  A = logits / exp(τ)   ← τ sharpens/softens the distribution
  Higher τ → softer, easier training; lower τ → sharper, harder
```

**Why symmetric loss?**

CLIP needs to work in BOTH directions:
- Image query → find matching text (image search)
- Text query → find matching image (text-to-image retrieval)

Symmetric loss ensures both directions are trained equally well.

**Scale matters critically:**

Larger N = more negatives = harder task = better representations.
CLIP trained on batches of 32,768 pairs — each image has 32,767 negatives.
This is why CLIP requires large compute: you need huge batches for the negatives
to be diverse and informative.

---

## 8. Zero-Shot Inference

### Image classification without any labeled data

```
Step 1 — Build text embeddings for each class:
  Template: "a photo of a {class}"
  → "a photo of a cat"    → text_emb_cat
  → "a photo of a dog"    → text_emb_dog
  → "a photo of a truck"  → text_emb_truck
  (multiple templates can be averaged for robustness)

Step 2 — Encode query image:
  image → ViT → image_emb

Step 3 — Score against all class embeddings:
  score_cat   = cosine_similarity(image_emb, text_emb_cat)   = 0.31
  score_dog   = cosine_similarity(image_emb, text_emb_dog)   = 0.89  ← highest
  score_truck = cosine_similarity(image_emb, text_emb_truck) = 0.04

Step 4 — argmax → prediction = "dog"
```

No fine-tuning, no labeled data for "dog". The model generalizes from its
contrastive pre-training on internet-scale image-text pairs.

**CLIP zero-shot vs supervised ResNet50 on ImageNet:**
- Zero-shot CLIP ViT-L/14: ~76% top-1 accuracy
- Supervised ResNet-50: ~76% top-1 accuracy
- Same accuracy — but zero-shot CLIP was never trained on ImageNet labels.

---

## 9. ViT Variants and CLIP Naming

```
Naming: ViT-{size}/{patch_size}

ViT-B/32:  Base model, 32×32 patches → (224/32)² = 49 patches → shorter sequence
ViT-B/16:  Base model, 16×16 patches → (224/16)² = 196 patches → richer but slower
ViT-L/14:  Large model, 14×14 patches → (224/14)² = 256 patches → CLIP's best
ViT-H/14:  Huge model, 14×14 patches → largest CLIP variant

Sizes:
  ViT-B: 12 layers, d_model=768,  12 heads, 86M params
  ViT-L: 24 layers, d_model=1024, 16 heads, 307M params
  ViT-H: 32 layers, d_model=1280, 16 heads, 632M params

Smaller patch size → more patches → longer sequence → richer representation
                                                    → more compute (O(N²) attention)
```

---

## 10. CLIP vs ResNet (CNN) — Why Transformers Win for Multi-Modal

```
ResNet (CNN):
  Convolution = local kernel sliding across image
  Layer 1: detects edges (3×3 receptive field)
  Layer 2: detects corners (9×9 effective receptive field)
  Layer N: detects object parts (large receptive field after many layers)
  
  Global context only available in DEEP layers.
  Image is processed in a hierarchical, local-to-global manner.

ViT (Transformer):
  Self-attention = every patch attends to every other patch from LAYER 1
  Layer 1: can already detect global relationships
  (eye patch attends to nose patch — skip the local-then-global progression)
  
  Global context available from the very first layer.
```

**Why global context matters for CLIP:**

```
CLIP needs to align image meaning with text meaning.

Text "a dog catching a frisbee":
  The TEXT encoder understands "dog", "catching", "frisbee" as semantic concepts
  and their RELATIONSHIP.

For the IMAGE encoder to align with this:
  It needs to understand the SCENE, not just local textures.
  "Dog" + "frisbee" + "action" all must be globally understood at once.

CNN: must wait until deep layers to have global view — information bottleneck.
ViT: global attention from layer 1 — no bottleneck.

This is why CLIP ViT-L/14 significantly outperforms CLIP ResNet variants.
```

---

## 11. How CLIP is Used in Video Recommendation

```
At video upload time:

  Video file
      ↓ sample K frames (e.g., 1 frame/sec)
      ↓
  CLIP Image Encoder (ViT)
      ↓ embed each frame
  frame_embeddings: [f_1, f_2, ..., f_K]
      ↓ mean-pool (or max-pool)
  video_visual_embedding ∈ R^d

  Also: title + description + hashtags
      ↓ CLIP Text Encoder
  video_text_embedding ∈ R^d

  Final video representation:
    concat([video_visual_embedding, video_text_embedding])
    → or: fusion MLP → final video_embedding
```

**Why this is more powerful than text-only video embedding:**

```
Text-only:   "skateboarding tricks compilation"
  Captures: topic (skateboarding), format (compilation)
  Misses: visual style (slow-mo, fisheye lens), trick difficulty,
          production quality, visual similarity to other skate content

CLIP visual: embeds actual frames → captures visual style, motion patterns,
             scene characteristics that never appear in the text description
             "skateboarding compilation" by GoPro footage ≠ same text embedding
             but ViT embeddings will be different (visual style encoded)

Zero-shot genre classification:
  "a skateboarder doing a kickflip" → text_embedding
  → cosine similarity with video_visual_embedding
  → no labeled genre training data needed
```

**CLIP in the user tower:**

```
User's recently watched videos → CLIP-embedded frames
  → visual preference profile: user likes close-up cooking shots vs wide-angle
  → goes beyond topic into visual style
  → captures preferences text metadata would miss entirely
```

---

## 12. Interview Summary

**Q: What is CLIP and what problem does it solve?**
CLIP learns a shared embedding space for images and text from 400M internet image-text
pairs using contrastive learning. It solves the fixed-label problem of supervised vision:
zero-shot classification of any concept just by writing its text description.

**Q: How does ViT tokenize an image?**
Splits image into P×P patches, flattens each patch, projects to d-dim via a learned
linear layer. Prepends a [CLS] token. Adds learned positional embeddings. The CLS token
output after L transformer layers is the image embedding.

**Q: Why doesn't ViT use causal attention?**
Images have no temporal order — there are no "future" patches to hide. Causal masking
would prevent a patch from attending to spatially adjacent patches on the right or below.
ViT uses full bidirectional attention: every patch attends to every other patch. This is
fundamentally correct for 2D spatial data, not a design choice.

**Q: What's the difference between ViT attention and GPT attention?**
GPT (causal): lower-triangular mask, each token attends only to itself and previous
tokens — necessary for autoregressive generation.
ViT (bidirectional): no mask, full N×N attention matrix — every patch attends to every
other patch. This lets the model capture non-local image relationships from layer 1.

**Q: How does CLIP enable zero-shot classification?**
Encode candidate class labels as text ("a photo of a {class}") → text embeddings.
Encode query image → image embedding. Score via cosine similarity → argmax.
No fine-tuning, no labeled data for the target classes.

**Q: Why ViT over ResNet for CLIP?**
ViT has global attention from layer 1 — can understand the semantic scene holistically.
ResNet only develops global context in deep layers through pooling. CLIP needs the image
encoder to capture semantic meaning aligned with text (whole-scene understanding),
which ViT does more naturally than local-to-global CNN hierarchy.
