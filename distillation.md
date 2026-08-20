# Knowledge Distillation — ML System Design Reference

---

## What Is Distillation?

A large model has high quality but is too slow to serve at production QPS. A small model
is fast enough but learns poorly from scratch — not enough capacity to discover complex
patterns from raw data alone.

**Distillation transfers a large model's quality into a small model's body.**

The small model (student) learns not from hard 0/1 labels but from the large model's
(teacher's) output probability distribution — soft labels.

```
Hard label (from human annotation or click logs):
  (query, job_A) → 1   ← "relevant"
  (query, job_B) → 1   ← "relevant"

Soft label (from teacher model):
  (query, job_A) → 0.95   ← "strongly relevant, teacher is confident"
  (query, job_B) → 0.51   ← "borderline relevant, teacher is uncertain"
```

Both job_A and job_B are labeled "1" with hard labels — but the teacher is telling you
one is a clear positive and the other is a coin flip. The student learns that nuance.
Hard labels destroy this information; soft labels preserve it.

---

## Why Soft Labels Are Richer

A teacher model trained on millions of examples has internalized patterns that hard
labels cannot express:

- **Uncertainty near decision boundaries:** a borderline job has prob 0.51 not 0.99 —
  the student learns this is a hard case, not a clear signal
- **Relative ordering:** prob 0.95 vs 0.72 vs 0.51 encodes a ranking, not just a binary
- **Dark knowledge:** the teacher's non-top class probabilities encode what the item is
  similar to — "this ML job is also somewhat relevant for a data science query (0.3)"

---

## The Training Loss — KL Divergence

Instead of cross-entropy against 0/1, the student minimizes KL divergence against the
teacher's full output distribution:

```
L_distill = KL(student_output || teacher_output)
           = Σ teacher(x) × log(teacher(x) / student(x))

student tries to match teacher's exact probability distribution,
not just its hard argmax decision
```

In practice, a temperature T is applied to soften the distributions further:

```
soft_prob = softmax(logits / T)

T=1: original distribution
T>1: softer — emphasizes small probability differences (more dark knowledge)
T<1: sharper — emphasizes the top class more
```

---

## The Economics — Why Distillation Exists

| Option | Quality | Serving cost |
|---|---|---|
| Serve 8B teacher directly | Best | Infeasible at 100k QPS, 500ms budget |
| Train 0.6B student from scratch on hard labels | Poor | Affordable |
| Train 0.6B student via distillation from 8B | Near-teacher quality | Affordable |

You pay the large model cost **once** during training (offline, batch), not on every live
request. The student inherits the teacher's quality at inference time without the cost.

---

## Multi-Teacher Distillation

When the student must optimize multiple objectives (e.g., relevance AND engagement),
a single teacher trying to do both produces weaker signal for each objective. Instead,
train specialized teachers — one per objective — and distill from all simultaneously.

```
          Teacher_A (relevance specialist)
               ↓ soft relevance labels
  Student ←────┤
               ↓ soft engagement labels
          Teacher_B (engagement specialist)

L_total = w_A × KL(student || teacher_A) + w_B × KL(student || teacher_B)
```

**Benefits:**
- Each teacher specializes → cleaner signal per objective
- Student learns both objectives in one forward pass → one inference call at serving
- Teachers can be updated or replaced independently
- Loss weights w_A, w_B tune the relevance-engagement tradeoff

**Warm-start strategy:** instead of distilling from an open-source checkpoint,
first train a relevance-specialized student (single teacher), then warm-start
multi-teacher distillation from that checkpoint. The student starts with strong
relevance, then focuses on learning engagement — avoiding relevance drift.

---

## Offline vs Online Multi-Teacher Distillation

### Online (default)
Teachers run in real time during training. For each training batch, the batch is
forwarded through all teachers to get soft labels, then the student updates.

```
batch → Teacher_A → soft_labels_A ──┐
batch → Teacher_B → soft_labels_B ──┤→ student update
                                    ┘
```

**Cost:** teacher inference on every batch. For 1.7B teachers, this is expensive.

### Offline (cached)
Teacher outputs are pre-computed once over the entire training dataset and stored
(HDFS/NFS). During student training, labels are fetched from storage — no live
teacher inference.

```
Pre-computation pass:
  all training pairs → Teacher_A → cached labels (stored)
  all training pairs → Teacher_B → cached labels (stored)

Student training:
  batch → fetch cached labels → student update (no teacher inference)
```

**Savings (LinkedIn):** 35% reduction in training time, 25% GPU-hours saved.
**Tradeoff:** teacher labels are frozen — if the teacher is updated, cache must be rebuilt.

---

## Distillation vs Fine-Tuning vs Training From Scratch

| Approach | Starting point | Learns from | When to use |
|---|---|---|---|
| Train from scratch | Random weights | Raw data + hard labels | Huge data, novel architecture |
| Fine-tuning | Pre-trained weights | Task-specific data + hard labels | Adapting general model to domain |
| Distillation | Pre-trained or random | Teacher's soft label distribution | Compressing a large model for serving |
| Fine-tuning + distillation | Pre-trained weights | Both teacher soft labels + hard labels | Best of both — domain + compression |

LinkedIn uses **fine-tuning + distillation** together: start from a pre-trained open-source
SLM, fine-tune with domain reasoning and soft labels from the oracle, then distill from
specialized teachers. Each step adds a layer of quality.

---

## Distillation in LinkedIn Semantic Search — The Full Chain

```
Search traffic logs
  (query, document) pairs
        ↓
SAGE 8B Oracle
  Input:  (query, document) as text prompt
  Output: ordinal grade {1, 2, 3, 4}
  Role:   policy-aligned judge — defines what "relevant" means
        ↓
        ├──────────────────────────────────────────────┐
        ↓                                              ↓
Relevance Teacher (1.7B)                    Bi-Encoder Training
  Input:  (query, doc) text + oracle grade    Input: oracle-labeled pos/neg pairs
  Output: calibrated Yes/No soft label       Loss:  InfoNCE contrastive
  Loss:   soft-label SFT + ranking loss      Output: embedding model for retrieval
  Role:   intermediate — decouples oracle
          cost from student training
        ↓
        ├──────────────────────────────────────────────┐
        ↓                                              ↓
Student SLM (0.6B)                        Engagement Teacher (1.7B)
  Input:  teacher soft labels (rel)          Input:  user action logs
          teacher soft labels (eng)          Output: engagement soft labels
  Loss:   weighted KL divergence            Role:   specialist for click/apply/connect
  Role:   production ranker

SERVING TIME:
  Only Student SLM (0.6B) runs on live traffic.
  Oracle + Teachers are offline — they never touch a live request.
```

---

## Key Interview Takeaways

| Question | Answer |
|---|---|
| Why distill instead of serving the teacher? | Teacher (8B) cannot meet latency/cost at 100k QPS; student (0.6B) can |
| Why soft labels instead of hard labels? | Soft labels preserve uncertainty and relative ordering — student learns nuance |
| Why multi-teacher? | Each teacher specializes in one objective → cleaner signal per task |
| Why warm-start MTD from relevance SLM? | Prevents relevance drift when adding engagement objectives |
| Do teachers run at serving time? | No — only the student runs live. Teachers are training-time label generators |
| What is offline distillation? | Pre-compute teacher labels once, cache them — eliminates live teacher inference during training |
