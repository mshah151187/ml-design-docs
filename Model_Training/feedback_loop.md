# Feedback Loops in ML Systems

## What Is a Feedback Loop?

A feedback loop is a mechanism that takes signals from a deployed model's behavior in
production and uses them to improve the model over time. The core idea is that a model
trained on historical data will degrade or miss edge cases over time — production feedback
is the only way to detect and correct this continuously.

---

## Main Aim

1. **Detect model failures automatically** — without requiring human review of every prediction
2. **Generate labeled training data from production** — cheaply and at scale
3. **Retrain the model on failures** — so the same mistakes don't repeat
4. **Close the loop continuously** — model improves as traffic grows

---

## Core Components

### 1. Signal Collection
Capture signals from production that indicate whether the model was right or wrong.

Two types of signals:

**Explicit signals** — user directly tells you the outcome:
- User corrects a prediction ("that's wrong")
- User clicks / doesn't click (CTR)
- User rates the response (thumbs up/down)
- Ground truth arrives later (e.g., did the user actually churn?)

**Implicit signals** — inferred from behavior:
- Model confidence / token probability (low confidence = likely wrong)
- User abandons session after model response
- User repeats the same query (model didn't satisfy the need)
- Downstream system rejects model output (e.g., payment fails after fraud score = low risk)

### 2. Hard Negative Mining
Not all production data is equally useful for retraining. Easy examples the model already
handles correctly add little training signal. Hard negatives — examples the model got wrong
or was uncertain about — are the most valuable.

Filter production traffic to extract hard negatives:
- Model confidence below threshold
- Explicit user correction
- Ground truth label diverges from model prediction

Store in a dedicated hard negative store (data lake, S3/Parquet). Volume is typically
small relative to total traffic — the filtering does the heavy lifting.

### 3. Labeling
Hard negatives need ground truth labels for retraining. Options:

| Method | Cost | Scale | Quality |
|---|---|---|---|
| Human labeling | High | Low | High |
| LLM-as-judge | Medium (API cost) | High | High for well-defined tasks |
| Ground truth from downstream events | Zero | High | Exact (but delayed) |
| Heuristic rules | Zero | High | Low-medium |

LLM-as-judge is the practical default when ground truth is not naturally available —
run a capable model offline on hard negatives to produce labels. This does not touch
the critical path.

### 4. Retraining
Retrain the model on a mix of:
- Original training data (prevents catastrophic forgetting)
- Hard negatives (upweighted — these are where the model is wrong)

Two approaches depending on label type:

**Hard labels** — single correct class per example. Standard cross-entropy loss.
Simple to implement, sufficient when class boundaries are clear.

**Soft labels (Knowledge Distillation)** — probability distribution over classes
(e.g., 70% class A, 25% class B). KL divergence loss. Richer signal — encodes
uncertainty and proximity to decision boundaries. Better for borderline cases.

### 5. Deployment and Monitoring
After retraining, deploy and monitor:
- Did failure rate on the previously hard examples drop?
- Did overall model metrics improve or regress?
- Watch for distribution shift — new hard negatives emerging in different areas

---

## Types of Feedback Loops

### Online (Real-time)
Signal collected and model updated continuously or near-continuously.
Used when: model decisions have immediate measurable outcomes (ad click, recommendation).
Risk: feedback noise can destabilize training if not filtered carefully.

### Offline (Batch)
Signals accumulate over a window (daily/weekly), model retrained in batch.
Used when: ground truth is delayed, labeling requires human or LLM review.
More stable, easier to audit, sufficient for most production ML systems.

### Human-in-the-Loop
Human reviewers label a sample of hard negatives before retraining.
Used when: mistakes are high-stakes (medical, legal, fraud) and automated labeling
is not trusted.

---

## Common Failure Modes

**Feedback loops amplify bias** — if the model is wrong about a subgroup, it generates
no confident predictions for that subgroup, so no hard negatives are collected, so it
never improves on that subgroup.

**Label leakage** — signal used for labeling is also available at inference time →
model learns to cheat rather than generalize.

**Distribution shift** — production data drifts away from training distribution over time.
Feedback loop must collect enough recent data to cover new distribution, not just
replay old training data.

**Delayed ground truth** — for some tasks (churn, fraud), true label only arrives days
or weeks later. Pipeline must handle this lag without training on incomplete labels.
