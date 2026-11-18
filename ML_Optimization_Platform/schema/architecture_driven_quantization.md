# Architecture-Driven Quantization

> **Purpose:** Define what can be inferred about quantization by looking only at the model architecture/config, before considering runtime engine support, hardware capability, calibration strategy, or empirical quality/performance.

## 1. Core Principle

Model architecture answers two questions:

```text
1. WHAT categories of quantization are relevant?
2. WHERE in the graph can low precision be applied?
```

It does **not** usually answer:

```text
INT4 vs INT8 vs FP8
AWQ vs GPTQ
which runtime kernel will execute it
whether the quantized model will meet quality targets
whether the quantized path will actually be faster
```

Those decisions require additional inputs later.

## 2. Architecture-Level Quantization Decision Flow

```text
+------------------------------------------------------+
|                  MODEL ARCHITECTURE                  |
|                                                      |
| Layers                                               |
| Operators                                            |
| State                                                |
| Attention structure                                  |
| Residual / branch structure                          |
| Heterogeneous modules                                |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|  STEP 1: IDENTIFY QUANTIZATION SURFACES              |
|                                                      |
|  Does the model contain large learned weights?       |
|      -> Weight quantization relevant                 |
|                                                      |
|  Does inference create large intermediate tensors?   |
|      -> Activation quantization relevant             |
|                                                      |
|  Does autoregressive decoding persist K/V state?     |
|      -> KV-cache quantization relevant               |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|  STEP 2: INSPECT GRAPH / OPERATOR PRECISION NEEDS    |
|                                                      |
|  Large GEMMs / Linear layers                         |
|      -> low-precision candidates                     |
|                                                      |
|  Norm / Softmax / reductions                         |
|      -> usually remain higher precision              |
|                                                      |
|  Residual / merge points                             |
|      -> scale / precision compatibility matters      |
|                                                      |
|  Heterogeneous modules                               |
|      -> module-specific precision likely             |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|            CANDIDATE MIXED-PRECISION MAP             |
|                                                      |
|  Which regions can be low precision?                 |
|  Which regions likely remain higher precision?       |
|  Which quantization categories are applicable?       |
+------------------------------------------------------+
```

## 3. Step 1 — Identify Quantization Surfaces

The first pass over architecture asks:

> Which kinds of tensors and persistent state exist in this model?

This gives us the major quantization surfaces.

### 3.1 Weight Quantization

If the architecture contains large parameterized operators such as:

```text
Linear
Conv
Embedding
Expert MLP
Projection layers
```

then weight quantization is structurally relevant.

For Transformer models, the dominant targets are usually:

```text
Q projection
K projection
V projection
O projection

MLP up projection
MLP gate projection
MLP down projection
```

These large matrices contain most of the model parameters.

Architecture therefore tells us:

```text
Large learned matrices exist
    ->
Weight quantization is applicable
```

It does **not** yet tell us whether those weights should be FP8, INT8, INT4, AWQ, or GPTQ.

### 3.2 Activation Quantization

If inference produces large intermediate tensors that feed expensive operators, activation quantization is structurally relevant.

For a Transformer:

```text
hidden state
    ->
Q/K/V projections
    ->
attention intermediate tensors
    ->
MLP intermediates
    ->
residual outputs
```

These tensors are activations.

Architecture therefore tells us:

```text
Large runtime intermediate tensors exist
    ->
Activation quantization is potentially applicable
```

Architecture alone does not determine static vs dynamic, FP8 vs INT8, or scale granularity.

### 3.3 KV-Cache Quantization

KV-cache quantization depends strongly on architecture.

For an autoregressive decoder:

```text
causal self-attention
+
incremental token generation
+
persistent K/V from previous tokens
```

there is persistent KV state.

Therefore:

```text
KV-cache quantization
    = structurally applicable
```

For a model without persistent autoregressive KV state, such as a conventional CNN classifier:

```text
KV-cache quantization
    = not applicable
```

This is one of the clearest cases where model architecture alone can eliminate an entire quantization category.

## 4. Step 2 — Inspect the Graph at Operator Level

The first pass identifies weights, activations, and KV cache.

The second pass asks:

> Where in the graph is low precision structurally sensible, and where should higher precision probably remain?

For a Transformer block:

```text
Input Hidden State
       |
       v
    RMSNorm
       |
       v
  Q/K/V Linear
       |
       v
Attention / Softmax
       |
       v
 Output Linear
       |
       +----------------+
       |                |
       v                |
  Residual Add <--------+
       |
       v
    RMSNorm
       |
       v
    MLP GEMMs
       |
       v
  Residual Add
```

A candidate precision map may look like:

```text
RMSNorm
    -> likely higher precision

Q/K/V Linear
    -> low-precision candidate

Attention score / Softmax
    -> likely higher precision

Output Linear
    -> low-precision candidate

Residual Add
    -> precision / scale compatibility matters

MLP Linear layers
    -> low-precision candidate
```

This is where architecture starts to imply **mixed precision**.

## 5. Architecture and Mixed Precision

A quantized model rarely means every operation runs at the same low precision.

Architecture often naturally leads to something like:

```text
Large Linear weights
    -> low precision

Large GEMM activations
    -> possibly low precision

Norm / Softmax / reductions
    -> higher precision

Sensitive output paths
    -> possibly higher precision
```

The architecture-level conclusion is therefore:

> Some regions are strong low-precision candidates, while other regions should remain in higher precision.

This is a **candidate mixed-precision structure**.

It is not yet the final bit-width assignment.

## 6. Architecture Does Not Usually Select INT4 vs FP8

Suppose the model is:

```text
Decoder-only Transformer
100B parameters
80 layers
GQA
RMSNorm
SwiGLU
BF16 canonical weights
```

From architecture alone, we can conclude:

```text
Weight quantization:
    applicable

Activation quantization:
    applicable around large GEMMs

KV-cache quantization:
    applicable because autoregressive KV persists

Norm / Softmax:
    likely remain higher precision

Mixed precision:
    expected
```

But architecture alone should **not** decide:

```text
QKV Linear = INT4
MLP = FP8
KV = FP8
```

That requires runtime support, hardware support, numerical behavior, quality constraints, and performance measurements.

## 7. MoE Example

A Mixture-of-Experts architecture contains:

```text
Router
    |
    +--> Expert 1
    +--> Expert 2
    +--> Expert 3
    ...
```

Each expert still contains weight matrices.

Therefore architecture alone tells us:

```text
Expert weights
    -> weight quantization is conceptually applicable
```

It does **not** tell us:

```text
INT4 expert execution is supported
```

That is a runtime/kernel question and belongs to the next stage.

So:

```text
MoE
    != INT4 impossible
```

Architecture only tells us that the model contains heterogeneous expert subgraphs and may need module-specific precision treatment.

## 8. Multimodal Example

Suppose:

```text
Vision Encoder
      |
Projection
      |
LLM Decoder
```

Architecture tells us the model is heterogeneous.

Therefore one global precision policy is probably inappropriate.

A candidate structure may be:

```text
Vision Encoder:
    precision decision A

Projection:
    precision decision B

LLM:
    precision decision C
```

Architecture does not necessarily eliminate INT4.

It tells us that quantization should likely be **module-specific**.

## 9. Residual Connections and Branches

Architectures with skip connections, residual branches, and multiple tensor merges introduce precision-management issues.

Example:

```text
          +-------------+
          |             |
X --------+--> Layer -->+--> Add
          |             |
          +-------------+
```

If branches use different quantization scales, the merge may require rescaling, higher-precision accumulation, or compatible scale handling.

Architecture therefore tells us:

```text
branch/merge points
    -> require precision compatibility
```

This does not make activation quantization impossible. It makes the quantization scheme structurally more complicated.

## 10. Attention Structure and KV Importance

Model configuration can reveal MHA, GQA, or MQA.

For example:

```text
num_attention_heads = 64
num_key_value_heads = 64
    -> MHA

num_attention_heads = 64
num_key_value_heads = 8
    -> GQA

num_key_value_heads = 1
    -> MQA
```

This changes KV-cache footprint.

Conceptually:

```text
MHA
    -> larger KV footprint

GQA
    -> smaller KV footprint

MQA
    -> much smaller KV footprint
```

Therefore architecture can change the **value** of KV quantization.

It does not necessarily change whether KV quantization is possible.

Important distinction:

```text
Applicability
    !=
Usefulness
```

## 11. Architecture-Level Output

```text
+------------------------------------------------------+
|             ARCHITECTURE-LEVEL OUTPUT                |
|                                                      |
| Quantization Surfaces                                |
|   - Weight quantization applicable?                  |
|   - Activation quantization applicable?              |
|   - KV-cache quantization applicable?                |
|                                                      |
| Candidate Low-Precision Regions                      |
|   - Large GEMMs / Linear layers                      |
|   - Expert MLPs                                      |
|   - Projection layers                                |
|                                                      |
| Candidate High-Precision Regions                     |
|   - Norms                                            |
|   - Softmax                                          |
|   - reductions                                       |
|   - sensitive merge/output regions                   |
|                                                      |
| Structural Complexity                                |
|   - residuals / branches                             |
|   - heterogeneous modules                            |
|   - MoE                                              |
|   - multimodal components                            |
|                                                      |
| Result                                               |
|   - candidate mixed-precision structure              |
+------------------------------------------------------+
```

## 12. The Two Architecture-Level Questions

The clean interview framing is:

```text
QUESTION 1

What categories of quantization are structurally relevant?

    Weight?
    Activation?
    KV Cache?


QUESTION 2

Where in the graph can low precision be applied?

    Large GEMMs?
    Linear layers?
    Expert MLPs?

Where should higher precision probably remain?

    Norm?
    Softmax?
    reductions?
    sensitive merge points?
```

That is the architecture-driven stage.

## 13. What Architecture Does Not Decide

Architecture alone generally does not decide:

```text
INT4 vs INT8 vs FP8
AWQ vs GPTQ
static vs dynamic activation scaling
whether INT4 will preserve quality
whether FP8 will be faster
whether a runtime has the required kernel
whether the target GPU has native support
```

Those decisions belong to later stages.

## 14. Overall Progression

```text
Step 1 — Model Architecture
    WHAT can be quantized?
    WHERE can low precision be applied?

Step 2 — Runtime Engine
    WHICH quantized representations and execution paths exist?

Step 3 — Hardware
    WHICH runtime paths have useful native / optimized support?

Step 4 — Empirical Validation
    WHICH candidate actually satisfies quality and performance?
```

## 15. Interview Summary

> **I first inspect the model architecture to identify the quantization surfaces: weights, activations, and persistent KV state. Then I inspect the graph at operator level to determine where low precision is structurally appropriate and where higher precision should remain, which gives me a candidate mixed-precision map. At this stage I am not yet choosing INT4 versus FP8 or AWQ versus GPTQ. Those decisions come later from runtime support, hardware capability, and empirical quality/performance validation.**

## 16. Locked Principle

> **Model architecture determines the quantization surfaces and candidate precision boundaries. It usually does not determine the final quantization format.**
