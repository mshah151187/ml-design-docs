# Runtime-Driven Quantization

> **Purpose:** Explain the second narrowing stage in quantization selection: after model architecture identifies *what can be quantized* and *where low precision can be applied*, the runtime determines which concrete low-precision representations it can actually load and execute.

---

# 1. Core Principle

The model-architecture pass gives us a structural quantization map.

Example:

```text
Large Linear / GEMM weights
    -> weight quantization applicable
    -> low precision allowed

Large GEMM activations
    -> activation quantization potentially applicable

Autoregressive KV state
    -> KV-cache quantization applicable

Norm / Softmax / reductions
    -> usually remain at higher precision
```

The runtime pass now asks:

> **Which of these low-precision representations can this runtime actually interpret and execute for the relevant operator types?**

So the runtime narrows the architecture-derived search space.

```text
Architecture-Derived Quantization Space
        |
        v
Runtime Compatibility Evaluation
        |
        v
Runtime-Supported Quantization Space
```

---

# 2. Runtime Compatibility Has Two Main Parts

For a quantized candidate to be usable, the runtime must satisfy both:

```text
1. Representation Compatibility

2. Execution Compatibility
```

Both are required.

---

# 3. Representation Compatibility

Representation compatibility means:

> **Can the runtime correctly interpret the quantized artifact?**

For weight quantization, the artifact may contain:

```text
packed low-bit weights

scales

group size

scale granularity

zero points, if applicable

symmetric/asymmetric metadata

weight packing/layout

module-level quantization metadata
```

For example, two models can both be called:

```text
INT4
```

while having different representations.

Example:

```text
Model A

weights:
    INT4

group_size:
    128

scheme:
    symmetric

metadata:
    one scale per group

packing:
    layout A
```

versus:

```text
Model B

weights:
    INT4

group_size:
    64

scheme:
    asymmetric

metadata:
    scale + zero point

packing:
    layout B
```

Both are INT4 weight quantization.

But a runtime that understands representation A does not automatically understand representation B.

Therefore a runtime compatibility matrix cannot simply contain:

```text
vLLM -> INT4 = YES
```

It must conceptually describe something closer to:

```text
operator:
    Linear

weight representation:
    INT4

activation representation:
    BF16

group sizes:
    supported set

scale layout:
    supported forms

zero point:
    supported / unsupported

packing format:
    supported formats
```

---

# 4. Execution Compatibility

Even if a runtime can parse the artifact, it must also have an execution path that can consume it efficiently and correctly.

For a weight-only candidate:

```text
W4A16
```

the runtime must be able to perform something conceptually like:

```text
BF16 / FP16 Activation
          x
Packed INT4 Weight
          |
          v
Scale / Dequantization Logic
          |
          v
Matrix Multiplication
```

This usually requires an appropriate backend/kernel implementation.

The kernel needs to understand:

```text
packed weight layout

group boundaries

scales

zero points

dequantization semantics

accumulation precision
```

So:

```text
Representation Compatible
        +
Execution Compatible
        =
Runtime-Supported Quantization Candidate
```

---

# 5. Runtime Compatibility Is Quantization-Technique Agnostic

This is an important distinction.

Techniques such as:

```text
AWQ
GPTQ
```

describe **how the quantized weights are produced offline**.

For example:

```text
AWQ

BF16 weights
+
activation statistics
        |
        v
choose scaling / protect important channels
        |
        v
INT4 weight representation
```

and:

```text
GPTQ

BF16 weights
+
calibration / second-order information
        |
        v
choose quantized weights that minimize output error
        |
        v
INT4 weight representation
```

At serving time, the runtime primarily cares about the resulting representation:

```text
packed INT4 weights

scales

group metadata

zero-point semantics

layout
```

It does not need to reproduce the AWQ or GPTQ optimization algorithm during inference.

Therefore the runtime-level question is not primarily:

```text
"Was this model produced by AWQ or GPTQ?"
```

The more fundamental question is:

```text
"Can I interpret this quantized representation,
and do I have an execution backend that can consume it?"
```

---

# 6. Important Nuance

This does **not** mean every runtime can load every AWQ or GPTQ artifact automatically.

Two offline quantization techniques may produce different:

```text
packing layouts

scale layouts

group sizes

zero-point behavior

module metadata
```

Therefore a runtime may expose support under names such as:

```text
AWQ

GPTQ

compressed-tensors

ModelOpt

other format/integration names
```

But from the systems-design perspective, these names are often shorthand for:

> **The runtime knows how to interpret the representation produced by that recipe and has an execution path for it.**

The deeper abstraction remains:

```text
Offline Quantization Technique
        |
        v
Concrete Quantized Representation
        |
        v
Runtime Representation Compatibility
        |
        v
Runtime Execution Compatibility
```

---

# 7. Architecture Output Mapped to Runtime Capabilities

Suppose architecture analysis produced:

```text
Q/K/V Linear weights
    -> low-precision weight candidate

MLP weights
    -> low-precision weight candidate

Linear activations
    -> low-precision activation candidate

KV state
    -> low-precision KV candidate

Norm / Softmax
    -> higher precision
```

The runtime pass maps each operator/state to its supported representations.

Conceptually:

| Architecture Surface | Runtime Question |
|---|---|
| Dense Linear weights | Which weight precisions/layouts can this runtime load and execute? |
| Dense Linear activations | Which activation precisions/scaling modes can it execute? |
| MoE expert weights | Which quantized expert representations are supported? |
| KV cache | Which KV dtypes/quantized layouts can the attention backend consume? |
| Mixed precision boundaries | Can different modules use different quantization treatments? |

The result is a runtime-supported subset.

---

# 8. Example Compatibility Matrix

A conceptual runtime matrix might look like:

| Operator / State | Runtime A | Runtime B | Runtime C |
|---|---|---|---|
| Dense Linear weights | BF16, FP8, INT8, INT4 variants | supported variants | supported variants |
| Dense Linear activations | BF16, FP8/INT8 variants | supported variants | supported variants |
| KV cache | BF16, FP8 variants | supported variants | supported variants |
| MoE expert weights | runtime-specific | runtime-specific | runtime-specific |
| Norm / Softmax | higher precision paths | higher precision paths | higher precision paths |

The exact entries are runtime-version dependent.

The architecture does not care how the runtime internally implements these paths.

The runtime compatibility layer simply narrows the candidate space to representations it can actually execute.

---

# 9. Example: Weight Quantization

Architecture says:

```text
Large Linear weights
    -> low precision is structurally applicable
```

Possible representations might include:

```text
FP8 weights

INT8 weights

INT4 weights
```

The runtime now checks:

```text
Can I load FP8 weights?

Can I load INT8 weights?

Can I load INT4 weights?

For INT4:
    which group sizes?
    which scale layouts?
    which packing formats?
    which zero-point modes?

Do I have a backend/kernel for the resulting W4A16 path?
```

Suppose the runtime supports:

```text
FP8
INT8
INT4 group-wise
```

but not a particular INT4 asymmetric layout.

Then that asymmetric INT4 candidate disappears from the search space.

---

# 10. Example: Activation Quantization

Architecture says:

```text
Large GEMM activations
    -> activation quantization is structurally applicable
```

The runtime now asks which activation forms it supports.

For example:

```text
FP8 dynamic activation scaling?

FP8 static activation scaling?

INT8 activation quantization?

per-tensor scaling?

per-token scaling?

block-wise scaling?
```

Architecture only said:

```text
activation quantization may be useful here
```

The runtime determines which concrete variants are executable.

---

# 11. Example: KV Quantization

Architecture says:

```text
Autoregressive decoder
    ->
persistent KV cache exists
    ->
KV quantization is structurally applicable
```

The runtime then asks:

```text
Which KV dtypes are supported?

Can the attention backend directly consume quantized KV?

Is dequantization fused into attention?

Are K and V scaled separately?

What scale granularity is supported?
```

If the runtime has no quantized-KV execution path, then KV quantization is removed from the candidate space even though it is architecturally meaningful.

---

# 12. Mixed Precision and Runtime Support

Architecture may produce a candidate mixed-precision map:

```text
QKV Linear:
    low precision candidate

MLP:
    low precision candidate

Norm:
    higher precision

Softmax:
    higher precision
```

The runtime must be able to honor those boundaries.

So the runtime asks:

```text
Can different module classes use different precisions?

Can selected layers be excluded from quantization?

Can some modules remain BF16 while others use INT4/FP8?

Can activation quantization be enabled only for selected operators?
```

If the runtime supports only a coarse global mode, it may restrict the architecture-derived mixed-precision plan.

---

# 13. Runtime Compatibility Matrix — Better Mental Model

Instead of thinking:

```text
Runtime
    ->
supported bit widths
```

think:

```text
Runtime Compatibility
    =
Operator Type
    x
Weight Representation
    x
Activation Representation
    x
Scale / Group Layout
    x
Packing Format
    x
Runtime Version
```

Conceptually:

```text
Linear
+
INT4 weights
+
BF16 activations
+
group_size = 128
+
symmetric scales
+
packing format X
+
runtime version Y
        |
        v
SUPPORTED / UNSUPPORTED
```

This is much closer to what the runtime actually needs to decide.

---

# 14. Artifact Materialization Follows Runtime Compatibility

Suppose architecture says:

```text
weight quantization applicable
```

and the runtime compatibility stage says:

```text
Runtime A:
    supports INT4 representation A

Runtime B:
    supports INT4 representation B
```

Then artifact materialization can generate runtime-compatible artifacts from the canonical model:

```text
                 Canonical BF16 Model
                        |
              +---------+---------+
              |                   |
              v                   v
    Runtime-A INT4 Artifact   Runtime-B INT4 Artifact
```

Sometimes this may be a layout/packing conversion.

In other cases, the target quantization semantics differ enough that the platform may need to return to the canonical higher-precision model and requantize.

Therefore:

```text
Runtime-compatible artifact generation
    != always simple file-format conversion
```

The canonical high-precision model remains the source of truth.

---

# 15. What the Runtime Stage Does Not Decide

The runtime stage answers:

> **Can I execute this representation?**

It does not yet answer:

```text
Will FP8 be faster than INT4 on H100?

Will this representation be efficient on A100?

Will the candidate satisfy the quality requirement?

Will this candidate minimize cost/token?
```

Those questions belong to later stages:

```text
Hardware Capability
        |
        v
Empirical Benchmark / Quality Evaluation
        |
        v
Objective-Specific Search
```

---

# 16. Complete Narrowing Flow

```text
+------------------------------------------------------+
|                 MODEL ARCHITECTURE                   |
|                                                      |
| What can be quantized?                               |
| Where can low precision exist?                       |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|       ARCHITECTURE-DERIVED QUANTIZATION SPACE        |
|                                                      |
| Weight surfaces                                      |
| Activation surfaces                                  |
| KV surfaces                                          |
| Candidate mixed-precision boundaries                 |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|                    RUNTIME PASS                      |
|                                                      |
| Representation compatibility                         |
|   - bit width / dtype                                |
|   - group size                                       |
|   - scale layout                                     |
|   - zero point                                       |
|   - packing                                          |
|                                                      |
| Execution compatibility                              |
|   - operator support                                 |
|   - backend/kernel                                   |
|   - accumulation path                                |
|   - mixed precision support                          |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|          RUNTIME-SUPPORTED QUANTIZATION SPACE        |
|                                                      |
| Only representations the runtime can load            |
| and execute remain                                   |
+------------------------------------------------------+
```

---

# 17. Interview Summary

A concise explanation is:

> **After architecture analysis tells me which parts of the model can use weight, activation, or KV quantization, I map those surfaces against a runtime compatibility matrix. The runtime narrows the space based on two things: representation compatibility—whether it understands the low-precision weights, scales, groups, zero points, and packing—and execution compatibility—whether it has an operator/backend/kernel that can consume that representation. This stage is fundamentally agnostic to how the offline quantizer arrived at those weights. AWQ and GPTQ are offline quantization techniques; at serving time the runtime mainly cares that the resulting INT4 representation and metadata match a format and execution path it supports.**

---

# 18. Locked Principle

> **Architecture determines where low precision is structurally possible. Runtime compatibility determines which concrete low-precision representations can actually be loaded and executed.**
