# Quantization Crash Course for Model Optimization / Serving Interviews

> **Purpose:** Build a strong practical understanding of quantization before connecting it back to Model Optimization platform decisions.

## 1. What Quantization Is

Neural-network inference commonly uses floating-point formats such as `FP32`, `BF16`, and `FP16`. Quantization represents some model values using lower-precision formats such as `FP8`, `INT8`, `INT4`, or `FP4`.

The goals are usually some combination of:

- less model memory
- less HBM bandwidth
- higher matrix-multiplication throughput
- more KV-cache capacity
- smaller communication volume
- lower inference cost

The main tradeoff is:

```text
precision decreases
    -> representation error increases
    -> possible model-quality degradation
```

The key observation is that neural-network inference often does not require every number to be represented with BF16-level precision.

## 2. Basic Quantization Math

For a floating-point value `x`, simple integer quantization can be written as:

```text
q = round(x / scale) + zero_point
```

Approximate dequantization:

```text
x ~= scale * (q - zero_point)
```

For symmetric quantization:

```text
zero_point = 0
q = round(x / scale)
x ~= q * scale
```

The original value normally cannot be represented exactly, which creates quantization error.

## 3. What Is an Activation?

For interview purposes:

> **Activation = the first runtime input tensor entering the network plus the intermediate tensors produced as data flows through the model.**

For a simple network:

```text
Input Data
    |
    v
Layer 1
    |
    v
Intermediate Tensor / Activation
    |
    v
Layer 2
    |
    v
Intermediate Tensor / Activation
```

For a Transformer there are many activations inside one block:

```text
hidden state
    |
    +--> Q projection output
    +--> K projection output
    +--> V projection output
    +--> attention output
    +--> MLP intermediate
    +--> residual output
```

Ordinary activations are transient. KV cache is derived from activations but persists and is reused across autoregressive decode steps, so it is usually treated separately.

## 4. Main Quantization Targets

The three major targets are:

```text
1. Weights
2. Activations
3. KV Cache
```

They solve different serving problems.

## 5. Weight Quantization

Example:

```text
BF16 weights -> INT4 weights
```

Weight quantization can reduce model storage, HBM footprint, weight-memory traffic, and sometimes communication volume.

For a 100B model, ignoring metadata and other overhead:

```text
BF16:  ~200 GB
INT8:  ~100 GB
FP8:   ~100 GB
INT4:  ~50 GB
```

This can change the feasible serving topology because a smaller representation may reduce the required tensor-parallel degree.

## 6. Activation Quantization

For a matrix multiplication:

```text
Y = XW
```

where `X` is the activation and `W` is the weight:

```text
W4A16 = 4-bit weights, 16-bit activations
W8A8  = 8-bit weights, 8-bit activations
W4A8  = 4-bit weights, 8-bit activations
```

Activation quantization is mainly useful because it can enable lower-precision matrix multiplication.

## 7. KV-Cache Quantization

KV memory grows roughly with:

```text
sequence length
* layers
* KV heads
* head dimension
* bytes/value
```

BF16 KV uses roughly 2 bytes/value; FP8 KV roughly 1 byte/value. So FP8 KV can roughly halve the raw KV element storage.

Benefits can include more concurrent sequences, longer contexts, more batching capacity, and less repeated KV-memory traffic.

Mental distinction:

```text
Weight quantization
    -> model-weight memory / weight bandwidth

Activation quantization
    -> lower-precision execution of current operations

KV quantization
    -> persistent per-token state / repeated decode bandwidth
```

## 8. Why Quantize Transient Activations?

Dynamic activation quantization is not primarily about storing activations more cheaply across requests.

It is about making the operation consuming the activation faster.

For:

```text
Y = XW
```

a dynamic path can look like:

```text
BF16 activation X
    -> compute scale
    -> quantize X
    -> FP8 / INT8 X
    -> low-precision GEMM
    -> Y
```

There is extra quantization work. It is worthwhile only when:

```text
cost(dynamic quantization)
    <
savings(low-precision GEMM)
```

For a large GEMM this can pay off. For a tiny operation it may not.

## 9. Why KV Quantization Is Different

Ordinary activation:

```text
created -> consumed -> discarded
```

KV cache:

```text
created -> stored -> reread on many future decode steps
```

So KV quantization has a strong persistent-memory benefit.

Dynamic activation quantization mainly targets compute and transient bandwidth. KV quantization mainly targets persistent memory, capacity, and repeated attention bandwidth.

## 10. Static vs Dynamic Quantization

In practical serving discussions, static vs dynamic applies mainly to activations because weights are fixed after training and are usually quantized once offline.

### Static activation quantization

```text
Calibration Dataset
    -> collect activation statistics
    -> choose scales / clipping
    -> deploy fixed quantization parameters
```

Advantages:
- no scale calculation on request path
- stable optimized execution

Tradeoff:
- one scale may not match every production input well

### Dynamic activation quantization

```text
Current Activation
    -> measure range/statistics
    -> compute scale
    -> quantize
    -> low-precision GEMM
```

Why dynamic? Because activation ranges can vary significantly across requests and tokens.

Dynamic scaling adapts to each activation distribution but adds runtime work.

## 11. Why Dynamic Quantization Can Still Be Fast

A naive path could require several kernels:

```text
find max
-> calculate scale
-> write quantized tensor
-> GEMM reads it
```

Optimized implementations try to fuse or tightly integrate this work:

```text
load activation
-> calculate scale / quantize
-> feed low-precision data into GEMM
```

This is why the actual kernel implementation matters as much as the numerical scheme.

## 12. PTQ vs QAT

### Post-Training Quantization (PTQ)

```text
trained BF16 model
    -> quantization
    -> FP8 / INT8 / INT4 model
```

No full retraining.

Advantages:
- comparatively cheap
- fast
- works on existing models
- natural fit for a post-training optimization platform

### Quantization-Aware Training (QAT)

Quantization effects are simulated during training/fine-tuning so the model learns to tolerate them.

QAT can preserve quality better at aggressive precision, but it requires training resources and changes the workflow significantly.

## 13. Scale Granularity

Possible weight-scale granularities:

```text
per-tensor
per-channel
per-group
```

Per-tensor is simple but coarse. Per-channel adapts better. Per-group, such as one scale per 128 weights, is common for aggressive INT4.

General tradeoff:

```text
finer scale granularity
    -> better numerical accuracy
    -> more metadata and kernel complexity
```

Activations may use per-tensor, per-token, or per-channel scaling.

## 14. Symmetric vs Asymmetric Quantization

Symmetric:

```text
range ~= [-a, +a]
zero_point = 0
```

Asymmetric uses a non-zero zero point and can better represent non-centered distributions.

Weights are often roughly centered around zero, so symmetric quantization is common. Activations can be harder because their distributions may contain strong asymmetry and outliers.

## 15. The Outlier Problem

Suppose most activations are in:

```text
[-2, +2]
```

but a few are around:

```text
+70
-55
```

If the scale must represent the whole range, most ordinary values occupy only a small portion of the low-precision range and lose resolution.

A central LLM quantization problem is:

> How do we preserve important outliers while still getting low-precision efficiency?

## 16. Clipping

If almost all values lie in `[-3, +3]` but there is a rare outlier at `+75`, clipping at a smaller threshold may improve precision for common values.

The tradeoff is:

```text
clipping error
vs
rounding/resolution error
```

## 17. Calibration

Calibration means running representative samples through the model to collect statistics used for scales, clipping thresholds, and other quantization parameters.

```text
Calibration Dataset
    -> model forward pass
    -> collect activation statistics
    -> choose quantization parameters
```

Calibration data should resemble production traffic because unrepresentative data can produce poor quantization parameters.

## 18. Weight-Only Quantization

A common LLM scheme is:

```text
W4A16
```

Meaning INT4 weights with BF16/FP16 activations.

Conceptual path:

```text
load packed INT4 weights
    -> unpack / dequantize
    -> multiply with BF16 activation
```

Good kernels fuse dequantization into the GEMM.

Important principle:

> Compression ratio does not automatically equal runtime speedup.

Poor kernels or expensive dequantization can erase the benefit.

## 19. FP8 Quantization

FP8 is an 8-bit floating-point format.

Common forms:

```text
E4M3
E5M2
```

Roughly:

```text
E4M3 -> more precision, smaller dynamic range
E5M2 -> larger dynamic range, less precision
```

FP8 can provide lower memory/bandwidth plus efficient low-precision compute on hardware that has strong FP8 execution support.

## 20. INT8

A common form is:

```text
W8A8
```

Both weights and activations participate in 8-bit execution.

INT8 is numerically easier than INT4, but LLM activation outliers make naive W8A8 difficult. This motivates techniques such as SmoothQuant.

## 21. SmoothQuant

For:

```text
Y = XW
```

SmoothQuant conceptually rewrites:

```text
Y = (X / s) * (sW)
```

The output is mathematically unchanged.

The idea is to reduce difficult activation outliers by moving some of the scaling burden into the weights.

Key mental model:

> Move quantization difficulty from activations into weights because weights are generally easier to quantize.

## 22. GPTQ

GPTQ is primarily a post-training weight quantization technique, commonly used for INT4.

Its key insight is that not all weight errors affect layer output equally. It uses approximate second-order/Hessian information to choose quantization decisions that reduce output error.

Interview mental model:

> GPTQ tries to minimize the behavioral effect of aggressive weight quantization rather than only minimizing raw weight reconstruction error.

## 23. AWQ

AWQ stands for Activation-aware Weight Quantization.

It uses activation statistics to identify weight channels that are especially important and scales/protects them during aggressive quantization.

Mental model:

```text
activation statistics
    -> identify important weight channels
    -> preserve them more accurately
    -> quantize the rest aggressively
```

## 24. AWQ vs GPTQ vs SmoothQuant

| Technique | Main target | Typical use | Core idea |
|---|---|---|---|
| AWQ | Weights | INT4 weight-only | Use activation statistics to protect important weight channels |
| GPTQ | Weights | INT4 weight-only | Minimize output impact using second-order information |
| SmoothQuant | Weights + activations | INT8 W8A8 | Move activation-outlier difficulty into weights |
| FP8 quantization | Weights and/or activations | FP8 | Use narrow floating point with efficient hardware execution |

## 25. Quantized Model Does Not Mean Everything Is Low Precision

An “INT4 model” can still use:

```text
linear weights: INT4
activations: BF16
LayerNorm: BF16
Softmax: BF16 / FP32
accumulation: higher precision
logits: BF16 / FP32
```

Always ask:

> Which tensors are quantized, and what precision is used for accumulation?

## 26. Quantization and Kernels

A representation is useful only if the runtime and GPU have an efficient execution path.

For example, `W4A16` is not enough. The runtime needs kernels that understand packed weights, scales, group sizes, zero points, and dequantization.

The systems-level view is:

```text
quantization representation
+
runtime kernel
+
hardware
+
workload
```

not bit width alone.

## 27. Decode vs Prefill

### Decode

Decode handles one/few new tokens at a time. Arithmetic intensity is relatively low and repeated weight streaming often matters greatly.

Weight compression such as INT4 can therefore help by reducing HBM traffic.

### Prefill

Prefill processes many prompt tokens with large GEMMs and can become compute-heavy.

Hardware-native FP8 matrix multiplication can therefore be especially attractive.

Useful mental model:

```text
W4A16
    often attractive for weight-bandwidth-sensitive decode

FP8 W8A8
    often attractive when native low-precision compute helps large GEMMs
```

This is workload-dependent, not an absolute rule.

## 28. Quantization and Parallelism

For a 100B model:

```text
BF16 ~= 200 GB
FP8  ~= 100 GB
INT4 ~= 50 GB
```

Lower precision can reduce the minimum tensor-parallel degree.

That can create second-order benefits:

```text
quantization
    -> smaller TP
    -> fewer collectives
    -> more replicas
    -> possible cost improvement
```

Therefore quantization and topology should be evaluated together.

## 29. Quantization and Quality

Different model components tolerate low precision differently.

Practical schemes may keep some layers at higher precision:

```text
most linear layers: INT4
lm_head: BF16
sensitive layer: BF16
```

The goal is not to make everything low precision. The goal is to place low precision where it improves efficiency without crossing the quality boundary.

## 30. Practical PTQ Workflow

```text
BF16 Canonical Model
    -> choose quantization recipe
    -> prepare calibration dataset
    -> collect statistics
    -> compute scales/clipping/group parameters
    -> generate quantized artifact
    -> load in target runtime
    -> quality evaluation
    -> performance benchmark
    -> accept / reject
```

## 31. What to Benchmark

For each quantized candidate measure:

```text
quality delta
model weight size
peak HBM
TTFT
TPOT
input-token throughput
output-token throughput
maximum concurrency
cost/token
startup/load time
runtime stability
```

## 32. Why Quantization Sometimes Fails to Help

Possible reasons include:

```text
runtime lacks optimized kernel
dequantization overhead dominates
workload bottleneck is elsewhere
attention / KV dominates
communication becomes the bottleneck
calibration is poor
quality loss forces fallback
scale/group metadata adds overhead
kernel shapes are inefficient
```

Therefore real serving optimization requires empirical benchmarking.

## 33. Interview Mental Checklist

When discussing a quantization technique, ask:

```text
1. What gets quantized?
   weights, activations, KV?

2. What representation?
   FP8, INT8, INT4?

3. What granularity?
   per-tensor, per-channel, per-group, per-token?

4. How are scales determined?
   static calibration or dynamic?

5. How is accuracy preserved?
   clipping, SmoothQuant, AWQ, GPTQ, mixed precision?

6. How is it executed?
   runtime, kernel, GPU?

7. What does it improve?
   memory, bandwidth, compute, capacity?

8. What can it hurt?
   quality, latency, dequant overhead, kernel efficiency?
```

## 34. Thirty-Second Interview Answer

> Quantization represents model values using lower precision, for example converting BF16 weights to FP8, INT8, or INT4. It can target weights, activations, or KV cache. Weight quantization reduces model memory and weight bandwidth; activation quantization can enable faster low-precision matrix multiplication; KV quantization reduces persistent per-token state. The important design choices are the numerical format, scale granularity, static versus dynamic activation scaling, calibration, outlier handling, and whether the runtime and GPU have efficient kernels for the representation. Techniques such as AWQ and GPTQ make aggressive INT4 weight quantization more practical, while SmoothQuant handles difficult activation outliers for W8A8-style quantization. Every candidate then needs empirical quality and serving-performance validation.

## 35. Connection Back to the Platform

At the architecture level, we can keep the statement simple:

> **Search Foundation selects applicable quantization candidates based on the input model, available runtime engines, available hardware, and hard serving requirements.**

The next technical topic is:

```text
How do:
    model architecture
    runtime engine
    hardware

and their combination

determine which quantization techniques are actually applicable
and which have an efficient execution path?
```
