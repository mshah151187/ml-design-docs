# Attention Backends for LLM Serving

**Interview question:** What is an attention backend, which major backend families exist, and why would a serving runtime use different backends for different models, GPUs, Prefill/Decode phases and KV representations?

The first distinction to remember is:

> **Attention backend != attention algorithm.**

FlashAttention is an algorithm/kernel family. An attention backend is the serving/runtime implementation that decides how attention executes for a concrete combination of:

```text
Q shape
KV length
MHA / MQA / GQA / MLA
Prefill vs Decode
paged / contiguous KV
KV dtype
mask / sparsity semantics
GPU architecture
head dimension
block size
cache layout
```

A backend can itself dispatch among multiple kernels.

---

# 1. Why Multiple Backends Exist

Mathematically, standard attention is:

\[
Attention(Q,K,V)
=
softmax(QK^T/\sqrt d)V
\]

But these two executions are very different.

Prefill:

```text
Q length  = 8192
KV length = 8192

many Q rows
many K/V rows
large 2-D parallel workload
```

Decode:

```text
Q length  = 1
KV length = 65536

almost no Q parallelism
huge KV history to scan
```

Likewise, cache representations differ:

```text
contiguous KV
paged KV
quantized KV
MLA compressed latent state
sparse/indexed KV
```

No single kernel decomposition is best for all of these.

Modern serving engines therefore expose an **attention backend registry plus compatibility/heuristic selection**.

Current vLLM, for example, maintains multiple backend families and selects a compatible high-priority backend based on architecture, GPU generation, dtype, head size and feature support.

---

# 2. Useful Taxonomy

| Backend / family | Primary problem it solves |
|---|---|
| **FlashAttention** | IO-efficient dense attention, especially long-Q Prefill |
| **PagedAttention** | attention directly over non-contiguous paged serving KV |
| **FlashInfer** | inference-first toolkit with distinct Prefill/Decode/paged paths |
| **TensorRT-LLM Context FMHA** | fused context/Prefill attention |
| **TensorRT-LLM Masked MHA** | Decode-specialized generation attention |
| **XQA** | specialized MQA/GQA Decode |
| **Triton Attention** | programmable, broad serving-kernel coverage |
| **FlexAttention** | programmable attention semantics / masks / block sparsity |
| **PyTorch SDPA / xFormers / cuDNN dispatchers** | unified APIs dispatching to optimized implementations |
| **MLA backends** | DeepSeek-style compressed latent attention |
| **Sparse attention backends** | avoid attending to all KV blocks/tokens |
| **Quantized-cache backends** | efficiently attend over compressed KV |

---

# 3. FlashAttention

FlashAttention is the useful baseline.

Naive attention conceptually does:

```text
QK^T
  |
  v
write large score matrix to HBM
  |
softmax
  |
  v
read/write intermediate
  |
  v
multiply by V
```

FlashAttention tiles the computation and uses online softmax:

```text
load Q/K/V tiles
      |
      v
partial QK
      |
online softmax state
      |
accumulate output
      |
      v
avoid full QK^T materialization in HBM
```

### What It Optimizes

```text
HBM IO
large dense attention
many Q and K/V positions
```

This is particularly natural during Prefill:

```text
q_len = thousands
kv_len = thousands
```

### Important Point

A FlashAttention backend may itself refer to multiple implementations or generations:

```text
FA2
FA3
FA4
varlen variants
paged variants
hardware-specific kernels
```

Current vLLM uses hardware-sensitive backend selection rather than treating one FlashAttention path as universal.

---

# 4. PagedAttention

PagedAttention attacks a different serving problem:

> **How do we perform attention over KV that is physically stored in non-contiguous pages?**

Suppose request A's logical KV blocks are:

```text
logical block 0
logical block 1
logical block 2
logical block 3
```

but physical GPU pages are:

```text
block 0 -> physical page 91
block 1 -> physical page 17
block 2 -> physical page 204
block 3 -> physical page 8
```

The runtime maintains:

```text
BlockTable[A] = [91, 17, 204, 8, ...]
```

A naive path would gather all KV into contiguous temporary memory:

```text
paged KV
   |
   | copy / gather
   v
contiguous KV
   |
   v
attention
```

That defeats much of the memory-management advantage.

Paged attention lets the kernel follow the block table directly:

```text
Q
 |
 v
block table
 |
 +--> page 91
 +--> page 17
 +--> page 204
 +--> page 8
 |
 v
attention reduction
```

### FlashAttention vs PagedAttention

```text
FlashAttention:
    How do I avoid expensive dense-attention intermediates?

PagedAttention:
    How do I compute attention directly over a paged KV working set?
```

Modern kernels can combine both ideas, so these are not mutually exclusive technologies.

---

# 5. FlashInfer

FlashInfer is better thought of as an **inference attention toolkit/runtime** than one single algorithm.

Its current API exposes distinct operations for:

```text
single-request Decode
batched Decode
paged Decode

Prefill
batched Prefill
paged Prefill

TRT-LLM-backed Decode variants
XQA Decode variants
cuDNN-backed variants

MLA / sparse / quantized-related paths
```

For example, current FlashInfer exposes separate:

```text
BatchDecodeWithPagedKVCacheWrapper
BatchPrefillWithPagedKVCacheWrapper
```

### Why This Matters

FlashInfer can reason about:

```text
q_len
KV length
paged block tables
batch metadata
CUDA graph capture
KV layout HND/NHD
backend selection
```

and execute a phase-appropriate path.

Think:

```text
FlashAttention
    = important algorithm/kernel family

FlashInfer
    = inference-serving attention toolkit
      containing multiple algorithms/backends
```

---

# 6. Decode Has a Different Parallelization Problem

Long-context Decode can look like:

```text
Q:
    [1, d]

K/V:
    [64000, d]
```

There may not be enough parallelism in:

```text
batch × heads
```

to fill a large GPU.

A common Decode technique is to split the KV dimension:

```text
64K KV history

0---------8K
8K-------16K
16K------24K
...
```

Different CUDA blocks process separate chunks:

```text
block 0 -> partial attention on segment A
block 1 -> partial attention on segment B
block 2 -> partial attention on segment C
...
```

Then merge:

```text
partial softmax statistics
+
partial output accumulators
       |
       v
final attention output
```

This family of techniques is often called:

```text
split-KV
multi-block attention
flash decoding
```

It is a very different decomposition from Prefill's large Q×K tiling problem.

---

# 7. TensorRT-LLM Context FMHA

TensorRT-LLM explicitly separates context and generation attention paths.

During context/Prefill, its fused MHA path executes MHA/MQA/GQA as a fused operation.

For larger sequences, the implementation uses FlashAttention-style algorithms. It also supports variants such as FP8 Context FMHA and paged Context FMHA on supported configurations.

### Prefill Problem Being Solved

```text
many Q positions
many K/V positions
large fused operation
avoid QK^T materialization
high Tensor-Core utilization
```

Context FMHA is therefore a production backend/dispatcher for the **Prefill attention regime**.

---

# 8. TensorRT-LLM Masked MHA

Generation/Decode uses a dedicated masked multi-head attention kernel.

The kernel can perform several operations on-the-fly, including:

```text
Q/K/V preprocessing
QKV bias
RoPE
KV load/store
quantization / dequantization
attention
```

This is valuable because Decode is a tiny repeated operation; extra launches and extra memory passes matter.

### Multi-Block Mode

If:

```text
batch_size × num_heads
```

does not create enough CUDA blocks to occupy the GPU, the implementation can split one head's history across several thread blocks:

```text
one head's 64K KV history
        |
        +--> block A
        +--> block B
        +--> block C
        +--> block D
```

TensorRT-LLM uses heuristics to choose when multi-block execution becomes profitable.

---

# 9. XQA

XQA is a Decode optimization aimed particularly at MQA/GQA.

Consider GQA:

```text
32 Q heads
8 KV heads
```

Multiple Q heads share each KV head:

```text
Q0 --\
Q1 ---\
Q2 ----> KV head 0
Q3 ---/

Q4 --\
...
```

A generic MHA kernel may not organize memory/work optimally around this sharing.

XQA specializes:

```text
KV loading
query-group processing
shared-memory usage
thread-block mapping
paged-KV traversal
```

for those MQA/GQA relationships.

Current TensorRT-LLM uses heuristics to choose between XQA and masked MHA and supports several paged-KV block sizes and cache dtypes on the XQA path.

### Mental Model

> **XQA is a Decode-specialized MQA/GQA kernel family, not a universal replacement for FlashAttention.**

---

# 10. A Backend Can Contain Multiple Algorithms

A runtime backend might internally do:

```text
attention request
       |
       v
Prefill?
   |
   +--> context FMHA
          |
          +--> short-context strategy
          +--> FlashAttention-style strategy

Decode?
   |
   +--> masked MHA
   |       |
   |       +--> normal
   |       +--> multi-block
   |
   +--> XQA when heuristic says it wins
```

Therefore:

```text
BACKEND
    !=
ONE KERNEL
```

A backend is often:

```text
compatibility validation
metadata preparation
workspace management
kernel dispatch
```

plus one or more optimized kernels.

---

# 11. Triton Attention

Triton lets developers express GPU kernels using a higher-level DSL rather than maintaining every path as handwritten CUDA/CUTLASS.

A Triton attention kernel can directly implement:

```text
load Q tile
load K/V page
dot
online softmax
accumulate
```

with tunable parameters such as:

```text
BLOCK_M
BLOCK_N
warps
pipeline stages
KV splits
layout
```

### Why Use Triton?

Flexibility.

It is attractive for:

```text
different head dimensions
unusual KV formats
different masks
new model architectures
different page sizes
rapid kernel iteration
```

### Tradeoff

A highly specialized hand-tuned kernel can spend enormous engineering effort on one exact GPU/workload.

Triton gives much faster programmability and broad coverage.

That is a design tradeoff, not a universal speed ranking.

Current vLLM contains a high-performance Triton attention backend alongside FlashAttention and FlashInfer.

---

# 12. FlexAttention

FlexAttention solves another problem:

> **How do we express non-standard attention semantics without writing a new custom CUDA kernel for every mask or scoring rule?**

It supports programmable:

```text
score_mod
mask_mod
BlockMask
```

Examples:

```text
causal attention
sliding-window attention
prefix-LM masks
document masks
attention sinks
block-sparse patterns
custom score transformations
```

PyTorch's `BlockMask` represents sparsity at block granularity so kernels can skip entire Q/KV blocks.

Current FlexAttention can itself choose among implementation paths, including Triton and specialized short-Q Decode execution under supported configurations.

### FlashAttention vs FlexAttention

```text
FlashAttention:
    aggressively optimize known dense attention structures.

FlexAttention:
    keep efficient generated kernels while making
    attention semantics programmable.
```

---

# 13. PyTorch SDPA

PyTorch's:

```python
scaled_dot_product_attention(Q, K, V, ...)
```

is best thought of as a **dispatcher API**.

Depending on shape, hardware and features, PyTorch may route to:

```text
FlashAttention
memory-efficient attention
cuDNN attention
math/reference implementation
```

### Why Useful?

Framework code expresses one semantic attention operation.

The framework decides which implementation is legal and performant.

### Why Serving Engines Need More Control

vLLM/TRT-LLM additionally reason about:

```text
paged KV
block tables
prefix caching
quantized KV
continuous batching
Decode q_len≈1
KV transfer compatibility
```

so they often integrate specialized serving backends directly.

---

# 14. xFormers Memory-Efficient Attention

xFormers historically provides a broad memory-efficient-attention abstraction and dispatcher.

Its value includes:

```text
multiple optimized operator implementations
ragged/block-diagonal patterns
attention-bias abstractions
dispatch based on features/shapes
```

Modern xFormers can itself use FlashAttention-family implementations.

Therefore:

```text
xFormers vs FlashAttention
```

is not always a strict either/or comparison.

It can be:

```text
xFormers dispatcher
       |
       v
FlashAttention implementation
```

---

# 15. cuDNN Attention

cuDNN supplies NVIDIA-vendor-optimized SDPA paths.

Conceptually:

```text
framework/runtime
    |
    | Q/K/V + metadata
    v
cuDNN
    |
    | internal heuristics / architecture tuning
    v
optimized GPU execution
```

The value is that low-level architecture tuning can remain inside a vendor library.

FlashInfer currently exposes cuDNN-backed Prefill/Decode attention paths for supported configurations.

---

# 16. MLA Backends

DeepSeek-style Multi-head Latent Attention changes the cache representation itself.

Ordinary MHA/GQA mental model:

```text
cache K/V per attention head
```

MLA uses compressed latent state and model-specific projection structure.

That creates a separate backend family.

Current vLLM contains several MLA implementations, including families such as:

```text
FlashInfer MLA
TokenSpeed MLA
CUTLASS MLA
FlashAttention MLA
FlashMLA
Triton MLA
sparse MLA variants
```

### Why So Many?

The best implementation depends on:

```text
GPU generation
KV dtype
query-head count
sparse vs dense MLA
head dimensions
Prefill vs Decode
```

Current vLLM has a separate priority order for MLA rather than reusing its standard MHA/GQA order.

### KV Compatibility Consequence

```text
standard GQA KV
    !=
MLA latent cache
```

This may be a semantic representation difference, not merely a tensor permutation.

---

# 17. Sparse Attention Backends

For very long contexts, reading every KV token can dominate Decode.

Sparse attention changes the problem:

```text
128K KV history
       |
       | select useful blocks/tokens
       v
smaller effective attention set
```

Approaches include:

```text
sliding window
block sparse
learned / indexed retrieval
threshold-based skipping
sparse MLA
```

Now an attention backend may be responsible not only for:

```text
how do I compute attention?
```

but also:

```text
which KV regions participate?
```

Modern runtimes increasingly contain sparse attention or sparse MLA paths.

---

# 18. Quantized-KV Attention Backends

Suppose KV is stored in FP8 or another compressed representation.

Naive:

```text
compressed KV
     |
     v
dequantize full cache
     |
     v
ordinary attention
```

Better:

```text
load compressed KV tile
     |
dequantize / fuse
     |
dot / softmax / accumulate
```

The backend must understand:

```text
KV dtype
scale metadata
packing
vectorization
supported head dimensions
```

This is why backend support matrices include cache dtype and layout constraints.

It also directly affects P/D KV compatibility.

---

# 19. Current vLLM Backend Selection Example

As of August 2026, vLLM's standard MHA/MQA/GQA CUDA priority differs by GPU generation.

For Blackwell, the current order begins:

```text
FlashInfer
FlashAttention
Triton Attention
FlexAttention
TurboQuant
```

For Ampere/Hopper, it begins:

```text
FlashAttention
FlashInfer
Triton Attention
FlexAttention
TurboQuant
```

Do **not** memorize this order.

The lesson is:

> **Backend choice is hardware- and configuration-dependent, and the runtime validates compatibility before selecting a kernel family.**

MLA uses a separate backend priority list entirely.

---

# 20. Compare Backends by the Problem They Solve

## FlashAttention

```text
Problem:
    dense-attention IO and intermediate materialization

Best mental model:
    long-Q Prefill / dense attention
```

## PagedAttention

```text
Problem:
    serving KV is non-contiguous and managed through pages

Best mental model:
    block-table-aware attention
```

## FlashInfer

```text
Problem:
    production inference needs multiple phase-specific
    Prefill/Decode/paged paths

Best mental model:
    inference attention toolkit/runtime
```

## Context FMHA

```text
Problem:
    fuse and optimize context/Prefill attention

Best mental model:
    TensorRT-LLM Prefill path
```

## Masked MHA / XQA

```text
Problem:
    q_len≈1 and large KV history

Best mental model:
    TensorRT-LLM Decode path
```

## Triton

```text
Problem:
    need performant, rapidly programmable kernels
    across many shapes/features

Best mental model:
    flexible custom serving kernel
```

## FlexAttention

```text
Problem:
    attention semantics themselves vary

Best mental model:
    programmable masks / score modification / block sparsity
```

## MLA backends

```text
Problem:
    architecture and cache representation differ fundamentally

Best mental model:
    architecture-specific attention
```

---

# 21. Dimensions to Compare Any Two Backends

When comparing backends, ask:

## Q shape

```text
Prefill q_len large?
Decode q_len ~ 1?
```

## KV shape / length

```text
2K?
32K?
1M?
```

## KV storage

```text
contiguous?
paged?
latent?
quantized?
sparse?
```

## Attention architecture

```text
MHA?
MQA?
GQA?
MLA?
hybrid?
```

## Parallelization dimension

Does the kernel expose work across:

```text
Q tiles
heads
batch
KV tiles / splits
pages
```

## Intermediate behavior

```text
materialize QK^T?
online softmax?
```

## Fusion

Does it fuse:

```text
RoPE
QKV bias
KV insertion
dequantization
attention
output conversion
```

## GPU generation

```text
Ampere
Hopper
Blackwell
```

## KV dtype

```text
BF16
FP16
FP8
other compressed cache
```

## Layout

```text
HND
NHD
paged engine layout
MLA latent
```

## Generality

```text
highly specialized hand-tuned path
vs
general programmable backend
```

---

# 22. Concrete P/D Example

Suppose:

```text
Llama-like GQA
H100
paged KV
64 active sequences
32K context
```

## Prefill

```text
q_len = 8K
```

An appropriate path may be:

```text
FlashAttention / Context FMHA
```

because:

```text
lots of Q parallelism
large dense matrix work
online softmax
Tensor-Core opportunity
```

## Decode

Now:

```text
q_len = 1
kv_len = 32K
```

A Decode-specialized path such as:

```text
XQA
FlashInfer paged Decode
masked-MHA multi-block
```

may be preferable because:

```text
Q parallelism is tiny
KV scan dominates
GQA shares KV heads
block-table traversal matters
split-KV / multi-block exposes more parallelism
```

The model mathematics did not change.

The best GPU work decomposition did.

---

# 23. Connection to P/D KV Compatibility

Suppose:

```text
P backend:
    context/FlashAttention-style backend
    KV layout = HND

D backend:
    FlashInfer/XQA-style backend
    KV layout = NHD
```

The mathematical K/V may be perfectly compatible.

Physical representation differs:

```text
P representation
      |
      | exact HND -> NHD transform
      v
D representation
```

But:

```text
P:
    conventional GQA cache

D:
    MLA latent backend
```

may imply a semantic state difference, not merely a transpose.

So attention backend selection participates in the broader KV compatibility contract.

---

# 24. Why Backend Selection Is a ServingProfile Property

An attention backend affects:

```text
Prefill throughput
Decode TPOT
KV block/layout requirements
supported cache dtype
head-size constraints
CUDA graph compatibility
workspace requirements
GPU architecture support
```

Therefore it belongs in the worker's structural ServingProfile.

But the backend name alone may not fully define transfer compatibility.

A stronger identity is:

```text
attention backend
+
backend/version
+
cache-layout contract
+
KV dtype/quantization
+
attention architecture
```

---

# 25. Short Interview Answer

> **I think of an attention backend as broader than an attention algorithm. FlashAttention solves IO-efficient dense attention by tiling Q/K/V and avoiding materialization of the full score matrix, which makes it especially natural for Prefill. Serving systems also need attention directly over paged KV, which is the PagedAttention problem. Decode has a very different shape—roughly one query token scanning a huge cached history—so runtimes use split-KV or multi-block strategies and specialized kernels such as TensorRT-LLM masked MHA or XQA for GQA/MQA. FlashInfer is more like an inference attention toolkit with separate paged Prefill and Decode paths. Triton gives kernel programmability, while FlexAttention makes attention semantics programmable through masks and score modifications. Architectures such as MLA need separate backends because even the cache representation changes. A production runtime therefore chooses a backend based on phase, Q/KV shape, cache layout/dtype, attention architecture and GPU generation rather than assuming one kernel is universally best.**

---

# 26. Memory Note

```text
ATTENTION BACKEND != ATTENTION ALGORITHM


FlashAttention
    dense-attention IO optimization
    especially natural for Prefill


PagedAttention
    direct attention over paged/non-contiguous KV


FlashInfer
    inference toolkit:
        Prefill
        Decode
        paged KV
        backend dispatch


TRT Context FMHA
    fused Prefill/context path


TRT Masked MHA / XQA
    Decode-specific
    q_len ~ 1
    paged KV
    split/multi-block
    GQA/MQA specialization


Triton
    programmable high-performance kernel implementation


FlexAttention
    programmable masks / score semantics / block sparsity


SDPA / xFormers / cuDNN
    dispatcher/library abstractions


MLA / sparse / quantized backends
    architecture/cache-representation specialization


COMPARE ANY BACKEND BY:
    Q length
    KV length
    paged vs contiguous
    MHA/GQA/MLA
    KV dtype
    cache layout
    parallelization dimension
    fusion
    GPU generation
    generality
```

---

# References

- vLLM — Attention Backend Feature Support and backend registry, August 2026.
- NVIDIA TensorRT-LLM — Context FMHA, generation masked MHA, multi-block mode and XQA.
- FlashInfer — Attention Kernels API, including distinct paged Prefill and Decode paths.
- PyTorch — FlexAttention and BlockMask documentation.
