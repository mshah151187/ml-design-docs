# Deep Dive — Prefill vs Decode Execution

**Interview question:** What really happens during Prefill and Decode, and why do they need different serving treatment?

The compact mental model is:

```text
PREFILL
    many tokens at once
    large GEMMs
    many Q/K/V vectors
    high arithmetic intensity
    compute-heavy
    TTFT

DECODE
    ~1 new token per active sequence per step
    small/skinny matrix shapes
    repeatedly reads weights + KV history
    lower arithmetic intensity
    bandwidth / KV-capacity sensitive
    TPOT
```

The important part is understanding *why*.

## 1. Start from one Transformer layer

At a high level:

```text
hidden states
     |
     +--> Q/K/V projections
     |
     +--> attention
     |
     +--> output projection
     |
     +--> MLP / MoE
     |
     v
next hidden states
```

The model weights are the same in Prefill and Decode. The execution shape is not.

## 2. Prefill dense-layer example

Suppose:

```text
prompt length = 8192 tokens
hidden size   = H
```

A dense projection has a shape conceptually like:

```text
X:
    [8192, H]

W:
    [H, 4H]

Y:
    [8192, 4H]
```

This is a large GEMM.

A GPU can load tiles of the weight matrix and reuse those values across many token rows:

```text
load W tile
   |
   +--> token row 1
   +--> token row 2
   +--> token row 3
   +--> ...
   +--> many more rows
```

So each byte moved from HBM can participate in a lot of arithmetic.

That means relatively high:

```text
FLOPs / byte
```

or **arithmetic intensity**.

Large GEMMs also map well onto Tensor Cores.

This is why Prefill is commonly characterized as compute-intensive.

## 3. Prefill attention example

For one attention head:

```text
Q = [8192, d]
K = [8192, d]
V = [8192, d]
```

Attention conceptually contains:

```text
Q × K^T
```

with a large amount of parallel work.

A naive implementation would create an enormous:

```text
[8192 × 8192]
```

intermediate attention matrix.

Modern context kernels such as FlashAttention/context FMHA avoid writing that full intermediate to HBM. They tile and fuse work such as:

```text
QK^T
softmax
softmax × V
```

to improve locality and reduce memory traffic.

TensorRT-LLM's context phase uses fused context-attention paths and can use FlashAttention-style algorithms for long sequences.

The Prefill kernel problem therefore looks like:

```text
many Q
many K
many V
large parallel tiles
high Tensor-Core opportunity
```

## 4. Prefill also creates persistent KV state

At every Transformer layer, Prefill computes and stores:

```text
K(token 1), V(token 1)
K(token 2), V(token 2)
...
K(token 8192), V(token 8192)
```

That becomes the KV cache.

After Prefill, future Decode steps can reuse the old K/V values rather than recomputing the entire prompt.

So Prefill does two important things:

```text
1. computes first-token state/logits
2. materializes the KV history needed by Decode
```

## 5. Decode dense-layer example

Now suppose 64 sequences are actively decoding.

Each sequence contributes roughly one new token to the next iteration.

The same projection now looks more like:

```text
X:
    [64, H]

W:
    [H, 4H]

Y:
    [64, 4H]
```

Compare:

```text
Prefill:
    [8192, H] × [H, 4H]

Decode:
    [64, H] × [H, 4H]
```

The same large weight matrix is used, but across far fewer token rows.

There is much less opportunity to amortize reading that weight over useful arithmetic.

That reduces arithmetic intensity.

## 6. Decode attention makes the memory behavior obvious

For one sequence, Decode has one new query:

```text
Q_new
```

but it attends to the entire prior context:

```text
K1 K2 K3 ... K8192
V1 V2 V3 ... V8192
```

Conceptually:

```text
one new Q
    |
    v
scan cached K history
    |
    v
attention weights
    |
    v
scan cached V history
```

Then the next output token does it again, against an even larger cache.

Therefore each Decode step repeatedly streams:

```text
model weights
+
KV history
```

while producing only one token per active sequence.

This makes Decode highly sensitive to:

```text
HBM bandwidth
KV-cache capacity
memory-access efficiency
sequence concurrency
```

That is why Decode is commonly called memory-bandwidth / memory-capacity dominated.

## 7. "Decode is memory-bound" is a first-order model

It is not an absolute law.

If Decode batch size becomes very large:

```text
1 token × 1000 active sequences
```

the dense layers gain more weight reuse, so Decode compute utilization improves.

Likewise, very long-context Prefill attention can become increasingly influenced by attention memory behavior.

The precise statement is:

> **Relative to Prefill, Decode usually has lower arithmetic intensity and is far more sensitive to weight/KV memory traffic and persistent KV capacity.**

## 8. End-to-end example

Request:

```text
input  = 8K tokens
output = 5 tokens
```

### Prefill

```text
tokens 1 ... 8192
      |
      v
process many token positions through all layers
      |
      +--> large dense GEMMs
      +--> context attention
      +--> MLP
      +--> create KV for every layer/token
      |
      v
first output token
```

### Decode step 1

```text
new token
   |
   +--> dense projections
   +--> one new Q
   +--> read KV history 1...8192
   |
   v
next token
```

### Decode step 2

```text
new token
   |
   +--> read KV history 1...8193
   |
   v
next token
```

This autoregressive loop repeats until generation ends.

## 9. Why Decode residency is expensive

A decoding request occupies KV memory for its entire lifetime:

```text
start generation
      |
      | KV stays resident and grows
      |
      | token
      | token
      | token
      |
      v
finish generation
```

At high concurrency:

```text
many sequences
×
long contexts
×
KV per layer
×
many layers
```

can make KV capacity the admission bottleneck.

The engine may have unused theoretical compute but still be unable to admit another long sequence because it lacks KV blocks.

This is one reason Decode profiles care strongly about:

```text
KV memory fraction
paged KV
KV precision
TP effects on memory
maximum active sequences
```

## 10. Why Prefill batch behavior differs

Very small Prefill work can underutilize a large GPU:

```text
tiny prompt/batch
    -> launch/scheduling overhead significant
    -> Tensor Cores not fully fed
```

Increasing prompt-token work:

```text
more prompt tokens
    -> larger GEMMs
    -> GPU saturation
```

But waiting to form an enormous batch adds queueing:

```text
more batching
    -> more throughput
    -> possibly worse TTFT
```

A useful Prefill principle is:

> **Use the smallest prompt workload that already saturates the device, rather than maximizing batch size for its own sake.**

Current Dynamo tuning guidance explicitly recommends this strategy.

## 11. Why Decode batch behavior differs

Decode benefits strongly from active-sequence batching:

```text
sequence A -> one token
sequence B -> one token
sequence C -> one token
...
```

The model weights can be amortized across more concurrent sequences.

But increasing concurrency also:

```text
increases KV residency
can lengthen each iteration
can worsen TPOT
```

So Decode has a three-way tradeoff:

```text
throughput
vs
TPOT
vs
KV capacity
```

## 12. Specialized Prefill attention kernels

Prefill has many queries and lots of two-dimensional parallelism:

```text
Q Q Q Q Q Q Q Q
        ×
K K K K K K K K
```

Kernel goals include:

```text
large tiles
Tensor-Core efficiency
fused attention
avoid QK^T materialization
reduce HBM traffic
```

Examples:

```text
FlashAttention
Context FMHA
Paged Context FMHA
FP8 Context FMHA
```

TensorRT-LLM documents a fused context path and FlashAttention-style behavior for larger context sequences.

## 13. Specialized Decode attention kernels

Decode has a different shape:

```text
          one/few Q
              |
              v
K K K K K K K K K K
V V V V V V V V V V
```

Kernel goals include:

```text
efficient paged-KV traversal
coalesced KV reads
good occupancy at q_len≈1
split long KV history across thread blocks
special handling for MQA/GQA
low launch overhead
```

TensorRT-LLM uses generation-specific masked MHA and has XQA for supported MQA/GQA cases. Multi-block execution can improve occupancy when the normal decode shape does not expose enough parallel work.

This is strong evidence that Prefill and Decode are not simply different scheduling labels; production runtimes can use distinct GPU kernel paths.

## 14. Dense kernels differ too

Attention gets the headlines, but dense-layer shapes differ as well.

```text
Prefill:
    [thousands of tokens, H] × [H, output]
    -> large GEMM

Decode:
    [tens/hundreds of active sequences, H] × [H, output]
    -> much skinnier GEMM regime
```

The underlying GEMM implementation can choose different:

```text
tiling
fusion
communication overlap
kernel variants
```

for these shapes.

## 15. Why all of this implies different ServingProfiles

```text
PREFILL
-------
user metric:
    TTFT

execution:
    many tokens
    high arithmetic intensity
    large matrix operations

priority:
    compute throughput
    just enough batching for saturation
    avoid unnecessary communication
    context kernels


DECODE
------
user metric:
    TPOT

execution:
    ~1 token / sequence / step
    recurrent weight + KV reads
    persistent KV state

priority:
    HBM bandwidth
    HBM capacity
    sequence concurrency
    KV-efficient kernels
```

This naturally motivates different:

```text
TP
PP
batch limits
token budgets
KV policies
quantization
CUDA graph strategies
attention kernels
hardware
```

## 16. Interview reasoning chain

```text
1. Prefill processes many prompt tokens together.

2. Dense layers become large GEMMs.
   Weight tiles are reused across many rows.
   -> high arithmetic intensity.

3. Prefill attention also exposes many Q/K/V operations.
   -> context/FlashAttention-style kernels.

4. Decode processes about one new token per active sequence.

5. The same model weights are moved for much less arithmetic,
   and every new query scans existing KV history.

6. Therefore Decode has lower arithmetic intensity and is much
   more bandwidth/KV-capacity sensitive.

7. Different execution shapes justify different parallelism,
   memory, scheduler and kernel choices.
```

## 17. Short interview answer

> **During Prefill, thousands of prompt tokens can be processed together. Dense layers become large GEMMs, so a weight tile fetched from HBM can be reused across many token rows, giving high arithmetic intensity and strong Tensor-Core utilization. Attention also has many Q/K/V vectors, which makes context-FMHA or FlashAttention-style tiled kernels effective. During Decode, each iteration processes roughly one new token per active sequence. There is much less weight reuse, and every new query repeatedly reads the existing KV history, so the workload has lower arithmetic intensity and becomes far more sensitive to HBM bandwidth and KV capacity. That difference is why Prefill and Decode naturally favor different batching, parallelism, memory policies and even attention kernels.**

## 18. Memory note

```text
PREFILL
-------
many tokens together
large GEMMs
many Q/K/V
high arithmetic intensity
compute-heavy
context FMHA / FlashAttention
TTFT


DECODE
------
~1 token / active sequence
skinny matrix shapes
one Q scans large cached K/V
stream weights + KV repeatedly
bandwidth + KV-capacity sensitive
masked MHA / XQA / paged decode
TPOT
```

## References

- Splitwise: Efficient Generative LLM Inference Using Phase Splitting.
- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.
- TensorRT-LLM attention documentation: Context FMHA, Generation Masked MHA, XQA.
- NVIDIA Dynamo disaggregated-serving and tuning documentation.
