# Prefill / Decode Tuning Knobs

**Interview question:** When setting up dedicated Prefill and Decode workers, which ServingProfile knobs would you tune differently, and why?

Do not memorize:

```text
Prefill = TP2
Decode  = TP4
```

Memorize:

> **Tune each phase around its bottleneck. Prefill wants enough compute-parallel work to minimize TTFT without unnecessary communication or queueing. Decode wants enough memory bandwidth, KV capacity and sequence concurrency to minimize TPOT while maximizing SLO-valid throughput.**

Exact values depend on:

```text
model architecture
model size
dense vs MoE
GPU generation
GPU topology
engine/backend
quantization
input/output length distribution
prefix reuse
TTFT/TPOT SLOs
```

---

# 1. Summary Mental Model

```text
PREFILL
-------
dominant pressure:
    prompt compute
    prompt length
    TTFT

typical bias:
    large efficient token work
    high Tensor-Core utilization
    avoid excessive distributed communication


DECODE
------
dominant pressure:
    weight traffic
    KV traffic/capacity
    active sequence concurrency
    TPOT

typical bias:
    aggregate HBM bandwidth
    large persistent KV arena
    efficient continuous batching
    repetitive decode-optimized execution
```

The knobs below are consequences of that difference.

---

# 2. Tensor Parallelism — TP

## What TP changes

Tensor Parallelism shards tensor operations across GPUs.

```text
TP1

GPU0
[ full tensor operation ]


TP4

GPU0 -> shard 0
GPU1 -> shard 1
GPU2 -> shard 2
GPU3 -> shard 3
          |
          v
     collective communication
```

Potential benefits:

```text
more aggregate compute
more aggregate HBM bandwidth
smaller weight shard per GPU
more aggregate memory
```

Costs:

```text
AllReduce / AllGather / ReduceScatter
synchronization
smaller local GEMMs
collective latency
```

## Prefill effect

Example, illustrative only:

```text
8K prompt

TP1:
    TTFT = 130 ms
    one GPU not enough compute

TP2:
    TTFT = 75 ms
    both GPUs have large efficient GEMMs

TP4:
    TTFT = 72 ms
    compute improves slightly,
    collectives become more visible

TP8:
    TTFT = 88 ms
    smaller local work + communication dominates
```

The important question is:

> **Has the local Prefill work already saturated the accelerator?**

Prefill already has thousands of token rows. Once a modest TP gives enough compute throughput, adding ranks can mainly:

```text
shrink local GEMM dimensions
increase communication
```

So Prefill often favors:

> **The smallest TP that fits the model and meets the desired TTFT/compute saturation point.**

## Decode effect

Decode has a different benefit from TP.

Suppose TP1:

```text
one GPU
    stores/streams a large weight footprint
    has one GPU's HBM bandwidth
    has less remaining memory for KV
```

TP4:

```text
weights distributed across 4 GPUs
+
aggregate HBM bandwidth of 4 GPUs
+
larger aggregate memory pool
+
potentially more usable KV capacity
```

That can improve:

```text
per-token latency
maximum active sequences
long-context capacity
```

even if pure compute is underutilized.

But eventually:

```text
TP4 -> TP8 -> TP16
```

causes collective latency to dominate.

### Interview rule

```text
Prefill:
    increase TP until extra compute no longer repays
    extra communication.

Decode:
    larger TP can buy bandwidth + memory/KV capacity,
    but stop when collectives dominate TPOT/throughput.
```

NVIDIA Dynamo explicitly uses smaller-P / larger-D TP as a representative disaggregated configuration, not as a universal constant.

---

## Toy Example — TP=1 vs TP=2

### Setup

```text
Weight matrix W  [4, 4]   (H_in=4, H_out=4)

W = [[ 1,  2,  3,  4],
     [ 5,  6,  7,  8],
     [ 9, 10, 11, 12],
     [13, 14, 15, 16]]

Prefill token matrix X  [2, 4]   (2 tokens, 4 hidden dims)

X = [[1, 0, 1, 0],    ← token A
     [0, 1, 0, 1]]    ← token B

Decode token vector x  [1, 4]   (1 new token)

x = [[1, 1, 0, 0]]
```

The projection to compute is:  `Y = X @ W`

---

### TP=1

One GPU holds the full weight matrix and does the entire matmul.

```text
Prefill (TP=1):

  GPU0 reads W [4, 4]  from HBM  (full matrix)
  GPU0 computes X [2, 4] @ W [4, 4]

  token A: [1,0,1,0] @ W = [1+9, 2+10, 3+11, 4+12]  = [10, 12, 14, 16]
  token B: [0,1,0,1] @ W = [5+13, 6+14, 7+15, 8+16]  = [18, 20, 22, 24]

  Y = [[10, 12, 14, 16],
       [18, 20, 22, 24]]

  Total multiply-adds:  2 tokens × 4 outputs × 4 inputs = 32
  All 32 on GPU0.


Decode (TP=1):

  GPU0 reads W [4, 4]  from HBM  (full matrix)
  GPU0 computes x [1, 4] @ W [4, 4]

  token: [1,1,0,0] @ W = [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]

  Total multiply-adds: 1 × 4 × 4 = 16
  HBM bytes read (W only): 4×4×2 = 32 bytes  (BF16)
  Arithmetic intensity: 16 FLOPs / 32 bytes = 0.5 FLOP/byte   ← very low
```

---

### TP=2  (column-parallel split on output dimension)

Split W by output columns. Each GPU owns half the output dimensions.

```text
GPU0 gets W0 = W[:, :2]   (output dims 0-1)

  W0 = [[ 1,  2],
        [ 5,  6],
        [ 9, 10],
        [13, 14]]

GPU1 gets W1 = W[:, 2:]   (output dims 2-3)

  W1 = [[ 3,  4],
        [ 7,  8],
        [11, 12],
        [15, 16]]

Both GPUs receive the full input X (replicated or AllGathered — small cost).
```

**Prefill (TP=2):**

```text
GPU0 computes X @ W0   [2,4] @ [4,2] = [2,2]
  token A: [1,0,1,0] @ W0 = [1+9,  2+10]  = [10, 12]
  token B: [0,1,0,1] @ W0 = [5+13, 6+14]  = [18, 20]
  Y0 = [[10, 12], [18, 20]]

GPU1 computes X @ W1   [2,4] @ [4,2] = [2,2]
  token A: [1,0,1,0] @ W1 = [3+11, 4+12]  = [14, 16]
  token B: [0,1,0,1] @ W1 = [7+15, 8+16]  = [22, 24]
  Y1 = [[14, 16], [22, 24]]

AllGather → Y = [[10, 12, 14, 16], [18, 20, 22, 24]]   ✓ same result

Each GPU does 16 multiply-adds  (vs 32 on one GPU at TP=1).
Both run in parallel  → wall-clock GEMM time halved.
```

**Decode (TP=2):**

```text
GPU0 reads W0 [4,2] from HBM  → 16 bytes
GPU1 reads W1 [4,2] from HBM  → 16 bytes
   (both read in parallel, total HBM bytes still 32 but split across two streams)

GPU0: [1,1,0,0] @ W0 = [1+5, 2+6]   = [6,  8]
GPU1: [1,1,0,0] @ W1 = [3+7, 4+8]   = [10, 12]

AllGather → [6, 8, 10, 12]   ✓

Each GPU reads only 16 bytes from HBM, running in parallel.
Effective HBM bandwidth doubles.
```

---

### Why TP=2 wins for Prefill vs Decode — different reasons

```text
PREFILL — TP=2 wins because of COMPUTE parallelism
───────────────────────────────────────────────────
TP=1:  32 multiply-adds on 1 GPU
TP=2:  16 multiply-adds on each GPU, running in parallel
       → wall-clock GEMM time halved

Prefill has many token rows. Each byte of W can be reused across multiple
token rows in the same GEMM tile — arithmetic intensity is already decent.
Adding a GPU not only gives more compute throughput but halves the wall-clock
GEMM time — both GPUs work on their shard simultaneously, so the user-visible
latency (TTFT) shrinks proportionally. The communication
(AllGather of small input) is cheap relative to the GEMM work.

The bottleneck TP=2 is attacking: COMPUTE.


DECODE — TP=2 wins because of BANDWIDTH aggregation
────────────────────────────────────────────────────
TP=1:  1 token row. Read full W from HBM, do 16 multiply-adds.
       Arithmetic intensity = 0.5 FLOP/byte → HBM-bandwidth bound.
       Adding more compute on the same GPU does nothing.

TP=2:  Each GPU reads half of W from its own HBM in parallel.
       Aggregate HBM bandwidth doubles.
       Both halves of the output are produced simultaneously.
       Wall-clock memory read time halved → token latency halved.

       ALSO: W0 and W1 live in separate GPU HBM.
       Memory freed per GPU can hold more KV blocks →
       more concurrent sequences, longer contexts.

The bottleneck TP=2 is attacking: HBM BANDWIDTH + HBM CAPACITY.


The key insight
───────────────
Same TP=2 configuration. Different bottleneck attacked.

Prefill:  many token rows → GEMMs are already compute-workable.
          TP adds parallel compute.

Decode:   one token row → GPU is just waiting on HBM reads.
          TP adds parallel HBM bandwidth and memory.
          Compute utilization barely changes.
          The win is purely in how fast weights can be streamed.
```

---

# 3. Pipeline Parallelism — PP

## What PP changes

Pipeline Parallelism splits Transformer layers into stages.

```text
Stage 0 / GPU group A:
    layers 0 ... 19

Stage 1 / GPU group B:
    layers 20 ... 39

Stage 2 / GPU group C:
    layers 40 ... 59

Stage 3 / GPU group D:
    layers 60 ... 79
```

Instead of a collective inside many tensor operations, PP transfers activations between stage boundaries.

### What is a pipeline bubble?

When a new batch enters the pipeline, stages cannot all be busy simultaneously at first —
each stage must wait for the previous one to finish before it has work to do.
That idle wait time is a **pipeline bubble**.

```text
4-stage pipeline, one microbatch:

Time →    t0      t1      t2      t3
Stage 0:  [work]  [idle]  [idle]  [idle]
Stage 1:  [idle]  [work]  [idle]  [idle]
Stage 2:  [idle]  [idle]  [work]  [idle]
Stage 3:  [idle]  [idle]  [idle]  [work]

Stages 0-2 sit idle after their microbatch passes through.
That wasted GPU time is the bubble.
```

The fix is to split the batch into multiple microbatches so stages overlap:

```text
4-stage pipeline, 4 microbatches (m0 m1 m2 m3):

Time →    t0    t1    t2    t3    t4    t5    t6
Stage 0:  [m0]  [m1]  [m2]  [m3]  ----  ----  ----
Stage 1:  ----  [m0]  [m1]  [m2]  [m3]  ----  ----
Stage 2:  ----  ----  [m0]  [m1]  [m2]  [m3]  ----
Stage 3:  ----  ----  ----  [m0]  [m1]  [m2]  [m3]

Stages overlap — bubble shrinks as microbatch count grows.
```

Bubble fraction ≈ (num_stages - 1) / (num_stages - 1 + num_microbatches).
More microbatches = smaller bubble, but also more end-to-end latency.

This is why Prefill (many token rows → many natural microbatches) can amortize
bubbles better than Decode (one token per sequence per step → very few microbatches).

Benefits:

```text
fit very large models
limit TP collective domain
map model across nodes
```

Costs:

```text
pipeline bubbles
stage imbalance
activation-transfer latency
more end-to-end stages
```

## Prefill effect

Long prompts provide substantial work that can be split into chunks/microbatches:

```text
chunk A -> S0 -> S1 -> S2
chunk B -> S0 -> S1 -> S2
chunk C -> S0 -> S1 -> S2
```

Enough token work can amortize pipeline bubbles.

However TTFT still includes the trip through all stages.

A deeper pipeline means:

```text
more stage boundaries
more activation transfer
more opportunity for bubbles
```

So PP is usually something we introduce because of:

```text
model fit
topology
or poor scalability of very large TP
```

not because "Prefill likes PP."

## Decode effect

Every output-token iteration traverses the full pipeline:

```text
new token
  -> stage 0
  -> network
  -> stage 1
  -> network
  -> stage 2
  -> ...
```

Therefore pipeline latency recurs every token and can directly appear in TPOT.

Large sequence batches/microbatches can improve PP utilization, but excessive PP depth can be painful for latency-sensitive Decode.

### Practical topology principle

For dense models a common starting point is:

```text
TP inside a high-bandwidth node/domain
PP across nodes
```

because TP collectives benefit strongly from NVLink/NVSwitch-class bandwidth, whereas PP sends stage activations across node boundaries.

Current Dynamo tuning guidance recommends this as a strong dense-model starting point.

### Interview rule

```text
Prefill:
    large token work can amortize PP bubbles better,
    but deep PP still increases TTFT.

Decode:
    PP stage latency happens every generated token,
    so avoid unnecessary pipeline depth.
```

---

# 4. Combining TP and PP

Suppose a model needs 8 GPUs.

## Option A

```text
TP8 PP1
```

Advantages:

```text
no pipeline bubble
model distributed broadly
large aggregate bandwidth
```

Potential problem:

```text
large collectives
especially bad if TP crosses a slow inter-node boundary
```

## Option B

```text
TP4 PP2
```

Advantages:

```text
TP collectives remain in fast 4-GPU domain
only activation handoff crosses PP boundary
```

Cost:

```text
pipeline bubble/latency
```

P and D can choose differently:

```text
Prefill:
    TP2 PP2

Decode:
    TP4 PP1
```

if the runtime's KV handoff can deal with the resulting P/D layout relationship.

This freedom is one of the real benefits of disaggregation.

---

# 5. Request Batch Size

## Prefill

Tiny Prefill batches can underfeed a large accelerator:

```text
small prompt/batch
    -> launch overhead matters
    -> insufficient GEMM size
    -> low GPU utilization
```

Adding prompt work:

```text
more prompt tokens
    -> larger GEMMs
    -> Tensor Cores saturate
```

But waiting to create a huge request batch increases queueing and TTFT.

Therefore the useful target is:

> **Smallest Prefill batch that already saturates useful GPU compute.**

Dynamo's tuning guide explicitly recommends this principle.

## Decode

Decode benefits from active-sequence batching:

```text
sequence A -> one new token
sequence B -> one new token
sequence C -> one new token
...
```

More active sequences allow better amortization of reading the weights.

But:

```text
more sequences
    -> more KV residency
    -> potentially longer iterations
    -> worse TPOT after some point
```

### Interview rule

```text
Prefill batching:
    enough token work to saturate compute,
    without unnecessary TTFT queueing.

Decode batching:
    maximize useful sequence concurrency,
    bounded by TPOT and KV capacity.
```

---

# 6. Maximum Batched Tokens / Token Budget

Request count alone is a weak measure because prompt lengths vary.

A token budget controls the amount of token work admitted into an iteration/batch.

## Prefill

Example:

```text
max batched tokens = 4K
```

might underutilize a large model/GPU.

Increasing:

```text
4K -> 16K -> 32K
```

can make GEMMs more efficient.

A dedicated P worker often tolerates a relatively large token budget because active Decode streams do not share that iteration.

But if the budget is huge:

```text
one request runs a massive work unit
    -> queued Prefill requests wait
    -> tail TTFT increases
```

So larger is not automatically better.

## Decode

Decode execution limits influence:

```text
workspace
intermediate buffers
communication buffers
scheduler structures
```

Oversized maxima can consume memory that could otherwise become persistent KV capacity.

Current Dynamo documentation specifically calls out the tension between Decode execution limits and available KV blocks.

### Interview rule

```text
P:
    large token units can improve compute efficiency.

D:
    size execution buffers conservatively so unused maxima
    do not steal memory from the KV working set.
```

---

# 7. Chunked Prefill

## Dedicated Prefill worker

Once P and D are isolated, chunked Prefill is no longer required primarily to protect Decode TPOT.

It can still help:

```text
very long contexts
fairness between Prefill requests
bounded memory/work-unit size
pipeline scheduling
```

## Decode worker / hybrid local Prefill

Chunking becomes very important if D is allowed to perform a small local Prefill.

Suppose D already owns 58K cached tokens and only 2K new tokens are uncached.

Rather than:

```text
remote P
+
KV transfer
```

D may compute those 2K locally.

If D has active generations:

```text
large chunk:
    fewer interruptions
    but larger TPOT spikes

small chunk:
    more frequent interruptions
    but shorter spikes
```

This knob becomes central in conditional/hybrid disaggregation.

---

# 8. KV Cache Memory Fraction / Arena Size

## Prefill

P needs memory to build context KV and hold it until:

```text
transfer
cache insertion
or release
```

But that state can be relatively transient compared with Decode residency.

Its main performance limiter is frequently prompt compute.

## Decode

D owns KV throughout generation:

```text
admit request
      |
      | KV stays resident and grows
      |
      | output token
      | output token
      |
      v
request completes
```

More available KV memory can directly mean:

```text
more concurrent sequences
longer contexts
fewer admission stalls
```

Decode ServingProfiles therefore often prioritize a large persistent KV arena.

### Interview rule

```text
P:
    KV is context-production/transfer state.

D:
    KV is the long-lived working set and often
    the admission/capacity bottleneck.
```

---

# 9. KV Cache Precision

Possible representations include, depending on backend/hardware:

```text
BF16 / FP16
FP8
INT8-style KV formats
```

## Prefill benefit

P writes KV.

Lower-precision KV can reduce:

```text
bytes written
P->D transfer size
temporary/cache footprint
```

## Decode benefit

D repeatedly reads KV.

A smaller KV representation can improve both:

```text
capacity
+
attention memory bandwidth
```

This is especially valuable for long-context Decode.

Example, conceptually:

```text
2-byte KV representation
        |
        v
1-byte KV representation

roughly half the bytes
```

subject to layout, scale metadata, kernel support and quality validation.

### Interview rule

> **KV quantization has a particularly strong Decode payoff because the same representation affects both how many sequences fit and how many bytes attention must read.**

P and D must of course agree on a transferable/consumable KV representation.

---

# 10. Weight / Activation Precision

Low-precision inference can help both phases for different reasons.

## Prefill

Prefill is compute-dense.

FP8/FP4-style paths, where supported, can increase Tensor-Core throughput:

```text
less expensive math / higher accelerator throughput
    -> faster prompt processing
```

## Decode

Decode repeatedly streams model weights.

Smaller quantized weights mean:

```text
fewer HBM bytes per token iteration
```

which directly targets the bandwidth bottleneck.

### Interview rule

```text
P quantization payoff:
    compute throughput

D quantization payoff:
    weight bandwidth + capacity

Both:
    subject to model-quality and kernel support
```

---

# 11. KV Block / Page Size

Paged KV systems manage cache in token blocks.

Compare:

```text
16-token block
vs
128-token block
```

## Small blocks

Advantages:

```text
fine-grained allocation
fine-grained prefix matching
less wasted tail space
```

Costs:

```text
more block metadata
more page-table work
more transfer descriptors/fragments
```

## Large blocks

Advantages:

```text
larger contiguous transfer chunks
fewer descriptors
less bookkeeping
```

Costs:

```text
coarser prefix-cache match
more internal fragmentation
```

For disaggregation this is especially important because the cache block is also a potential P->D transfer unit.

Dynamo has documented 128 tokens as a useful starting point for many dense-model deployments while explicitly stating that it is not a universal optimum.

### Phase perspective

```text
P:
    block size affects prefix granularity and transfer efficiency

D:
    block size affects allocation fragmentation and cache management
```

P and D need compatible block/layout semantics for direct transfer.

---

# 12. Prefix Caching

## Prefill

Prefix reuse can remove enormous prompt work.

Example:

```text
20K-token system prompt
shared by every request

without cache:
    compute 20K every time

with prefix cache:
    reuse cached 20K
    compute only request-specific suffix
```

That directly reduces P demand.

## Decode

After a multi-turn request generates new tokens, D may own the freshest session KV.

Keeping that state resident can make:

```text
local Prefill on D
```

or other hybrid execution preferable to sending the next turn to a remote P worker.

So prefix/session residency is not merely a Prefill optimization; it can affect whether disaggregation is worth doing.

---

# 13. Attention Kernel Selection

This is one of the clearest phase-specific knobs.

## Prefill-oriented kernels

Examples:

```text
FlashAttention
Context FMHA
Paged Context FMHA
FP8 Context FMHA
```

Work shape:

```text
many Q
many K
many V
```

Goals:

```text
large tiled operations
Tensor-Core utilization
fused softmax/attention
avoid materializing QK^T
reduce HBM traffic
```

TensorRT-LLM currently documents context-phase fused attention and FlashAttention-style behavior for longer sequences.

## Decode-oriented kernels

Examples:

```text
generation masked MHA
XQA for supported MQA/GQA cases
paged Decode attention
multi-block / split-KV strategies
FlashInfer Decode kernels
```

Work shape:

```text
q_len ~ 1
+
large cached K/V history
```

Goals:

```text
efficient paged-KV traversal
coalesced memory access
good occupancy with tiny Q dimension
parallelize a long KV scan
specialize MQA/GQA
```

TensorRT-LLM uses distinct context and generation paths and can choose XQA or masked MHA using internal heuristics.

### Interview rule

> **Kernel specialization is strong evidence that Prefill and Decode are genuinely different execution regimes, not merely two queue labels.**

---

# 14. CUDA Graphs / Repeated Execution

CUDA Graphs reduce CPU/kernel-launch overhead by capturing repeated execution.

## Prefill

Shapes vary widely:

```text
213 tokens
2K tokens
31K tokens
different ragged batches
```

Graph capture may require:

```text
bucketing
multiple graph variants
padding/shape strategies
```

and launch overhead is a smaller fraction of a large Prefill operation.

## Decode

Decode repeatedly executes:

```text
q_len ≈ 1
same model
again
again
again
```

with changing active batch size.

That repeated shape makes Decode especially attractive for:

```text
CUDA graphs
batch-size buckets
persistent/reduced-launch-overhead execution
```

### Interview rule

```text
P:
    graph optimization is useful,
    but shape variability makes it harder.

D:
    tiny repetitive steps make launch-overhead reduction
    especially valuable.
```

---

# 15. Continuous Batching / Scheduler Policy

## Prefill

A dedicated Prefill scheduler is mainly deciding:

```text
which prompts?
how many prompt tokens?
which long request goes first?
should a huge Prefill be chunked?
```

Useful goals:

```text
TTFT
GPU saturation
fairness
length-aware queueing
```

## Decode

Decode scheduler manages long-lived active sequences:

```text
A continues
B finishes
C enters
D continues
```

The batch changes continuously.

It balances:

```text
throughput
TPOT
KV admission
fairness
```

Continuous batching is therefore especially central to Decode.

---

# 16. Speculative Decoding

## Prefill

Speculative decoding does not target the main Prefill problem.

Prefill constructs the prompt context.

## Decode

Autoregressive serial generation is exactly what speculative decoding attacks:

```text
draft model:
    propose t1 t2 t3 t4

target:
    verify several in one target step

accepted:
    advance multiple tokens
```

Potential benefit:

```text
more accepted output tokens per expensive target iteration
```

Costs:

```text
draft compute
verification work
extra state
acceptance-rate dependence
```

This is primarily a Decode ServingProfile knob.

---

# 17. Hardware Shape

## Prefill-oriented priorities

A Prefill accelerator values:

```text
high dense-math / Tensor-Core throughput
fast low-precision matrix math
enough HBM for model + working state
fast TP fabric if sharded
```

The central question is:

> **How cheaply can I execute large prompt GEMMs under the TTFT target?**

## Decode-oriented priorities

D cares disproportionately about:

```text
HBM bandwidth
HBM capacity
cost per GB/s
cost per GB
power efficiency
fast TP interconnect
```

The highest peak-FLOP accelerator is not automatically the cheapest Decode device.

Splitwise's hardware characterization specifically motivates heterogeneous phase-specific machines because generation can underutilize the newest accelerator's compute capability.

---

# 18. Interconnect / Topology

Parallelism is not independent of topology.

## TP

TP performs frequent collectives.

Therefore place TP ranks in the fastest practical domain:

```text
NVLink / NVSwitch
high-bandwidth intra-node fabric
```

Cross-node TP over a slower fabric can expose too much collective latency.

## PP

PP communicates mostly at stage boundaries, making it a more natural mechanism for crossing nodes when a model needs multiple nodes.

This motivates the practical starting point:

```text
TP within node
PP across nodes
```

## P->D

Disaggregation introduces another path:

```text
Prefill GPU
     |
     | KV
     v
Decode GPU
```

Now P/D placement must consider:

```text
same node / NVLink
same rack / RDMA
cross-rack
network congestion
```

This isn't solely a worker-local knob, but it determines whether a given P/D profile pairing is worthwhile.

---

# 19. MoE Expert Parallelism — EP

For MoE models:

```text
token
  |
  v
expert router
  |
  +--> expert 3
  +--> expert 9
```

experts may be distributed across GPUs.

This creates all-to-all communication.

## Prefill

Many prompt tokens create larger expert batches:

```text
more tokens per expert
more useful compute per communication round
```

## Decode

One token per active sequence can create smaller, more latency-sensitive expert batches.

Communication may become a larger fraction of each Decode step.

Therefore the ideal combination:

```text
TP × EP × PP
```

can differ by phase.

Do not apply dense-model TP rules mechanically to MoE.

---

# 20. Communication / Kernel Fusion

With TP, a layer may resemble:

```text
GEMM
  |
  v
AllReduce
```

Systems can overlap or fuse computation and collective communication.

## Prefill

Large GEMMs give substantial computation that can overlap communication.

## Decode

Small per-step work can make collective latency a much larger fraction of TPOT.

Therefore:

```text
GEMM+AllReduce fusion
communication overlap
persistent collectives
```

can have phase-dependent importance.

---

# 21. What Is NOT a Worker-Profile Knob?

The following is related but should be kept separate:

```text
number of Prefill workers
number of Decode workers
```

That answers:

> **How much P and D capacity should the fleet provision?**

rather than:

> **How should one P or D worker execute?**

We should treat P:D fleet sizing as a later capacity-planning topic.

---

# 22. Summary Table

| Knob | Prefill Tendency | Decode Tendency | Core Reason |
|---|---|---|---|
| **TP** | Smallest TP that saturates compute / meets TTFT | Potentially larger TP for HBM bandwidth, weight sharding and KV capacity | Compute vs memory pressure |
| **PP** | Use for fit/topology; large token work can amortize bubbles | Avoid unnecessary depth; pipeline latency recurs every token | TPOT sensitivity |
| **Request batch** | Smallest batch that saturates GPU | High active-sequence concurrency within TPOT/KV limits | GEMM saturation vs weight reuse |
| **Token budget** | Relatively large work units | Controlled to preserve KV memory and iteration latency | Different working set |
| **Chunked Prefill** | Fairness/long-context tool | Important if D also performs local Prefill | Controls interference |
| **KV arena** | More transient production/transfer state | Large persistent working set | D residency |
| **KV precision** | Less write/transfer data | Less KV bandwidth + more capacity | Repeated D reads |
| **Weight precision** | Higher compute throughput | Lower weight traffic | Different bottlenecks |
| **KV block size** | Prefix/transfer granularity | Allocation/fragmentation | Cache management |
| **Prefix caching** | Saves prompt compute | Session residency can avoid remote P | Reuse |
| **Attention kernels** | Context FMHA / FlashAttention | masked MHA / XQA / paged Decode | Different Q/K/V shapes |
| **CUDA graphs** | Harder due variable shapes | Very attractive for repetitive q_len≈1 | Launch overhead |
| **Scheduler** | Prompt/token-work and TTFT oriented | Continuous sequence / TPOT / KV oriented | Different lifetimes |
| **Speculative decoding** | Little direct relevance | Potential major acceleration | Autoregressive loop |
| **Hardware** | Tensor-Core/dense compute | HBM bandwidth + capacity | Phase physics |
| **TP topology** | Keep collectives fast | Keep collectives fast, especially TPOT-sensitive | Communication |
| **EP for MoE** | Larger expert token batches | Smaller latency-sensitive expert batches | MoE all-to-all |

---

# 23. Important Interview Anti-Patterns

Do not say:

```text
"Prefill always uses TP2."
"Decode always uses TP4."
```

Say:

> **Prefill often prefers a smaller TP because its large prompt GEMMs may already saturate compute; extra ranks can mostly add collectives. Decode can justify larger TP because sharding exposes more aggregate HBM bandwidth and memory for KV, but only until communication dominates.**

Do not say:

```text
"Decode is always memory bound."
```

Say:

> **Decode generally has much lower arithmetic intensity than Prefill and is therefore much more memory-bandwidth/KV-capacity sensitive. Large Decode batches can improve compute utilization.**

Do not say:

```text
"PP is good for Prefill and bad for Decode."
```

Say:

> **PP is primarily a fit/topology knob. Prefill's large work can amortize bubbles more easily, while Decode pays the stage latency repeatedly on every token.**

---

# 24. Practical Tuning Workflow

For each:

```text
model revision
+
accelerator
+
engine/version
+
quantization
```

benchmark P and D separately.

## Prefill sweep

Vary:

```text
TP
PP
prompt length
batch size
max batched tokens
precision
attention backend
chunking
```

Measure:

```text
TTFT
Prefill tokens/sec
GPU/Tensor-Core utilization
collective fraction
queueing
cost per prompt token
```

Goal:

> **Find the cheapest/smallest profile that reaches the desired Prefill saturation point while meeting TTFT.**

## Decode sweep

Vary:

```text
TP
PP
active sequence count
context length
output length
KV precision
KV memory fraction
attention backend
CUDA graph mode
speculative decoding
```

Measure:

```text
TPOT / ITL
Decode tokens/sec
maximum SLO-valid concurrency
KV utilization
HBM bandwidth
communication fraction
cost per generated token
```

Goal:

> **Maximize SLO-valid Decode throughput while satisfying TPOT and KV-capacity constraints.**

Only afterward ask:

```text
how many P workers?
how many D workers?
```

---

# 25. Short Interview Answer

> **I would not hard-code rules such as Prefill TP2 and Decode TP4; I would tune each phase around its bottleneck. For Prefill, I want the smallest TP and prompt work unit that saturates compute and meets TTFT, because large prompt GEMMs already expose a lot of parallelism and extra TP can just add collectives. For Decode, I may use larger TP because weight sharding exposes aggregate HBM bandwidth and more memory for KV, improving TPOT and concurrency until communication dominates. PP is mainly a model-fit/topology knob: Prefill's large token work can amortize bubbles better, while Decode pays pipeline stage latency on every generated token. Prefill tends to use context-FMHA/FlashAttention-style kernels and large efficient token work units; Decode emphasizes paged/XQA-style attention, high active-sequence concurrency, a large persistent KV arena, lower-precision KV where supported, CUDA-graph-friendly repeated execution and possibly speculative decoding. Exact values come from profiling the model, hardware, runtime and request distribution against TTFT and TPOT SLOs.**

---

# 26. Memory Note

```text
PREFILL tuning objective
------------------------
smallest efficient profile
that saturates compute
and meets TTFT

Typical bias:
    smaller sufficient TP
    PP only as needed
    small request batch
    large token work unit
    context FMHA / FlashAttention
    compute-oriented hardware


DECODE tuning objective
-----------------------
maximize SLO-valid concurrent generation
while meeting TPOT

Typical bias:
    potentially larger TP
    avoid unnecessary PP depth
    large active sequence batch
    large persistent KV arena
    low-precision KV where supported
    paged Decode / XQA
    CUDA graphs
    speculative decoding
    bandwidth/capacity-oriented hardware
```

## References

Primary sources behind this note:

- NVIDIA Dynamo — Disaggregation and Performance Tuning.
- NVIDIA Dynamo — Disaggregated Serving.
- NVIDIA Dynamo — Advanced Performance Tuning.
- TensorRT-LLM — Context FMHA, generation masked MHA, XQA and KV-cache documentation.
- Splitwise: Efficient Generative LLM Inference Using Phase Splitting.
- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.
