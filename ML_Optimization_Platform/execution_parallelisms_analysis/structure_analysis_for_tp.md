# Structural Parallelism Analysis for Tensor Parallelism (TP)

> **Purpose:** Determine whether a proposed Tensor Parallel degree is structurally legal before evaluating its performance. This phase answers whether a Transformer layer can be partitioned into `TP` local operators while preserving model semantics, quantization representation, and runtime/kernel compatibility.

---

# 1. Natural-Language Interpretation of TP

Tensor Parallelism means:

> **Several GPUs cooperate on the same Transformer layer by splitting the large tensors/operators inside that layer.**

Without TP:

```text
GPU 0
  |
  v
Entire Attention
Entire MLP
```

With `TP = 4`:

```text
              One Transformer Layer

        +---------+---------+---------+---------+
        |         |         |         |         |
       GPU0      GPU1      GPU2      GPU3
        |         |         |         |
      shard0    shard1    shard2    shard3
        |         |         |         |
        +---------+---------+---------+
                      |
                 collective
                      |
                      v
                layer output
```

All TP ranks process the **same tokens and same layer**, but each rank owns and computes only a portion of the tensors.

The central structural question is:

> **Can every TP-sharded operator be decomposed into `TP` local operators without violating model semantics, tensor-shape constraints, quantization boundaries, or kernel support?**

---

# 2. Why Structural Analysis Comes Before Performance Analysis

At this stage we are not asking:

```text
Will TP=4 be fast?
```

We are asking:

```text
Can TP=4 execute this model correctly at all?
```

A candidate can fail before benchmarking because:

```text
attention heads cannot be partitioned cleanly
KV heads cannot be distributed or replicated
MLP dimensions do not shard into valid local shapes
quantization groups cross partition boundaries
kernel tile requirements fail on local tensors
runtime does not support the resulting partition
```

So Structural TP Analysis is fundamentally a **legality filter**.

---

# 3. Main Structural Checks

```text
TP Structural Analysis

1. Attention Partitionability
   - query heads distributable?
   - KV heads distributable or replicable?
   - head_dim / attention backend constraints valid?

2. Dense Tensor Partitionability
   - hidden dimensions shard cleanly?
   - intermediate dimensions shard cleanly?
   - embeddings / LM head compatible if sharded?

3. Quantization Partitionability
   - quantization groups remain intact?
   - scale / zero-point metadata shard cleanly?
   - packed representation supports local shape?

4. Runtime / Kernel Partitionability
   - local rank tensor shapes supported?
   - fused QKV kernels support this TP degree?
   - selected quantized backend supports the partition?

5. Structural Result
   - VALID
   - VALID_WITH_REPLICATION
   - VALID_WITH_PADDING
   - UNSUPPORTED
```

---

# 4. Attention Partitionability

## 4.1 Why Attention Heads Matter

Suppose:

```text
hidden_size = 8192
num_attention_heads = 64
head_dim = 128
```

because:

```text
64 × 128 = 8192
```

The Q tensor is conceptually:

```text
Q = [tokens, 64 heads, 128]
```

The natural TP partition is by attention heads.

With:

```text
TP = 4
```

we can assign:

```text
GPU0 -> heads  0-15
GPU1 -> heads 16-31
GPU2 -> heads 32-47
GPU3 -> heads 48-63
```

Each rank receives:

```text
64 / 4 = 16 heads
```

Natural interpretation:

> **Different TP ranks become responsible for different attention heads of the same token.**

Each rank can compute attention independently for its local heads.

---

# 5. Why Head Divisibility Is Necessary

Suppose:

```text
num_attention_heads = 60
TP = 8
```

Then:

```text
60 / 8 = 7.5
```

You cannot naturally assign half an attention head to a rank.

An attention head is a logical unit:

```text
Q_head
K_head
V_head
```

with a complete `head_dim`.

Therefore a common structural rule is:

```text
num_attention_heads % TP == 0
```

Natural-language meaning:

> **Can the attention-head set be split into equal groups so each TP rank owns complete heads?**

---

# 6. What Actually Gets Split in QKV Projection

Consider:

```text
X [tokens, hidden_size]

Wq [hidden_size, num_heads × head_dim]
```

The output dimension of `Wq` corresponds naturally to attention heads.

For TP=4:

```text
Wq:

+-----------+-----------+-----------+-----------+
| heads     | heads     | heads     | heads     |
| 0-15      | 16-31     | 32-47     | 48-63     |
+-----------+-----------+-----------+-----------+
     GPU0        GPU1        GPU2        GPU3
```

Each GPU sees the same `X`, but stores only one portion of `Wq`.

Then:

```text
GPU0:
    X @ Wq_0

GPU1:
    X @ Wq_1

GPU2:
    X @ Wq_2

GPU3:
    X @ Wq_3
```

This is a **column-parallel linear** pattern.

Each GPU computes different output features.

---

# 7. GQA and MQA Complicate Attention Partitioning

A simple check like:

```text
num_attention_heads % TP == 0
```

is not enough for Grouped Query Attention or Multi-Query Attention.

Example:

```text
num_query_heads = 64
num_kv_heads    = 8
```

For:

```text
TP = 8
```

we get:

```text
query_heads_per_rank = 8
kv_heads_per_rank    = 1
```

This is clean.

Now consider:

```text
TP = 16
```

Then:

```text
64 query heads / 16 = 4 query heads/rank
```

but:

```text
8 KV heads / 16 ranks
```

does not naturally divide.

So the structural analyzer must ask:

```text
Can KV heads be directly sharded?

If not:
    does the runtime support KV-head replication?
```

Possible outcomes:

```text
A. Directly shard KV heads
B. Replicate KV heads across subsets of TP ranks
C. Reject the TP degree
```

Natural-language interpretation:

> **Every TP rank needs a valid local Q/K/V structure. If there are fewer KV heads than TP ranks, the runtime must have a deliberate replication or grouped-partition mechanism.**

This is especially important for MQA:

```text
num_query_heads = 64
num_kv_heads = 1
```

One KV head cannot be naively divided across many TP ranks.

---

# 8. Dense Tensor Partitionability

Attention is only one part of the layer.

The MLP also contains large matrices.

Example:

```text
X
 |
 v
up_proj / gate_proj
hidden_size -> intermediate_size
 |
 v
activation
 |
 v
down_proj
intermediate_size -> hidden_size
```

Suppose:

```text
hidden_size       = 8192
intermediate_size = 28672
TP                = 4
```

The first projection can be split across its output dimension:

```text
28672 / 4 = 7168
```

So:

```text
GPU0 -> 7168 intermediate channels
GPU1 -> 7168
GPU2 -> 7168
GPU3 -> 7168
```

Natural interpretation:

> **The first projection fans the work out across GPUs.**

---

# 9. Column-Parallel Linear

For:

```text
Y = XW
```

with:

```text
W [K, N]
```

a column-parallel split partitions `N`.

```text
W = [W0 | W1 | W2 | W3]
```

Then:

```text
Yi = XWi
```

Each rank computes a different subset of output features.

```text
GPU0 -> Y0
GPU1 -> Y1
GPU2 -> Y2
GPU3 -> Y3
```

Natural interpretation:

> **Each GPU computes different columns of the output.**

Structural requirement:

```text
N must be compatible with TP
```

---

# 10. Row-Parallel Linear

The following projection often uses the opposite split.

Partition `K`:

```text
W =
[ W0 ]
[ W1 ]
[ W2 ]
[ W3 ]
```

and partition `X` correspondingly:

```text
X = [X0 | X1 | X2 | X3]
```

Each rank computes:

```text
Yi = Xi Wi
```

These are partial contributions to the same final output.

Therefore:

```text
Y = Y0 + Y1 + Y2 + Y3
```

This requires a collective such as an all-reduce.

Natural interpretation:

> **Each GPU computes part of the answer, then all partial answers must be summed.**

Structural requirement:

```text
K must be compatible with TP
```

---

# 11. Why Hidden/Intermediate Dimension Divisibility Matters

Suppose:

```text
intermediate_size = 28672
TP = 4
```

Then:

```text
28672 % 4 == 0
```

and every rank gets:

```text
7168
```

Fine.

But:

```text
intermediate_size = 28670
TP = 4
```

would lead to uneven partitions:

```text
7168
7168
7167
7167
```

While mathematically possible in principle, high-performance runtimes generally prefer regular equal partitions for:

```text
kernel shapes
collective shapes
memory allocation
communication symmetry
load balance
```

The correct rule is not:

```text
every model dimension must be divisible by TP
```

It is:

> **Every dimension actually sharded by the selected TP operator strategy must produce valid local tensor shapes.**

So the analyzer checks things such as:

```text
hidden_size % TP == 0?
intermediate_size % TP == 0?
attention projection dimensions compatible?
output projection dimensions compatible?
```

depending on how each operator is partitioned.

---

# 12. Quantization Partitionability

This is where Quantization Analysis connects directly to topology analysis.

Suppose:

```text
W [8192, 28672]

quantization:
    INT4
    group_size = 128
```

Globally, this representation is valid.

Now choose:

```text
TP = 4
```

A column-parallel partition gives:

```text
local W:
    [8192, 7168]
```

We must now ask:

> **Is this local tensor still a legal quantized representation?**

Global compatibility does not automatically imply local compatibility.

---

# 13. Quantization Group Boundaries

Suppose quantization groups are along `K`:

```text
group_size = 128
```

If TP shards the `K` dimension:

```text
K = 8192
TP = 4

local_K = 2048
```

Then:

```text
2048 % 128 == 0
```

Each rank contains complete quantization groups.

Good.

But imagine a partition produced:

```text
local_K = 2112
```

Then:

```text
2112 % 128 != 0
```

A quantization group could cross TP-rank boundaries.

Conceptually:

```text
one 128-value group
        |
   +----+----+
   |         |
 GPU0       GPU1
```

This is undesirable unless the runtime explicitly supports such a representation.

Natural-language interpretation:

> **TP must cut the quantized matrix along boundaries that preserve the atomic quantization groups.**

This may require checking:

```text
partitioned_dimension % group_size == 0
```

depending on the quantization axis.

---

# 14. Scale and Zero-Point Partitioning

Quantized tensors are not just low-bit weights.

They may include:

```text
qweight
scales
zero_points
auxiliary metadata
```

If the weight tensor is sharded, associated metadata must shard consistently.

For example:

```text
weight group 17
    |
    +--> scale 17
    +--> zero point 17
```

The analyzer must ensure the TP partition does not create ambiguous metadata ownership.

Natural-language interpretation:

> **The weight shard and the metadata needed to interpret that shard must remain colocated and structurally valid.**

---

# 15. Runtime/Kernel Partitionability

Even if the model and quantization math are valid, the local tensor shapes must still be supported by the selected kernel/backend.

Suppose globally:

```text
W [8192, 28672]
```

and:

```text
TP = 4
```

produces:

```text
local W [8192, 7168]
```

A quantized kernel may require:

```text
local_N % tile_N == 0
local_K % tile_K == 0
```

If the local shape satisfies those constraints, it remains executable.

But another TP degree may produce a shape that violates the backend's tile constraints.

Therefore the validation chain is:

```text
Global Tensor
    |
    | TP partition
    v
Local Rank Tensor
    |
    +--> model semantics valid?
    +--> tensor shape valid?
    +--> quantization groups valid?
    +--> kernel tile shape valid?
```

Natural-language interpretation:

> **A globally valid layer can become an invalid kernel problem after TP changes the local shape.**

---

# 16. Packing and Layout Constraints

From the quantization-runtime analysis, we know that kernels may expect:

```text
packed weights
specific packing direction
specific scale layout
specific aligned dimensions
specific kernel-native tiles
```

TP changes the local tensor dimensions.

So structural analysis must ensure that the partitioned representation can still be prepared for the chosen backend.

The question is not:

```text
Does the kernel support INT4?
```

It is closer to:

```text
Does the kernel support this INT4 representation
for this local TP shard shape?
```

---

# 17. TP Structural Checks as Three Layers

A useful mental model is:

```text
1. MODEL SEMANTICS

   Can attention heads / KV structure be divided correctly?


2. TENSOR SHAPE

   Can every sharded matrix produce valid local tensor shapes?


3. EXECUTION REPRESENTATION

   Do quantization groups, packing, metadata,
   runtime and kernel constraints survive the partition?
```

Only after all three pass should a TP degree remain in the candidate set.

---

# 18. Concrete Example

Assume:

```text
Model:

hidden_size       = 8192
query_heads       = 64
kv_heads          = 8
head_dim          = 128
intermediate_size = 28672

Quantization:
    W4A16
    group_size = 128

Runtime:
    vLLM
```

Evaluate:

```text
TP = 4
```

## Attention

```text
Q heads/rank:
    64 / 4 = 16

KV heads/rank:
    8 / 4 = 2
```

Clean.

## Hidden / Intermediate Dimensions

```text
hidden_size / TP:
    8192 / 4 = 2048

intermediate_size / TP:
    28672 / 4 = 7168
```

Clean.

## Quantization Groups

If a TP-sharded quantization axis produces:

```text
local K = 2048
```

then:

```text
2048 / 128 = 16 complete groups
```

Clean.

## Kernel Shape

If the selected quantized backend supports the local tile shape:

```text
VALID
```

Therefore:

```text
TP4:
    STRUCTURALLY FEASIBLE
```

---

# 19. Example with KV-Head Replication

Now evaluate:

```text
TP = 16
```

Query heads:

```text
64 / 16 = 4
```

Valid.

KV heads:

```text
8 / 16
```

not directly shardable.

The analyzer now checks:

```text
Does runtime support KV-head replication?
```

Possible result:

```text
TP16 {

    Q_head_partition:
        VALID

    KV_head_partition:
        REQUIRES_REPLICATION

    MLP_partition:
        VALID

    quantization_partition:
        VALID

    runtime_support:
        VALID_WITH_KV_REPLICATION

    structural_status:
        FEASIBLE_WITH_CONSTRAINT
}
```

This is more useful than reducing everything to one boolean.

---

# 20. Recommended Output Contract

For each candidate TP degree:

```text
TPStructuralAnalysis {

    tp_degree

    attention_partition {

        query_head_partition:
            VALID | INVALID

        query_heads_per_rank:
            ...

        kv_head_partition:
            DIRECT
            | REPLICATED
            | UNSUPPORTED

        kv_heads_per_rank:
            ...

        head_dim_compatible:
            true | false
    }


    tensor_partition {

        hidden_size_local:
            ...

        intermediate_size_local:
            ...

        embedding_partition:
            ...

        lm_head_partition:
            ...

        uniform_partition:
            true | false
    }


    quantization_partition {

        quantization_axis:
            ...

        group_size_candidates:
            [...]

        groups_preserved:
            true | false

        scale_metadata_partition:
            VALID | INVALID

        zero_point_partition:
            VALID | NOT_APPLICABLE | INVALID
    }


    runtime_kernel_partition {

        local_tensor_shapes_supported:
            true | false

        fused_qkv_supported:
            true | false

        quantized_backend_supported:
            true | false

        padding_required:
            true | false
    }


    result {

        status:
            VALID
            | VALID_WITH_REPLICATION
            | VALID_WITH_PADDING
            | UNSUPPORTED

        constraints:
            [...]

        rejection_reasons:
            [...]
    }
}
```

---

# 21. What This Phase Does Not Decide

Structural TP Analysis should not yet rank candidates by performance.

It does **not** answer:

```text
Which TP degree has the best TTFT?

Which TP degree minimizes cost/token?

How much all-reduce latency will TP=8 add?

Will TP=4 outperform TP=2?
```

Those belong to later communication/performance/search analysis.

This phase only answers:

```text
Which TP degrees are legal enough to remain in the topology candidate set?
```

---

# 22. Interview Summary

> **Tensor Parallelism splits the large tensors inside one Transformer layer across GPUs. Structural TP analysis therefore checks whether all logical units and tensor dimensions survive that split. Attention heads must be distributable, GQA/MQA KV heads must either shard or be deliberately replicated, MLP dimensions must produce valid local tensors, quantization groups and metadata must remain coherent, and the resulting local shapes must be executable by the selected runtime and quantized kernel. The goal is not to decide whether TP is fast yet, but to eliminate TP degrees that are mathematically or operationally invalid before benchmarking them.**

---

# 23. Locked Mental Model

Remember TP as:

```text
ONE LAYER
    |
    v
CUT ITS LARGE OPERATORS ACROSS GPUS
```

Then every structural question becomes:

```text
Does this cut preserve:

1. model semantics?
2. tensor shape validity?
3. quantization representation?
4. runtime/kernel executability?
```

If yes:

```text
keep the TP candidate
```

If no:

```text
reject it before performance analysis
```
