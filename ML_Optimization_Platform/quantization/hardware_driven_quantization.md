# Hardware-Driven Quantization

> **Purpose:** Explain, purely from the hardware point of view, how GPU characteristics affect whether a quantized representation is feasible and what performance benefit it is expected to provide.

## 1. Core Principle

Hardware analysis answers two questions for every quantized representation:

```text
1. Is this precision / execution path feasible on this device?

2. If it is feasible, what hardware benefit should we expect?
```

A useful conceptual output is:

```text
HardwareEvaluation {
    feasible
    expected_compute_gain
    expected_bandwidth_gain
    expected_capacity_gain
    expected_topology_gain
}
```

## 2. Native Low-Precision Instruction Support

The most direct hardware factor is whether the GPU has native instructions for a given low-precision datatype.

Examples:

```text
BF16 / FP16 Tensor Core operations
FP8 Tensor Core operations
INT8 matrix operations
INT4 / low-bit weight execution paths
newer FP4-class operations
```

If native support exists:

```text
low-precision operands
    -> native matrix instruction
    -> higher useful arithmetic throughput
```

If it does not exist, the device may require conversion, dequantization, higher-precision execution, or another less efficient path.

The representation may still save memory even when the compute benefit is weak.

## 3. Relative Compute Throughput

Supporting a precision is not enough. Different precisions can have very different throughput.

Conceptually:

```text
BF16 throughput = baseline
FP8 throughput = potentially much higher
INT8 throughput = hardware-dependent
INT4 throughput = highly execution-path-dependent
```

This matters especially for dynamic activation quantization:

```text
BF16 activation
    -> calculate scale
    -> FP8 / INT8 activation
    -> low-precision GEMM
```

The conversion is worthwhile only when:

```text
dynamic_quantization_overhead
    <
low_precision_compute_savings
```

Hardware determines how large that compute saving can be.

## 4. HBM Bandwidth

Quantization reduces bytes moved from HBM.

Approximate storage:

```text
BF16:
    2 bytes/value

FP8 / INT8:
    1 byte/value

INT4:
    ~0.5 byte/value
```

For bandwidth-bound execution:

```text
smaller representation
    -> fewer bytes transferred
    -> lower HBM pressure
```

This is particularly important for large weights repeatedly streamed during decode.

A low-bit weight path may therefore win mainly through bandwidth rather than native low-bit arithmetic.

## 5. HBM Capacity

Quantization changes how much state can fit per GPU.

Example for 100B parameters, ignoring metadata:

```text
BF16 weights:
    ~200 GB

FP8 / INT8 weights:
    ~100 GB

INT4 weights:
    ~50 GB
```

This can change:

```text
number of GPUs needed to hold the model
room available for KV cache
maximum concurrency
feasible model-parallel topology
```

Capacity benefit can be as important as raw kernel speed.

## 6. KV-Cache Capacity Benefit

Lower-precision KV directly reduces persistent decode state.

Conceptually:

```text
BF16 KV:
    2 bytes/value

FP8 KV:
    1 byte/value
```

Benefits can include:

```text
longer context
more resident sequences
larger effective batch
```

## 7. KV-Cache Bandwidth Benefit

KV is repeatedly read during decode.

Therefore:

```text
smaller KV representation
    -> fewer bytes read by attention
    -> lower HBM pressure
```

This can improve decode attention when it is bandwidth-sensitive.

## 8. Low-Bit Weight Bandwidth Benefit

A low-bit weight representation can be useful even when arithmetic happens at a wider precision internally.

Conceptually:

```text
packed low-bit weights in HBM
    -> smaller transfer
    -> unpack / scale
    -> arithmetic
```

Important distinction:

```text
storage precision
    != arithmetic precision
```

and:

```text
memory-speed benefit
    != compute-speed benefit
```

## 9. Accumulation Precision

Low-precision inputs do not imply low-precision accumulation.

For example:

```text
FP8 x FP8
    -> wider accumulation
```

or a low-bit weight path may accumulate in:

```text
FP16
BF16
FP32
```

Hardware analysis should therefore distinguish:

```text
operand precision
from
accumulation precision
```

because this affects numerical behavior and execution efficiency.

## 10. Hardware Generation / Compute Capability

Low-precision features differ across GPU generations.

A newer generation may add:

```text
new low-precision datatypes
new matrix instructions
higher Tensor Core throughput
different scaling support
newer low-bit execution modes
```

A useful support classification is:

```text
UNSUPPORTED

SUPPORTED BUT NOT NATIVE / LESS ATTRACTIVE

NATIVE / OPTIMIZED
```

This is more useful than a simple yes/no.

## 11. Memory Hierarchy and Cache Effects

Smaller tensors may also improve:

```text
L2 cache residency
shared-memory efficiency
memory transactions
register pressure in some kernels
```

For example:

```text
smaller packed weights
    -> more useful data fits in cache
    -> less repeated HBM traffic
```

## 12. Dequantization Cost

Low-bit storage often requires:

```text
unpack
scale
zero-point adjustment
conversion
```

before or during compute.

If:

```text
dequantization overhead
    >
bandwidth / compute savings
```

the low-bit path may not help.

Strong hardware execution paths minimize this overhead through close integration with matrix multiplication.

## 13. Interconnect and Topology Benefit

Quantization can indirectly reduce cross-GPU cost.

If lower precision allows:

```text
TP8
    -> TP4
```

then there may be:

```text
fewer participating GPUs
less collective communication
less synchronization
more independent replicas
```

The value depends on the device topology and interconnect.

## 14. Hardware Benefit Is Multi-Dimensional

Do not collapse hardware benefit into one scalar too early.

Example:

```text
FP8 W8A8
    -> strong compute benefit
    -> lower weight bandwidth
    -> lower activation bandwidth

INT4 W4A16
    -> very strong weight-memory reduction
    -> very strong weight-bandwidth reduction
    -> compute benefit depends on execution path

FP8 KV
    -> strong KV-capacity benefit
    -> strong KV-bandwidth benefit
```

A useful reward vector is:

```text
ExpectedHardwareBenefit {
    compute_gain
    memory_bandwidth_gain
    memory_capacity_gain
    kv_capacity_gain
    topology_gain
}
```

## 15. Example Hardware Evaluation

Illustrative example:

```text
Candidate:
    FP8 W8A8

feasible:
    true

expected benefit:
    compute_gain = HIGH
    bandwidth_gain = HIGH
    capacity_gain = HIGH
```

Another:

```text
Candidate:
    INT4 W4A16

feasible:
    true

expected benefit:
    compute_gain = MEDIUM / path-dependent
    bandwidth_gain = VERY_HIGH
    capacity_gain = VERY_HIGH
```

Another:

```text
Candidate:
    FP8 KV

feasible:
    true

expected benefit:
    kv_capacity_gain = HIGH
    kv_bandwidth_gain = HIGH
```

## 16. Hardware Output

For each candidate representation:

```text
precision:
    INT4 W4A16

hardware_feasible:
    true

support_class:
    NATIVE_OR_OPTIMIZED

expected_hardware_benefit:
    compute_gain: MEDIUM
    memory_bandwidth_gain: VERY_HIGH
    memory_capacity_gain: VERY_HIGH
    topology_gain: POSSIBLE
```

or:

```text
precision:
    some low-precision representation

hardware_feasible:
    false

reason:
    required execution capability unavailable
```

## 17. Feasibility vs Expected Reward

Hardware analysis should distinguish:

```text
FEASIBILITY
```

from:

```text
EXPECTED REWARD
```

A representation may be feasible but unattractive. Two feasible representations can be attractive for completely different reasons.

## 18. Interview Summary

> **From the hardware point of view, quantization has two questions: can the device execute this low-precision representation, and if so, what benefit should we expect? Native low-precision instructions determine compute opportunity; HBM bandwidth determines the value of reducing bytes moved; HBM capacity determines how quantization changes model and KV residency; and interconnect can create topology benefits if lower precision allows fewer GPUs per replica. I therefore attach a multi-dimensional expected hardware benefit to every feasible precision instead of treating hardware support as only yes/no.**

## 19. Locked Principle

> **Hardware determines feasibility and expected reward for a quantized representation through compute capability, bandwidth, capacity, memory hierarchy, accumulation support, and topology effects.**
