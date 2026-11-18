# Final Quantization Decision Process

> **Purpose:** Describe how model architecture, runtime compatibility, and hardware analysis progressively construct, narrow, and annotate the quantization search space before empirical validation.

---

# 1. Core Idea

Quantization selection is not just choosing an independent weight precision, activation precision, and KV precision.

The useful unit is a **concrete quantization configuration**:

```text
QuantizationConfiguration {
    operator_scope

    weight_precision
    activation_precision
    kv_precision

    representation_details
    runtime

    hardware_evaluations
}
```

The overall process is:

```text
Model Architecture
        |
        | What surfaces exist?
        | What precisions are acceptable per operator?
        v
Architecture Precision Envelopes
        |
        | Form theoretical W/A/KV combinations
        v
Theoretical Quantization Combinations
        |
        | Which combinations can a runtime
        | actually represent and execute?
        v
Runtime-Compatible Quantization Configurations
        |
        | For each configuration + hardware:
        | feasible?
        | expected reward?
        v
Hardware-Annotated Quantization Space
        |
        v
Experiment / Quality Validation
```

The important refinement is:

> **Hardware evaluates complete runtime-compatible quantization configurations, not merely INT4, FP8, or KV precision as isolated independent choices.**

---

# 2. Stage 1 — Architecture Analysis

Architecture analysis answers:

```text
Which quantization surfaces exist?
    weights
    activations
    KV cache

Which operators are strong low-precision candidates?

Which operators should remain at higher precision?

What mixed-precision structure is plausible?
```

It produces an **allowed precision envelope** for each operator/state.

Example:

```text
operator:
    Dense Linear

weight_precision_possible:
    [BF16, FP16, FP8, INT8, INT4]

activation_precision_possible:
    [BF16, FP16, FP8, INT8]
```

For persistent KV:

```text
state:
    KV Cache

kv_precision_possible:
    [BF16, FP16, FP8]
```

For a numerically sensitive operation:

```text
operator:
    Softmax

activation_precision_possible:
    [FP32, BF16, FP16]
```

Architecture therefore gives us possible precision sets, but not yet executable combinations.

---

# 3. Stage 2 — Generate Theoretical Quantization Combinations

Suppose architecture produces:

```text
Weight Precision:
    [INT4, INT8, FP8]

Activation Precision:
    [BF16, INT8, FP8]

KV Precision:
    [BF16, FP8]
```

A conceptual starting point is the Cartesian product:

```text
Weight
    x
Activation
    x
KV
```

which gives:

```text
3 x 3 x 2 = 18
```

theoretical combinations.

Examples:

```text
INT4 W + BF16 A + BF16 KV
INT4 W + BF16 A + FP8 KV
INT4 W + INT8 A + BF16 KV
INT4 W + FP8 A + FP8 KV
INT8 W + INT8 A + FP8 KV
FP8 W  + FP8 A  + FP8 KV
...
```

At this point these are only **numerically/structurally admissible hypotheses**.

They are not necessarily executable.

---

# 4. Stage 3 — Runtime Compatibility Filters the Combination Space

Runtime compatibility has two parts:

```text
Representation Compatibility
+
Execution Compatibility
```

The runtime must understand the resulting:

```text
bit width / dtype
scale layout
group size
zero-point semantics
packing format
module metadata
```

and must also have an execution path for the corresponding operator combination.

This matters because support is often **joint**.

For example, a runtime may support:

```text
BF16 x BF16

FP8 weight x FP8 activation

INT4 weight x BF16 activation
```

but not:

```text
INT4 weight x FP8 activation

FP8 weight x INT8 activation

INT8 weight x BF16 activation
```

So it is insufficient to say independently:

```text
INT4 supported
FP8 supported
INT8 supported
```

The useful runtime capability is closer to:

```text
SupportedQuantizationTuple {
    operator_type
    weight_representation
    activation_representation
    quantization_layout
}
```

KV support is then combined subject to the selected attention backend/runtime support.

---

# 5. Runtime-Compatible Quantization Configurations

Suppose the runtime supports:

```text
Dense Linear:
    BF16 x BF16
    FP8 x FP8
    INT4 x BF16

KV:
    BF16
    FP8
```

Then the surviving combinations might be:

```text
Q1:
    W = BF16
    A = BF16
    KV = BF16

Q2:
    W = BF16
    A = BF16
    KV = FP8

Q3:
    W = FP8
    A = FP8
    KV = BF16

Q4:
    W = FP8
    A = FP8
    KV = FP8

Q5:
    W = INT4
    A = BF16
    KV = BF16

Q6:
    W = INT4
    A = BF16
    KV = FP8
```

These are now concrete runtime-compatible quantization configurations.

The invariant is more precisely:

```text
runtime_supported_configurations
    ⊆
architecture_allowed_combinations
```

---

# 6. Quantization Technique Is Not the Primary Runtime Dimension

Offline techniques such as:

```text
AWQ
GPTQ
```

describe how a low-bit weight representation was produced.

At runtime, the important question is:

```text
Can this runtime interpret the resulting representation?

and

Does it have a kernel/backend that can execute it?
```

So two artifacts produced by different offline techniques may map to the same runtime-level class if they produce compatible:

```text
packed weights
scales
group metadata
zero-point semantics
layout
```

Conversely, two INT4 artifacts are not automatically interchangeable if their representations differ.

---

# 7. Stage 4 — Hardware Evaluates Each Surviving Configuration

Hardware analysis receives a **runtime-compatible configuration**, for example:

```text
Q6:

Weight:
    INT4

Activation:
    BF16

KV:
    FP8

Runtime:
    Runtime-A
```

For each hardware type, it adds:

```text
hardware_feasible

expected_hardware_benefit
```

For example:

```text
H100:

hardware_feasible:
    true

expected_hardware_benefit:
    weight_compute_gain: MEDIUM
    weight_bandwidth_gain: VERY_HIGH
    weight_capacity_gain: VERY_HIGH

    kv_bandwidth_gain: HIGH
    kv_capacity_gain: HIGH
```

Another configuration:

```text
Q4:

Weight:
    FP8

Activation:
    FP8

KV:
    FP8
```

might receive:

```text
H100:

hardware_feasible:
    true

expected_hardware_benefit:
    gemm_compute_gain: VERY_HIGH
    weight_bandwidth_gain: HIGH
    activation_bandwidth_gain: HIGH
    weight_capacity_gain: HIGH

    kv_bandwidth_gain: HIGH
    kv_capacity_gain: HIGH
```

The point is that hardware compares:

```text
W4A16 + FP8 KV
```

against:

```text
FP8 W8A8 + FP8 KV
```

rather than simply comparing:

```text
INT4 vs FP8
```

---

# 8. Why Hardware Reward Must Be Configuration-Level

Some benefits can be estimated per component:

```text
INT4 weights
    -> strong weight-memory reduction

FP8 KV
    -> strong KV-memory reduction
```

But some benefits are inherently joint.

For example:

```text
FP8 weight + FP8 activation
```

may unlock a native FP8 GEMM path.

Whereas:

```text
FP8 weight + BF16 activation
```

may use a different path entirely.

Therefore:

```text
HardwareReward(W, A)
    !=
HardwareReward(W)
+
HardwareReward(A)
```

in general.

Component-level priors are useful, but final feasibility/reward should be attached to the complete executable configuration.

---

# 9. Hardware Output Is a Reward Vector

Do not collapse expected hardware benefit into one scalar too early.

For each configuration:

```text
ExpectedHardwareBenefit {
    compute_gain
    weight_bandwidth_gain
    activation_bandwidth_gain
    kv_bandwidth_gain

    weight_capacity_gain
    kv_capacity_gain

    topology_gain
}
```

Different configurations can be attractive for different reasons.

Example:

```text
FP8 W8A8
    -> especially attractive for compute-heavy GEMMs

INT4 W4A16
    -> especially attractive for weight bandwidth/capacity

FP8 KV
    -> especially attractive for KV capacity/bandwidth
```

---

# 10. Hardware Does Not Expand the Quantization Space

Hardware only evaluates configurations that survived architecture and runtime constraints.

```text
hardware_evaluated_configurations
    ⊆
runtime_supported_configurations
```

If hardware analysis reveals that another precision would be especially attractive, that is a **new candidate hypothesis**.

It must go back through:

```text
architecture / numerical validation
    ->
runtime compatibility
    ->
hardware evaluation
```

Hardware does not directly introduce it.

---

# 11. Result of Quantization Analysis

At the end of the full quantization-analysis phase, the output should look conceptually like:

```text
QuantizationSpace {

    Q1:
        operator_precision_map:
            Dense Linear:
                W = BF16
                A = BF16

            KV:
                BF16

            Norm / Softmax:
                BF16

        runtime:
            Runtime-A

        hardware_evaluations:
            H100:
                feasible = true
                expected_reward = ...

            H200:
                feasible = true
                expected_reward = ...


    Q2:
        operator_precision_map:
            Dense Linear:
                W = FP8
                A = FP8

            KV:
                FP8

            Norm / Softmax:
                BF16

        runtime:
            Runtime-A

        hardware_evaluations:
            H100:
                feasible = true
                expected_reward = ...

            H200:
                feasible = true
                expected_reward = ...


    Q3:
        operator_precision_map:
            Dense Linear:
                W = INT4
                A = BF16

            KV:
                FP8

            Norm / Softmax:
                BF16

        runtime:
            Runtime-A

        hardware_evaluations:
            H100:
                feasible = true
                expected_reward = ...

            H200:
                feasible = true
                expected_reward = ...
}
```

This is the important handoff to later execution-space analysis.

The downstream unit is no longer merely:

```text
hardware = H100
```

It is closer to:

```text
QuantizationConfiguration
+
Runtime
+
Hardware
```

because those three together determine model memory footprint, kernel behavior, KV footprint, and eventually feasible parallelism/topology.

---

# 12. Decision Diagram

```text
+------------------------------------------------------+
|               MODEL ARCHITECTURE                     |
|                                                      |
| identify weights / activations / KV                  |
| define precision envelope per operator/state         |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|          ARCHITECTURE PRECISION SETS                 |
|                                                      |
| W -> [BF16, FP8, INT8, INT4]                         |
| A -> [BF16, FP8, INT8]                               |
| KV -> [BF16, FP8]                                    |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|        THEORETICAL QUANTIZATION COMBINATIONS         |
|                                                      |
| W x A x KV                                           |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|                    RUNTIME                           |
|                                                      |
| representation compatibility                         |
| execution compatibility                              |
| operator/backend compatibility                       |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|    RUNTIME-COMPATIBLE QUANTIZATION CONFIGURATIONS    |
|                                                      |
| Q1 = BF16/BF16 + BF16 KV                             |
| Q2 = FP8/FP8   + FP8 KV                              |
| Q3 = INT4/BF16 + FP8 KV                              |
| ...                                                  |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|                    HARDWARE                          |
|                                                      |
| For each Qx + hardware pair:                         |
|   feasible?                                          |
|   compute reward?                                    |
|   bandwidth reward?                                  |
|   capacity reward?                                   |
|   topology implications?                             |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|             QUANTIZATION SPACE                       |
|                                                      |
| Concrete runtime-compatible configurations           |
| + per-hardware feasibility/reward                    |
+--------------------------+---------------------------+
                           |
                           v
+------------------------------------------------------+
|              EMPIRICAL VALIDATION                    |
|                                                      |
| actual quality / latency / throughput / cost         |
+------------------------------------------------------+
```

---

# 13. Worked Example — 100B MoE Decoder

The following is an **illustrative reasoning example**, not a claim about any exact runtime version's current support matrix.

Assume:

```text
100B total parameters
decoder-only autoregressive Transformer
GQA attention
RMSNorm
SwiGLU
Mixture-of-Experts FFN blocks
Top-2 expert routing
128K context
BF16 canonical checkpoint
```

---

# 14. Architecture Analysis — Quantization Surfaces

From the model structure:

```text
Q/K/V/O projections
    -> weight quantization surface
    -> activation quantization surface

Expert up/gate/down projections
    -> weight quantization surface
    -> activation quantization surface

Autoregressive GQA attention
    -> persistent KV state
    -> KV-cache quantization surface

RMSNorm / Softmax / reductions
    -> higher-precision regions
```

Architecture output:

```text
Dense Attention Weights:
    [BF16, FP8, INT8, INT4]

Dense Attention Activations:
    [BF16, FP8, INT8]

MoE Expert Weights:
    [BF16, FP8, INT8, INT4]

MoE Expert Activations:
    [BF16, FP8, INT8]

KV:
    [BF16, FP8]

RMSNorm / Softmax:
    [BF16, FP16]
```

---

# 15. Runtime Compatibility — Hypothetical Runtime A

Assume Runtime A exposes:

```text
Dense Linear:
    BF16 x BF16
    FP8 x FP8
    INT4 x BF16

MoE Expert Linear:
    BF16 x BF16
    FP8 x FP8

KV:
    BF16
    FP8

RMSNorm / Softmax:
    BF16 / FP16
```

Notice that:

```text
INT4 experts
```

were architecturally possible, but do not survive this runtime.

---

# 16. Build Runtime-Compatible Model Configurations

The runtime-compatible mixed-precision configurations could include:

```text
Q1 — Baseline

Attention:
    BF16 x BF16

MoE:
    BF16 x BF16

KV:
    BF16

Norm / Softmax:
    BF16
```

```text
Q2 — FP8 Compute + FP8 KV

Attention:
    FP8 x FP8

MoE:
    FP8 x FP8

KV:
    FP8

Norm / Softmax:
    BF16
```

```text
Q3 — INT4 Attention Weights + FP8 Experts + FP8 KV

Attention:
    INT4 x BF16

MoE:
    FP8 x FP8

KV:
    FP8

Norm / Softmax:
    BF16
```

```text
Q4 — FP8 Compute + BF16 KV

Attention:
    FP8 x FP8

MoE:
    FP8 x FP8

KV:
    BF16

Norm / Softmax:
    BF16
```

These are now executable **model-level quantization configurations**, not independent precision lists.

---

# 17. Hardware Evaluation — Configuration Q2

For an H100-class target:

```text
Q2:

Attention:
    FP8 x FP8

MoE:
    FP8 x FP8

KV:
    FP8
```

Illustrative hardware annotation:

```text
hardware_feasible:
    true

expected_hardware_benefit:
    gemm_compute_gain:
        HIGH

    weight_bandwidth_gain:
        HIGH

    activation_bandwidth_gain:
        HIGH

    weight_capacity_gain:
        HIGH

    kv_bandwidth_gain:
        HIGH

    kv_capacity_gain:
        HIGH
```

---

# 18. Hardware Evaluation — Configuration Q3

```text
Q3:

Attention:
    INT4 x BF16

MoE:
    FP8 x FP8

KV:
    FP8
```

Illustrative hardware annotation:

```text
hardware_feasible:
    true

expected_hardware_benefit:
    attention_weight_bandwidth_gain:
        VERY_HIGH

    attention_weight_capacity_gain:
        VERY_HIGH

    attention_compute_gain:
        MEDIUM / path-dependent

    expert_compute_gain:
        HIGH

    kv_bandwidth_gain:
        HIGH

    kv_capacity_gain:
        HIGH
```

This shows why hardware should evaluate complete configurations.

Q2 and Q3 achieve their expected benefit through different mechanisms.

---

# 19. Hardware Evaluation Across Multiple Devices

The same configuration may receive different annotations:

```text
Q3:

H100:
    feasible = true
    reward = ...

H200:
    feasible = true
    reward = ...

Other GPU:
    feasible = false
```

So the quantization-space output contains:

```text
QuantizationConfiguration
    x
Runtime
    x
HardwareEvaluation
```

not merely a global statement that INT4 or FP8 is supported.

---

# 20. Handoff to Execution-Space Analysis

Once Quantization Analysis finishes, downstream execution analysis receives configurations such as:

```text
(Q2, Runtime-A, H100)

(Q2, Runtime-A, H200)

(Q3, Runtime-A, H100)

(Q3, Runtime-A, H200)
```

Each tuple already carries:

```text
model weight precision/layout
activation execution precision
KV precision
runtime execution path
hardware feasibility
expected hardware reward
```

Now execution analysis can ask:

```text
How much memory does this representation consume?

What minimum sharding is necessary?

Which TP / PP / CP / EP topologies are plausible?

Which topologies pass detailed memory feasibility?
```

This is why quantization analysis naturally feeds both:

```text
Quantization Space
```

and the initial information needed to construct:

```text
Execution Space
```

---

# 21. Empirical Validation Still Decides the Winner

Even a configuration with excellent hardware reward may:

```text
hurt model quality

have unexpected dequantization overhead

miss TTFT / TPOT requirements

underperform under the real traffic distribution
```

Therefore:

```text
Expected Hardware Reward
    !=
Final Quantization Decision
```

Actual experiment evidence determines the qualified configuration.

---

# 22. Final Decision Principle

```text
Architecture
    -> defines precision envelopes

Combination Generation
    -> creates theoretical W/A/KV candidates

Runtime
    -> removes combinations it cannot represent/execute

Hardware
    -> evaluates every surviving configuration
       for feasibility + expected reward

Experiment
    -> measures actual quality and serving performance
```

---

# 23. Locked Invariants

```text
runtime_supported_configurations
    ⊆
architecture_allowed_combinations
```

```text
hardware_evaluated_configurations
    ⊆
runtime_supported_configurations
```

```text
HardwareReward(W, A, KV)
    is not generally equal to
HardwareReward(W)
+
HardwareReward(A)
+
HardwareReward(KV)
```

```text
Expected Hardware Reward
    !=
Final Quantization Decision
```
# 24. Final Output:

QuantizationSpace {

    Q1 {
        operator_precision_map {

            attention_linear {
                weight_precision     = BF16
                activation_precision = BF16
            }

            moe_expert_linear {
                weight_precision     = BF16
                activation_precision = BF16
            }

            rms_norm {
                activation_precision = BF16
            }

            softmax {
                activation_precision = BF16
            }

            kv_cache {
                precision = BF16
            }
        }

        runtime {
            engine = vLLM
            execution_paths {
                attention = BF16_GEMM
                moe       = BF16_MOE
                kv        = BF16_ATTENTION
            }
        }

        hardware_evaluations {

            H100 {
                feasible = true

                expected_reward {
                    compute_gain          = BASELINE
                    weight_bandwidth_gain = BASELINE
                    kv_bandwidth_gain     = BASELINE
                    weight_capacity_gain  = BASELINE
                    kv_capacity_gain      = BASELINE
                }
            }

            H200 {
                feasible = true

                expected_reward {
                    compute_gain          = BASELINE
                    weight_bandwidth_gain = BASELINE
                    kv_bandwidth_gain     = BASELINE
                    weight_capacity_gain  = BASELINE
                    kv_capacity_gain      = BASELINE
                }
            }
        }
    }


    Q2 {
        operator_precision_map {

            attention_linear {
                weight_precision     = FP8
                activation_precision = FP8
            }

            moe_expert_linear {
                weight_precision     = FP8
                activation_precision = FP8
            }

            rms_norm {
                activation_precision = BF16
            }

            softmax {
                activation_precision = BF16
            }

            kv_cache {
                precision = FP8
            }
        }

        representation {
            weight_format      = FP8
            activation_format  = FP8
            kv_format          = FP8
            scaling_scheme     = ...
        }

        runtime {
            engine = vLLM

            execution_paths {
                attention = FP8_GEMM
                moe       = FP8_MOE
                kv        = FP8_KV_ATTENTION
            }
        }

        hardware_evaluations {

            H100 {
                feasible = true

                expected_reward {
                    compute_gain             = HIGH
                    weight_bandwidth_gain    = HIGH
                    activation_bandwidth_gain= HIGH
                    kv_bandwidth_gain        = HIGH
                    weight_capacity_gain     = HIGH
                    kv_capacity_gain         = HIGH
                }
            }

            H200 {
                feasible = true

                expected_reward {
                    compute_gain             = HIGH
                    weight_bandwidth_gain    = HIGH
                    activation_bandwidth_gain= HIGH
                    kv_bandwidth_gain        = HIGH
                    weight_capacity_gain     = HIGH
                    kv_capacity_gain         = HIGH
                }
            }
        }
    }


    Q3 {
        operator_precision_map {

            attention_linear {
                weight_precision     = INT4
                activation_precision = BF16
            }

            moe_expert_linear {
                weight_precision     = FP8
                activation_precision = FP8
            }

            rms_norm {
                activation_precision = BF16
            }

            softmax {
                activation_precision = BF16
            }

            kv_cache {
                precision = FP8
            }
        }

        representation {
            attention_weight_format = INT4
            attention_group_size    = 128
            attention_scale_scheme  = ...
            moe_format              = FP8
            kv_format               = FP8
        }

        runtime {
            engine = vLLM

            execution_paths {
                attention = W4A16_GEMM
                moe       = FP8_MOE
                kv        = FP8_KV_ATTENTION
            }
        }

        hardware_evaluations {

            H100 {
                feasible = true

                expected_reward {
                    attention_compute_gain          = MEDIUM
                    attention_weight_bandwidth_gain = VERY_HIGH
                    attention_weight_capacity_gain  = VERY_HIGH

                    moe_compute_gain                = HIGH

                    kv_bandwidth_gain               = HIGH
                    kv_capacity_gain                = HIGH
                }
            }

            H200 {
                feasible = true

                expected_reward {
                    attention_compute_gain          = MEDIUM
                    attention_weight_bandwidth_gain = VERY_HIGH
                    attention_weight_capacity_gain  = VERY_HIGH

                    moe_compute_gain                = HIGH

                    kv_bandwidth_gain               = HIGH
                    kv_capacity_gain                = HIGH
                }
            }
        }
    }
}

# 25. Interview Summary

> **I first derive operator-level precision envelopes from the model architecture. Those weight, activation, and KV sets form a theoretical combination space. The runtime then filters combinations based on representation and execution compatibility; for example it may support W4A16 and FP8 W8A8 but not arbitrary INT4/FP8 pairings. Hardware analysis is then performed on each surviving quantization configuration, not on each precision independently, because compute paths and rewards can depend on combinations such as FP8 weight plus FP8 activation. The output is a quantization space consisting of concrete runtime-compatible configurations with per-hardware feasibility and expected reward. That configuration-runtime-hardware tuple becomes the input to later memory and parallelism analysis, while empirical experiments still decide final quality and performance.**
