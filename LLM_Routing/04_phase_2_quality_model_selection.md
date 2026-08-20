# Component — Phase 2 QualityModelSelector

**Status:** Canonical V5 component contract  
**Scope:** Given an immutable model-independent logical request context and the set of qualified model revisions, choose an ordered set of model revisions that are acceptable from a quality perspective.

---

# 1. Why Phase 2 Exists

After request normalization and context initialization, the platform understands the **logical request** independent of provider wire format and independent of any target model.

At this point the question is:

> **Which model revision should be trusted to serve this request?**

A serving platform may expose many models:

```text
small general model
large general model
reasoning model
coding-specialized model
multilingual model
vision-language model
embedding model
future specialized models
```

There are two trivial policies:

```text
always choose the strongest model
```

which preserves quality but wastes cost and scarce capacity, or:

```text
always choose the cheapest model
```

which reduces cost but can create unacceptable quality regressions.

Phase 2 exists to solve the constrained decision:

> **Choose the least expensive eligible model revision that is predicted to satisfy the request's required quality bar.**

That is fundamentally different from:

> choose the globally best model.

### Design Principle 1

> **Model selection is a constrained quality problem, not a “pick the biggest model” problem.**

---

# 2. V5 Position in the Request Path

V5 supports both stateless and session-driven requests.

Before Phase 2:

```text
WireRequest
    |
    v
RequestNormalizer
    |
    v
NormalizedRequest
    |
    v
ContextInitializer
    |
    +--> SessionStore for SESSION requests
    |
    v
LogicalContextSnapshot
```

Phase 2 consumes that immutable **logical** context.

It deliberately runs **before** exact target-model tokenization:

```text
LogicalContextSnapshot
        |
        v
QualityModelSelector
        |
        v
ModelRoutePlan
        |
        v
TargetModelContextMaterializer
        |
        +--> exact chat template
        +--> exact tokenizer
        +--> exact token IDs
        +--> exact prefix block keys
```

This is necessary because exact tokenization depends on the model revision that Phase 2 selects.

### Design Principle 2

> **Quality selection operates on model-independent semantics; exact serving representation is created only after a model revision is known.**

---

# 3. Responsibility

`QualityModelSelector` owns:

```text
explicit model-pin handling
deterministic model capability filtering
request-model quality prediction
quality-threshold policy
model-level cost policy
uncertainty / conservative escalation policy
construction of quality-approved ordered fallback model phases
```

It does **not** own:

```text
session storage
session-to-replica stickiness
target-model tokenization
ServingProfile selection
hardware selection
engine selection
replica discovery
replica load
RouterOutstanding state
KV residency
cross-replica KV transfer
TTFT/E2E replica prediction
replica-level execution cost
```

Those belong to downstream components.

---

# 4. Phase 2 Chooses a Model Revision, Not a Deployment or Replica

This distinction is critical in V5.

Phase 2 may choose:

```text
Llama-3B revision v7
```

Later Phase 3 may discover:

```text
D1 = H100 + vLLM + FP8
D2 = H100 + TensorRT-LLM + FP8
D3 = A100 + vLLM + BF16
```

all serving that same model revision.

Phase 2 decides whether `Llama-3B-v7` is good enough for the request.

It does **not** decide whether the request should execute on:

```text
D1/R3
D2/R7
D3/R9
```

That is the Planner + `SLOCostAwareReplicaScheduler` responsibility.

### Design Principle 3

> **Phase 2 answers WHICH model revision is acceptable. Phase 3 answers WHERE and HOW that model should execute.**

---

# 5. Capability Filtering Comes Before Quality Prediction

Before predicting whether a model will answer well, the system must first determine whether the model can serve the request at all.

Example model pool:

```text
General-8B
General-70B
Code-32B
Vision-70B
Embedding-BGE
```

For:

```text
operation = EMBEDDING
```

there is no reason to ask the quality predictor whether `General-70B` is likely to succeed.

Likewise, a text-only model should never enter the quality competition for an image request.

The first stage is deterministic:

```text
Logical Request
      |
      v
Capability Filter
      |
      v
Eligible Model Revisions
```

Typical hard filters include:

```text
operation compatibility
modality support
tool semantics
structured-output support
policy/compliance eligibility
required language/domain contract when deterministic
qualified production status
context-window feasibility when it can be established model-independently
```

These are model contracts, not predictions.

### Design Principle 4

> **Never ask a learned quality model to infer deterministic capability constraints. Filter invalid model revisions first.**

This makes:

```text
quality predictor
```

simpler, more interpretable, and easier to calibrate.

---

# 6. Explicit Model Pins Bypass Automatic Quality Routing

A client/product may explicitly request:

```text
model = "general-70b-v12"
```

In that case the model identity is part of the API/product contract.

The router should:

```text
validate model exists
validate capability/policy
produce a pinned ModelRoutePlan
```

but should not silently substitute a cheaper model.

Conceptually:

```text
QualitySelectionRequest
        |
        v
Explicit model?
   /          \
 yes           no
  |             |
  v             v
Validate      Automatic
Pinned        Quality Routing
Model
```

Automatic quality selection applies only when policy permits:

```text
model = auto
internal routing endpoint
application-level routing policy
service quality tier
```

### Design Principle 5

> **Automatic model routing must preserve explicit model-selection semantics.**

---

# 7. Quality Requirement Is Policy, Not a Property of the Predictor

Suppose the quality router predicts:

```text
Model A -> P(success) = 0.91
Model B -> P(success) = 0.97
```

Which model should we use?

There is no answer until the platform knows the required quality.

For:

```text
required_quality_probability = 0.90
```

Model A may be sufficient.

For:

```text
required_quality_probability = 0.95
```

Model B may be required.

The required quality can come from:

```text
product policy
application workload class
tenant tier
endpoint configuration
request override
risk class
```

Therefore:

```text
Quality Predictor
    -> estimates capability

Quality Policy
    -> defines how much capability is required
```

### Design Principle 6

> **The predictor estimates whether a model can succeed; policy determines how much confidence is required.**

This allows one quality model to support many products without retraining it for every policy threshold.

---

# 8. Predict Request × Model Success, Not Abstract Request Difficulty

A tempting design is:

```text
Request
   |
   v
Difficulty Classifier
   |
   +-- EASY   -> small model
   +-- MEDIUM -> medium model
   +-- HARD   -> large model
```

That abstraction is too weak.

Consider:

```text
Request: implement an LRU cache
```

Possible outcomes:

```text
Code-specialized model -> very strong
small generic model    -> maybe acceptable
large generic model    -> strong
```

A multilingual or tool-heavy request may create a different ordering.

The right question is:

> **Will this specific model revision satisfy the quality criterion for this specific request?**

The core learned target is:

\[
P(\text{acceptable quality} \mid \text{request}, \text{model revision})
\]

### Design Principle 7

> **Quality is a property of the request-model pair, not of the request alone.**

---

# 9. Quality Predictor Feature Groups

The conceptual predictor is:

\[
\hat Q(x,m) = f(R(x), M(m), I(x,m))
\]

where:

```text
R(x) = request representation
M(m) = model representation
I(x,m) = request × model interaction features
```

## 9.1 Request representation

The dominant signal should be a lightweight semantic representation:

```text
LogicalContextSnapshot
      |
      v
Router Encoder
      |
      v
Request Embedding
```

It may capture latent properties such as:

```text
coding
math/reasoning
summarization
translation
domain
language
instruction complexity
conversation semantics
tool-use intent
ambiguity
knowledge intensity
```

This is preferable to maintaining a giant handcrafted rule taxonomy.

### Small structural features

Some cheap explicit features remain useful:

```text
operation
approximate logical context size
conversation depth
requested output bound
modality summary
tools present
structured-output requirement
attachment count/type
```

The goal is not to collect every possible field. Add only signals that materially affect model success.

---

## 9.2 Model representation

Model identity must be a concrete revision:

```text
model_id
model_revision
```

not merely:

```text
family
parameter count
vendor label
```

because quality can differ materially due to:

```text
pretraining
post-training
instruction tuning
preference optimization
reasoning training
tool training
revision changes
```

Useful model representations may include:

```text
learned model-revision embedding
empirical quality profile
static capability metadata
qualification status
```

---

## 9.3 Request × model interaction

The key prediction problem is compatibility.

Useful interaction features can include:

```text
request language × model language quality profile
request domain × model empirical quality
operation × model reliability
structured-output requirement × model structured-output reliability
long-context pressure × model long-context quality profile
tool requirement × tool-use quality
```

### Design Principle 8

> **A generic quality router should learn request-model compatibility, not encode model preference as static rules.**

---

# 10. Cost Belongs in the Phase-2 Decision Policy, Not in the Quality Predictor

The quality predictor must preserve a stable semantic meaning:

```text
P(acceptable quality | request, model revision)
```

It should not become:

```text
quality - cost - latency - queue depth
```

because those signals change independently.

For example:

```text
GPU pricing changes
provider pricing changes
hardware mix changes
batching efficiency changes
replica load changes every millisecond
```

None of these changes the model's semantic ability to answer the request.

Therefore:

```text
Request + Model
      |
      v
Quality Predictor
      |
      v
P(success)
      |
      +----------------+
                       |
Model-level cost prior |
Quality requirement    |
Product policy         |
                       v
                Decision Policy
```

### Important V5 distinction

Phase 2 may use a **model-level cost prior** to choose between quality-valid model revisions.

Phase 3 later uses **deployment/replica/context-materialization execution cost**.

These are not the same thing.

Example:

```text
Phase 2:
    Model A quality-valid, model-level expected cost = 1x
    Model B quality-valid, model-level expected cost = 4x
    -> prefer Model A

Phase 3 for Model A:
    H100/vLLM
    H100/TRT-LLM
    A100/vLLM
    LOCAL_KV_REUSE
    TRANSFER_KV
    RECOMPUTE
```

Phase 3 chooses the least-cost SLO-feasible execution plan among the serving options for the selected model revision.

### Design Principle 9

> **Keep quality prediction independent from cost; combine quality and model-level cost only in the Phase-2 decision policy. Keep live serving cost/load in Phase 3.**

---

# 11. Session State Affects Quality Through Semantics, Not Placement

For a V5 session request:

```text
session_id = S123
new_turn = "What about the second option?"
```

`ContextInitializer` reconstructs an immutable logical context snapshot.

Phase 2 may use:

```text
conversation semantics
conversation depth
tool history
summaries
attachments
```

as part of quality prediction.

But Phase 2 must not care that:

```text
R1 currently holds 28K KV tokens
```

or:

```text
last request executed on R1
```

Those are execution-locality signals and belong to Phase 3.

### Design Principle 10

> **Conversation content may influence model quality; conversation placement must not.**

---

# 12. Quality Router Tokenizer Is Not the Target Model Tokenizer

The quality router can use its own small semantic encoder/tokenizer:

```text
LogicalContextSnapshot
      |
      v
Router Tokenizer / Encoder
      |
      v
semantic_features_ref
```

This representation answers:

```text
What is this request asking?
Which model revision is likely to answer it well?
```

It does not need to reproduce:

```text
exact target-model token IDs
exact chat-template serialization
exact KV block boundaries
exact prefill work
```

Those are generated after model selection.

### Design Principle 11

> **Quality-routing representation and serving representation are separate abstractions of the same logical request.**

---

# 13. Exact Model-Specific Validation Happens After Model Selection

Avoid exact tokenization for every candidate model unless there is a proven need.

The preferred path is:

```text
Logical Context
      |
      v
QualityModelSelector
      |
      v
ModelRoutePlan
      |
      v
TargetModelContextMaterializer per phase
      |
      +--> exact serialization
      +--> exact tokenization
      +--> exact context-length validation
```

If exact materialization reveals that a selected phase cannot serve the context:

```text
phase 0 materialization invalid
      |
      v
use next quality-approved ModelPhasePlan
```

rather than asking Phase 3 to invent a new quality decision.

### Design Principle 12

> **Avoid candidate-by-candidate exact preprocessing on the hot path unless measured evidence justifies the cost.**

---

# 14. Input — `QualitySelectionRequest`

```text
QualitySelectionRequest {
    request_id: string
    context_snapshot_id: string

    operation: Operation

    logical_context_ref: string
    logical_context_digest: string

    request_mode: STATELESS | SESSION
    session_id: string?
    base_session_version: int?

    explicit_model: string?

    modality_summary: set<Modality>
    tool_requirements: set<string>
    structured_output_requirement: string?

    approximate_context_size: int?
    conversation_depth: int?

    max_output_tokens: int?

    quality_requirement: QualityRequirement

    tenant_id: string
    policy_tags: set<string>

    semantic_features_ref: string?

    quality_policy_id: string
}
```

## Field semantics

| Field | Meaning |
|---|---|
| `request_id` | Request identity used for observability and correlation. |
| `context_snapshot_id` | Immutable logical-context identity from `ContextInitializer`. |
| `operation` | `CHAT`, `COMPLETION`, `EMBEDDING`, etc. |
| `logical_context_ref` | Reference to model-independent logical context. |
| `logical_context_digest` | Digest used for idempotency/debugging. |
| `request_mode` | Stateless vs session-driven inference. |
| `session_id` | Session identity for stateful requests; not a replica-affinity signal. |
| `base_session_version` | Durable session version represented by this request context. |
| `explicit_model` | Optional model pin. |
| `modality_summary` | Text/image/audio/video/etc. |
| `tool_requirements` | Required tool-call semantics/capabilities. |
| `structured_output_requirement` | Schema/format requirement if any. |
| `approximate_context_size` | Cheap structural input-size signal; not exact target-model token count. |
| `conversation_depth` | Structural session/chat feature. |
| `max_output_tokens` | Requested generation upper bound where applicable. |
| `quality_requirement` | Required acceptable-quality probability/class. |
| `tenant_id` | Tenant/product identity for policy. |
| `policy_tags` | Product/compliance constraints. |
| `semantic_features_ref` | Optional precomputed router-specific semantic representation. |
| `quality_policy_id` | Versioned decision policy. |

---

# 15. Candidate Model Metadata

```text
ModelCapabilityRecord {
    model_id: string
    model_revision: string

    supported_operations: set<Operation>
    supported_modalities: set<Modality>

    tool_capabilities: set<string>
    structured_output_capabilities: set<string>

    max_context_contract: int?

    qualification_state:
        QUALIFIED
        SHADOW
        DISABLED
        UNQUALIFIED

    model_cost_prior: float

    empirical_quality_profile_ref: string?
    model_representation_ref: string?

    policy_tags: set<string>

    capability_record_version: string
}
```

`model_cost_prior` is a coarse model-level decision-policy input.

It is **not** the live execution cost estimate later produced by the V5 replica scheduler.

---

# 16. Processing Pipeline

```text
QualitySelectionRequest
        |
        v
Explicit Model Pin?
   /            \
 yes             no
  |               |
  v               v
Validate       Capability Filter
Pinned             |
Model               v
              Eligible Models
                    |
                    v
             Request Representation
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
      Model A     Model B     Model C
        |           |           |
        +-----------+-----------+
                    |
                    v
          Request × Model Scorer
                    |
                    v
       Calibrated P(success | x,m)
                    |
                    v
       Quality Threshold / Uncertainty
                    |
                    v
            Quality-Valid Models
                    |
                    v
       Model-Level Cost / Product Policy
                    |
                    v
              ModelRoutePlan
```

Algorithmically:

1. Handle explicit model pin if present.
2. Deterministically filter incapable/unqualified model revisions.
3. Build/reuse the request semantic representation.
4. Score each remaining `(request, model_revision)`.
5. Calibrate/interpret quality probability.
6. Apply quality threshold.
7. Apply conservative escalation for uncertainty/OOD policy.
8. Order quality-valid models using model-level cost/product policy.
9. Preserve bounded quality-approved fallback phases.
10. Emit `ModelRoutePlan`.

---

# 17. Decision Policy

For auto routing:

\[
m^*
=
\arg\min_{m \in EligibleModels}
ModelCostPrior(m,x)
\]

subject to:

\[
\hat Q(x,m) \ge Q_{required}
\]

This means:

```text
Model A
P(success)=0.96
cost=1x
-> quality valid

Model B
P(success)=0.82
cost=0.5x
-> rejected on quality

Model C
P(success)=0.99
cost=5x
-> quality valid
```

If:

```text
Q_required = 0.95
```

then:

```text
primary  = Model A
fallback = Model C
```

The cheaper but quality-invalid Model B never enters the route plan.

---

# 18. Why Phase 2 Returns a Route Plan, Not One Model

Suppose Phase 2 returns only:

```text
Model A
```

and later:

```text
all Model-A deployments unavailable
```

Phase 3 should not suddenly make a semantic quality decision about Model B or C.

Instead Phase 2 returns:

```text
ModelRoutePlan

phase 0 -> Model A
phase 1 -> Model C
```

where every fallback has already passed the quality policy.

This keeps ownership clean:

```text
Phase 2
    decides WHICH model revisions are quality-valid

Phase 3
    decides WHERE/HOW each quality-approved model may execute
```

### Design Principle 13

> **Replica scheduling may consume quality-approved model fallbacks, but it must not invent its own model-quality policy.**

---

# 19. Output — `ModelRoutePlan`

```text
ModelRoutePlan {
    request_id: string
    context_snapshot_id: string

    phases: list<ModelPhasePlan>

    quality_policy_id: string
    quality_router_model_version: string
    created_at: timestamp
}
```

Each phase:

```text
ModelPhasePlan {
    phase: int

    model_id: string
    model_revision: string

    predicted_quality_probability: float
    required_quality_probability: float

    quality_confidence: float
    uncertainty_class: string?

    model_cost_prior: float

    selection_reason: string

    capability_record_version: string
    model_representation_version: string?
}
```

### Phase semantics

```text
phase 0 = primary quality-selected model revision
phase 1 = quality-approved fallback 1
phase 2 = quality-approved fallback 2
```

These phase identities are later preserved through:

```text
TargetModelContextMaterializer
Planner
ReplicaScheduler
ExecutionCoordinator
RequestQueue
RequestExecutionRegistry
```

and are used for stale-work fencing.

---

# 20. Calibration and Uncertainty

If Phase 2 uses:

```text
P(success) >= 0.95
```

as a production decision threshold, the score must be calibrated enough for the probability to have operational meaning.

Useful model-quality metrics include:

```text
calibration error
quality retention
quality regression rate
cost at fixed quality target
quality at fixed cost target
OOD/uncertainty behavior
fallback frequency
```

Routing mistakes are often asymmetric:

```text
unnecessarily strong model
    -> cost regression

too-weak model
    -> user-visible quality regression
```

For many products the second is worse.

Therefore uncertain/OOD requests should generally escalate conservatively toward stronger qualified models.

### Design Principle 14

> **If quality evidence is weak, prefer conservative escalation over aggressive down-routing.**

---

# 21. New-Model Cold Start

A new model revision creates a cold-start problem.

The router should not assume that a new model is quality-equivalent merely because:

```text
same family
same parameter count
same vendor
```

A reasonable onboarding lifecycle is:

```text
New Model Revision
      |
      v
Capability Validation
      |
      v
Offline Qualification
      |
      v
Empirical Quality Profile
      |
      v
Shadow Routing
      |
      v
Conservative Production Traffic
      |
      v
Production Outcomes
      |
      v
Refined Model Representation
```

The model representation can combine:

```text
empirical quality profile
+
learned model-revision embedding
```

This avoids requiring full quality-router retraining for every new model while still requiring evidence before automatic routing.

### Design Principle 15

> **New-model evaluation is unavoidable; full router retraining for every model revision should not be.**

---

# 22. Failure Semantics

Phase 2 can fail for reasons distinct from Phase 3.

## Phase-2 failures

```text
explicit model does not exist
explicit model violates capability/policy
no model satisfies deterministic capability constraints
no model satisfies required quality threshold
quality predictor unavailable
quality estimate too uncertain
model revision is not production-qualified
```

Possible configured behavior:

```text
reject
route to strongest qualified default
bypass learned routing and use conservative default
exclude cold/unqualified models from auto routing
```

## Not Phase-2 failures

These belong downstream:

```text
all replicas overloaded
replica unavailable
regional serving capacity exhausted
KV cache missing
KV transfer failed
context recompute failed on one replica
SLO infeasible on current serving fleet
```

Separating these error domains keeps operational diagnosis clear.

---

# 23. Observability

The component should emit enough information to answer:

```text
Why was this model selected?
Which models were capability-filtered?
What quality probability was predicted?
What quality threshold was required?
Which fallbacks were quality-approved?
Did uncertainty force escalation?
How much model-level cost did auto-routing save?
How well calibrated is the router?
How do new model revisions behave?
```

Useful fields:

```text
request_id
context_snapshot_id

candidate model revisions
capability-filter reasons

predicted quality per candidate
required quality threshold
quality confidence / uncertainty

selected primary
fallback phases

model cost priors

quality policy version
quality router model version
model representation versions

actual downstream quality outcome when later available
estimated quality-routing cost savings
```

Prompt/session content should not be logged by default.

---

# 24. Invariants

1. Every auto-selected phase satisfies deterministic capability requirements.
2. Every auto-selected phase satisfies the configured quality threshold unless an explicit degraded-mode policy is surfaced.
3. Explicit model pins are preserved after capability/policy validation.
4. The quality predictor estimates request-model quality, not serving latency/load.
5. Phase 2 never chooses hardware, engine, deployment, replica, or context-transfer source.
6. Session content may affect quality prediction; session/KV placement does not.
7. Model-level cost may affect Phase-2 ordering, but live replica execution cost belongs to Phase 3.
8. Exact target-model tokenization happens after model selection.
9. Every fallback in `ModelRoutePlan` is quality-approved before Phase 3 sees it.
10. Quality-router outputs used as probabilities should be calibrated and versioned.

---

# 25. Design Principles Summary

1. **Model selection is a constrained quality problem, not “pick the biggest model.”**
2. **Quality selection operates on model-independent semantics; target-model preprocessing comes later.**
3. **Phase 2 selects a model revision; Phase 3 selects serving execution.**
4. **Filter deterministic capability constraints before learned quality prediction.**
5. **Respect explicit model pins.**
6. **Keep the required quality threshold outside the predictor.**
7. **Predict `P(success | request, model revision)`, not request difficulty alone.**
8. **Learn request-model compatibility.**
9. **Keep quality prediction independent from cost; combine model-level cost only in the decision policy.**
10. **Conversation semantics may affect quality; KV/session placement may not.**
11. **Router semantic representation and target-model token representation are different abstractions.**
12. **Avoid candidate-by-candidate exact tokenization unless its benefit justifies the cost.**
13. **Return quality-approved model fallbacks so Phase 3 does not make semantic quality judgments.**
14. **Treat calibration and uncertainty as production requirements.**
15. **Treat new-model cold start and qualification as part of the routing lifecycle.**

---

# 26. Next Component

For each `ModelPhasePlan`, `TargetModelContextMaterializer` receives:

```text
LogicalContextSnapshot
+
model_id
+
model_revision
```

and produces the exact model-specific execution artifact:

```text
chat-template serialization
token IDs / token artifact reference
exact input-token count
canonical prefix block keys
model/tokenizer/template revisions
target-context digest
KV compatibility inputs
```

After this boundary the system can reason about precise prefill work, KV locality, context transfer, and replica scheduling without changing the quality decision already made in Phase 2.
