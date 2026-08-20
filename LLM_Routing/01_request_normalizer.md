# Component — RequestNormalizer

## Why RequestNormalizer Is Needed

An LLM routing platform does not receive one universal request format.

Different public APIs and internal products can represent logically similar inference work using different wire schemas:

```text
OpenAI Responses / Chat-style requests
Anthropic Messages requests
plain text Completion requests
Embedding requests
multimodal requests
internal session-driven chat requests
future provider-specific APIs
```

The difference is not merely field naming. Providers may use different:

```text
message structures
content-block formats
system-instruction conventions
tool definitions
model fields
output-token fields
multimodal representations
conversation/session conventions
```

If downstream components consume these raw schemas directly, provider knowledge leaks throughout the routing layer:

```text
                           +--> QualityModelSelector understands OpenAI
                           +--> QualityModelSelector understands Anthropic
Raw provider request ------+--> Planner understands OpenAI
                           +--> Planner understands Anthropic
                           +--> Session logic understands every API
                           +--> Observability understands every API
```

Adding a new API then requires changes across the entire system.

Instead we establish one protocol boundary:

```text
OpenAI -----------+
                  |
Anthropic --------+
                  |
Completion -------+--> Protocol Adapter / RequestNormalizer
                  |               |
Embedding --------+               v
                  |        NormalizedRequest
Internal Session -+               |
                                  +--> ContextInitializer
                                  +--> QualityModelSelector
                                  +--> policy / observability
```

The provider-specific wire protocol is understood once. Everything downstream reasons about stable routing semantics.

### Design Principle

> **Normalize heterogeneous wire APIs before routing so downstream components do not depend on provider-specific request schemas.**

---

## Representative Request Types

The examples below are intentionally simplified. Public API schemas evolve; that is exactly why provider-specific details belong behind protocol adapters.

### Example 1 — OpenAI Responses-Style Generation

A representative OpenAI-style generation request can look conceptually like:

```json
{
  "model": "model-A",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Explain paged attention."
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "lookup_document"
    }
  ],
  "max_output_tokens": 500
}
```

The OpenAI adapter extracts logical semantics:

```text
operation           = CHAT
request_mode        = STATELESS

inline_messages     = [
    Message(role=USER, content="Explain paged attention.")
]

tool_definitions    = [lookup_document]
explicit_model      = "model-A"
max_output_tokens   = 500
```

Downstream components do not need to know that the source API called the top-level input field `input`.

---

### Example 2 — Anthropic Messages-Style Generation

A representative Anthropic-style request can have a different wire shape:

```json
{
  "model": "model-B",
  "max_tokens": 500,
  "system": "You are a systems-design assistant.",
  "messages": [
    {
      "role": "user",
      "content": "Explain paged attention."
    }
  ]
}
```

The Anthropic adapter produces the same logical routing concepts:

```text
operation           = CHAT
request_mode        = STATELESS

inline_messages     = [
    Message(
        role=SYSTEM,
        content="You are a systems-design assistant."
    ),
    Message(
        role=USER,
        content="Explain paged attention."
    )
]

explicit_model      = "model-B"
max_output_tokens   = 500
```

The provider wire formats differ, but `ContextInitializer` and `QualityModelSelector` see the same internal semantics.

---

### Example 3 — Plain Text Completion

A completion API may carry a raw prompt rather than role-structured messages:

```json
{
  "model": "model-C",
  "prompt": "The advantages of paged KV cache are:",
  "max_tokens": 300,
  "temperature": 0.2
}
```

Normalization preserves the semantic distinction:

```text
operation           = COMPLETION
request_mode        = STATELESS
explicit_model      = "model-C"
max_output_tokens   = 300

sampling_params = {
    temperature: 0.2
}
```

The completion input is represented internally as logical completion context rather than pretending it was originally a provider-specific chat request.

This matters later because a completion model may not use a chat template at all.

---

### Example 4 — Embedding Request

An embedding request is a different operation:

```json
{
  "model": "embedding-model",
  "input": [
    "paged attention",
    "continuous batching"
  ]
}
```

Conceptually:

```text
operation           = EMBEDDING
request_mode        = STATELESS
explicit_model      = "embedding-model"
decode work         = not applicable
```

The important design point is that downstream routing should use `operation=EMBEDDING`, not repeatedly inspect raw JSON to infer that this request has no decode phase.

An implementation can represent embedding inputs in an operation-specific normalized payload while retaining the same common request envelope.

---

### Example 5 — Multimodal Chat

A provider may encode text and media as typed content blocks:

```json
{
  "model": "vision-model",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is unusual in this image?"
        },
        {
          "type": "image_url",
          "url": "https://..."
        }
      ]
    }
  ]
}
```

The adapter converts provider-specific media syntax into logical content references:

```text
inline_messages = [
    Message(
        role=USER,
        content=[
            TextPart("What is unusual in this image?"),
            AttachmentPart(ImageRef(...))
        ]
    )
]

attachments = [ImageRef(...)]
```

The RequestNormalizer does **not** convert that image into model-specific vision tokens. That is target-model preprocessing.

---

### Example 6 — V5 Session-Driven Chat

Our own stateful API may deliberately avoid resending the full conversation:

```json
{
  "session_id": "S123",
  "expected_session_version": 27,
  "message": "What if R1 is overloaded?",
  "max_output_tokens": 500
}
```

Normalization becomes:

```text
operation                  = CHAT
request_mode               = SESSION

session_id                 = "S123"
expected_session_version   = 27

new_turn = Message(
    role=USER,
    content="What if R1 is overloaded?"
)

inline_messages            = null
max_output_tokens          = 500
```

The RequestNormalizer does not fetch the previous 27 versions of conversation state.

That belongs to:

```text
NormalizedRequest
      |
      v
ContextInitializer
      |
      v
SessionStore
```

This keeps provider parsing and durable session-state ownership separate.

---

## Why Normalization Is Not Model-Specific Preprocessing

Normalization produces a **model-independent logical request**.

It must not apply a generic tokenizer or assume one universal chat template.

The same logical chat request can become different execution sequences for different model revisions:

```text
Logical Conversation
        |
        +--> Model A chat template/tokenizer -> token sequence A
        |
        +--> Model B chat template/tokenizer -> token sequence B
```

In V5, Phase 2 can choose the model revision after request normalization. Exact target-model formatting and tokenization therefore belongs later:

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
    v
LogicalContextSnapshot
    |
    v
QualityModelSelector
    |
    v
Selected ModelRevision
    |
    v
TargetModelContextMaterializer
    |
    +--> exact prompt serialization
    +--> exact token IDs
    +--> input token count
    +--> canonical prefix block keys
```

This boundary is required for correct:

```text
context-window validation
prefill cost estimation
KV-prefix lookup
cross-replica KV transfer compatibility
session-driven model changes
```

### Design Principle

> **Normalize semantics first; perform exact model-dependent preprocessing only after the target model revision is known.**

---

## What This Boundary Buys Us

After normalization, the rest of the platform can be tested independently of provider protocols:

```text
Protocol tests
    -> Does OpenAI / Anthropic / internal API normalize correctly?

Context tests
    -> Given NormalizedRequest, is logical context reconstructed correctly?

Quality-routing tests
    -> Given logical context, which model revision is acceptable?

Replica-routing tests
    -> Given target-model context and replica state, what execution plan is selected?
```

A new provider API should normally require:

```text
new / updated Protocol Adapter
        |
        v
same NormalizedRequest contract
```

rather than changes to every downstream component.

---

## Responsibility

Normalize heterogeneous wire APIs into one stable internal request contract. It does not load session history, choose a model, tokenize for a target model, or perform replica scheduling.

## Input — `WireRequest`

The exact public API may differ. The normalizer extracts these logical fields:

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | Globally unique request identifier. Generated if client did not provide one. |
| `idempotency_key` | string? | Client/request retry identity. |
| `operation` | enum | `CHAT`, `COMPLETION`, `EMBEDDING`, future operations. |
| `session_id` | string? | Durable conversation identifier for stateful mode. |
| `expected_session_version` | int? | Optional optimistic client version. |
| `messages` | list<Message>? | Full stateless context when supplied by client. |
| `new_turn` | Message? | New turn for stateful session mode. |
| `attachments` | list<AttachmentRef> | Images/files/tool payload references. |
| `tool_definitions` | list<ToolDefinition> | Tools available for this request. |
| `explicit_model` | string? | Explicit model pin; bypasses auto model selection after validation. |
| `max_output_tokens` | int | Requested generation bound. |
| `sampling_params` | map | Temperature/top-p/etc.; carried through, not interpreted for routing except where explicitly modeled. |
| `quality_requirement` | float/string | Product quality threshold/class. |
| `ttft_slo_ms` | float? | Time-to-first-token SLO if applicable. |
| `e2e_slo_ms` | float? | End-to-end latency SLO if applicable. |
| `slo_probability_target` | float | Required probability of meeting the active latency SLO, e.g. `0.99`. |
| `cost_policy_id` | string | Cost/accounting policy to apply downstream. |
| `tenant_id` | string | Tenant/account identity for policy/fairness/accounting. |
| `region` | string? | Required/preferred serving region. |
| `policy_tags` | set<string> | Compliance/product routing constraints. |
| `received_at` | timestamp | Router ingress time. |

`Message` is a logical structured message, not a model-specific token sequence:

```text
Message {
    role
    content / content_ref
    tool_call_id?
    metadata
}
```

## Output — `NormalizedRequest`

```text
NormalizedRequest {
    request_id: string
    idempotency_key: string?

    operation: Operation
    request_mode: STATELESS | SESSION

    session_id: string?
    expected_session_version: int?

    inline_messages: list<Message>?
    new_turn: Message?
    attachments: list<AttachmentRef>
    tool_definitions: list<ToolDefinition>

    explicit_model: string?

    max_output_tokens: int
    sampling_params: map

    quality_requirement: QualityRequirement

    ttft_slo_ms: float?
    e2e_slo_ms: float?
    slo_probability_target: float
    cost_policy_id: string

    tenant_id: string
    region: string?
    policy_tags: set<string>

    received_at: timestamp
}
```

## Validation invariants

- `SESSION` requires `session_id` and `new_turn`.
- `STATELESS` requires sufficient inline context for the operation.
- A request cannot ambiguously specify both a full mutable session continuation and an unrelated full conversation without an explicit API contract.
- SLO probability must be in `(0, 1]`.
- Hard API/capability validation happens here where deterministic.

## Next component

`ContextInitializer` receives exactly one `NormalizedRequest`.
