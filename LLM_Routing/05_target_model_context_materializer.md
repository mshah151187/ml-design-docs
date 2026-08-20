# Component — TargetModelContextMaterializer

## Responsibility

Convert a model-independent `LogicalContextSnapshot` into the exact tokenized context required by one selected model revision.

This is where chat template, tokenizer revision, special-token policy, truncation/compaction contract, and exact token IDs become concrete.

This component does **not** decide where the request executes and does not move KV between replicas.

## Input — `TargetContextMaterializationRequest`

```text
TargetContextMaterializationRequest {
    request_id: string
    phase: int

    context_snapshot_id: string
    logical_context_ref: string
    logical_context_digest: string

    session_id: string?
    base_session_version: int?

    model_id: string
    model_revision: string

    max_output_tokens: int
    sampling_params_ref: string
}
```

## Model serialization metadata

```text
TargetModelSerializationProfile {
    model_id: string
    model_revision: string

    tokenizer_revision: string
    chat_template_revision: string
    special_token_policy_version: string

    max_context_tokens: int
    serialization_policy_id: string
}
```

## Processing

1. Resolve the exact tokenizer/template for the selected model revision.
2. Serialize structured logical events into the model's exact chat/completion format.
3. Apply the configured context-window/compaction policy.
4. Tokenize exactly as the target serving deployment expects.
5. Store/reuse the token artifact in a reconstructible artifact cache.
6. Compute a token-context digest.

Prefix block keys are namespace-dependent because different KV compatibility profiles can use different block/hash schemas. Therefore the canonical output is exact tokens; the Planner/KV-index client derives block keys for each candidate `kv_compatibility_id`.

## Output — `TargetModelContextArtifact`

```text
TargetModelContextArtifact {
    artifact_id: string

    request_id: string
    phase: int

    context_snapshot_id: string
    session_id: string?
    base_session_version: int?

    model_id: string
    model_revision: string

    tokenizer_revision: string
    chat_template_revision: string
    special_token_policy_version: string
    serialization_policy_id: string

    token_ids_ref: string
    input_token_count: int

    token_context_digest: string

    max_output_tokens: int
    expected_output_tokens: int
    sampling_params_ref: string

    artifact_version: string
    created_at: timestamp
    expires_at: timestamp?
}
```

### Field semantics

| Field | Meaning |
|---|---|
| `token_ids_ref` | Reference to exact target-model input token IDs; large token arrays need not traverse every RPC. |
| `token_context_digest` | Identity of exact model-specific context used for KV validation/transfer. |
| `expected_output_tokens` | Routing estimate, distinct from hard `max_output_tokens`. |
| `artifact_version` | Artifact schema/version for safe cache reuse. |

## Invariants

- Same artifact identity means identical model-specific context serialization.
- Token IDs are reconstructible from SessionStore/logical snapshot; they are not the durable conversation truth.
- Any KV reused or transferred for this request must correspond to the same model revision and token-context prefix under a compatible KV namespace.

## Next component

`Planner` consumes this artifact for one model phase.
