# Model Bundle — Production Cross-Stage Artifact Contract

## Status

This document replaces the earlier Model Bundle schema draft.

The original design correctly established a rigid cross-stage bundle, protobuf metadata, feature contracts, canonical/original artifacts, and serving-specific derived artifacts. The production schema below keeps those ideas but fixes the inconsistencies around undefined types, duplicated metadata, vague fields, artifact references, serving profiles, immutability, and runtime ownership.

The main contract is:

```text
Training / Export
      |
      v
Canonical Model
      |
      v
Model Optimization
      |
      +--> qualified ServingProfile(s)
      |
      v
Immutable Model Bundle
      |
      v
Model Registry
      |
      v
Serving Platform
```

---

# 1. Core Definition

A **Model Bundle** is the immutable, versioned contract exchanged between:

```text
Training Platform
Evaluation / Safety
Model Optimization Platform
Model Registry
Serving Platform
Compliance / Lineage
Observability
```

It contains:

```text
canonical model truth
+
optional auxiliary pre/post-processing components
+
qualified serving realizations
+
lineage / evaluation / approval metadata
+
artifact integrity metadata
```

The root protobuf is:

```text
bundle_manifest.pb
```

and is the **single authoritative metadata entry point**.

There are no duplicated authoritative `model_metadata.pb` files that another writer must keep synchronized.

---

# 2. Canonical Model vs Serving Realization

The most important separation is:

```text
CanonicalModel
    =
    what was trained/exported

ServingProfile
    =
    one qualified way of serving it
```

Therefore the canonical model does **not** contain:

```text
vLLM
SGLang
TensorRT-LLM
TP degree
GPU SKU
quantization
CUDA Graph settings
max batch size
GPU memory utilization
container image
```

Those belong to `ServingProfile`.

---

# 3. Published Bundle Immutability

Use two lifecycle concepts:

```text
BundleBuildWorkspace
    mutable while pipeline stages are running

Published ModelBundleVersion
    immutable
```

Once published:

```text
model = assistant-70b
bundle_version = 42
```

the files and `bundle_manifest.pb` never change.

A new optimization result or material artifact change produces another immutable bundle version.

Mutable aliases such as:

```text
champion
candidate
canary
```

exist in Model Registry, outside the bundle.

---

# 4. Directory Structure

```text
model_bundle/
│
├── bundle_manifest.pb                    # Only authoritative metadata manifest.
│
├── canonical/
│   ├── model/
│   │   ├── config.json                   # Architecture/model configuration.
│   │   ├── model-00001-of-000NN.safetensors
│   │   └── model.safetensors.index.json
│   │
│   ├── tokenizer/                        # Optional for non-text models.
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   └── ...
│   │
│   └── interface/                        # Optional extra interface assets.
│       └── chat_template.jinja
│
├── auxiliary/
│   ├── preprocessing/                    # Optional.
│   │   └── model.pt
│   └── postprocessing/                   # Optional.
│       └── model.pt
│
├── serving_profiles/
│   ├── h200-vllm-fp8-tp4/
│   │   └── artifacts/
│   │       └── quantized/
│   │
│   ├── h100-vllm-fp8-tp8/
│   │   └── artifacts/
│   │       └── quantized/
│   │
│   ├── h100-trtllm-fp8-tp8/
│   │   └── artifacts/
│   │       ├── trtllm_checkpoint/
│   │       ├── engine/
│   │       └── build_manifest.pb
│   │
│   └── a100-vllm-awq-tp8/
│       └── artifacts/
│           └── quantized/
│
└── reports/
    ├── evaluation/
    ├── optimization/
    └── attestations/
```

`bundle_manifest.pb` embeds the metadata for canonical model, auxiliary components, and serving profiles. The directories contain artifact bytes, not competing authoritative metadata copies.

---

# 5. Complete Production Protobuf Schema

The schema below is intentionally self-contained: every referenced enum and message is defined.

```proto
syntax = "proto3";

package ml.platform.modelbundle.v1;

import "google/protobuf/duration.proto";
import "google/protobuf/timestamp.proto";

// ============================================================================
// ENUMS
// ============================================================================

// High-level inference behavior of the canonical model.
// This describes model semantics, not the serving engine.
//
// Example:
//   Llama-3.1-70B-Instruct -> INFERENCE_TYPE_TEXT_GENERATION
//   text-embedding-3       -> INFERENCE_TYPE_TEXT_EMBEDDING
enum InferenceType {
  INFERENCE_TYPE_UNSPECIFIED = 0;
  INFERENCE_TYPE_TEXT_GENERATION = 1;
  INFERENCE_TYPE_TEXT_EMBEDDING = 2;
  INFERENCE_TYPE_RERANKING = 3;
  INFERENCE_TYPE_CLASSIFICATION = 4;
  INFERENCE_TYPE_REGRESSION = 5;
  INFERENCE_TYPE_IMAGE_INFERENCE = 6;
  INFERENCE_TYPE_MULTIMODAL_GENERATION = 7;
  INFERENCE_TYPE_CUSTOM = 100;
}

// Scalar storage / execution data types used by canonical and optimized
// artifacts. Do not use free-form strings for common dtypes.
//
// Example:
//   Canonical Llama weights -> DATA_TYPE_BFLOAT16
//   H200 FP8 serving profile -> DATA_TYPE_FP8_E4M3
enum DataType {
  DATA_TYPE_UNSPECIFIED = 0;
  DATA_TYPE_FLOAT32 = 1;
  DATA_TYPE_FLOAT16 = 2;
  DATA_TYPE_BFLOAT16 = 3;
  DATA_TYPE_FP8_E4M3 = 4;
  DATA_TYPE_FP8_E5M2 = 5;
  DATA_TYPE_INT8 = 6;
  DATA_TYPE_UINT8 = 7;
  DATA_TYPE_INT4 = 8;
  DATA_TYPE_INT32 = 9;
  DATA_TYPE_INT64 = 10;
  DATA_TYPE_BOOL = 11;
}

// Input/output modality understood by the canonical model interface.
//
// Example:
//   LLM -> input TEXT, output TEXT
//   VLM -> input TEXT + IMAGE, output TEXT
enum Modality {
  MODALITY_UNSPECIFIED = 0;
  MODALITY_TEXT = 1;
  MODALITY_IMAGE = 2;
  MODALITY_AUDIO = 3;
  MODALITY_VIDEO = 4;
  MODALITY_TENSOR = 5;
}

// Optional auxiliary model components carried in the same bundle.
//
// Example:
//   PREPROCESSING may be a TorchScript feature transformation module.
enum ModelComponentType {
  MODEL_COMPONENT_TYPE_UNSPECIFIED = 0;
  MODEL_COMPONENT_TYPE_PREPROCESSING = 1;
  MODEL_COMPONENT_TYPE_POSTPROCESSING = 2;
}

// Logical kind of an artifact reference.
//
// DIRECTORY means sha256 is computed over a deterministic directory manifest,
// not over filesystem metadata such as mtime.
enum ArtifactKind {
  ARTIFACT_KIND_UNSPECIFIED = 0;
  ARTIFACT_KIND_FILE = 1;
  ARTIFACT_KIND_DIRECTORY = 2;
}

// Format of a model/runtime artifact.
//
// Example:
//   canonical/model/ -> ARTIFACT_FORMAT_SAFETENSORS
//   TensorRT-LLM engine directory -> ARTIFACT_FORMAT_TRTLLM_ENGINE
enum ArtifactFormat {
  ARTIFACT_FORMAT_UNSPECIFIED = 0;
  ARTIFACT_FORMAT_SAFETENSORS = 1;
  ARTIFACT_FORMAT_QUANTIZED_SAFETENSORS = 2;
  ARTIFACT_FORMAT_TRTLLM_CHECKPOINT = 3;
  ARTIFACT_FORMAT_TRTLLM_ENGINE = 4;
  ARTIFACT_FORMAT_ONNX = 5;
  ARTIFACT_FORMAT_TORCHSCRIPT = 6;
  ARTIFACT_FORMAT_CUSTOM = 100;
}

// Serving runtime used by one qualified profile.
//
// Runtime version is stored separately because vLLM 0.x and vLLM 1.x may have
// materially different behavior even though runtime_kind is the same.
enum RuntimeKind {
  RUNTIME_KIND_UNSPECIFIED = 0;
  RUNTIME_KIND_VLLM = 1;
  RUNTIME_KIND_SGLANG = 2;
  RUNTIME_KIND_TENSORRT_LLM = 3;
  RUNTIME_KIND_TRITON = 4;
  RUNTIME_KIND_ONNX_RUNTIME = 5;
  RUNTIME_KIND_CUSTOM = 100;
}

// Quantization method applied to one ServingProfile.
//
// NONE means profile uses the canonical precision or another non-quantized
// serving dtype.
enum QuantizationType {
  QUANTIZATION_TYPE_UNSPECIFIED = 0;
  QUANTIZATION_TYPE_NONE = 1;
  QUANTIZATION_TYPE_FP8 = 2;
  QUANTIZATION_TYPE_INT8 = 3;
  QUANTIZATION_TYPE_INT4_AWQ = 4;
  QUANTIZATION_TYPE_INT4_GPTQ = 5;
  QUANTIZATION_TYPE_SMOOTHQUANT = 6;
  QUANTIZATION_TYPE_CUSTOM = 100;
}

// Role of a ServingProfile.
//
// COLOCATED is the normal single-engine case.
// PREFILL / DECODE allow the same bundle to carry qualified profiles for a
// disaggregated serving topology without pretending both roles are identical.
enum ServingRole {
  SERVING_ROLE_UNSPECIFIED = 0;
  SERVING_ROLE_COLOCATED = 1;
  SERVING_ROLE_PREFILL = 2;
  SERVING_ROLE_DECODE = 3;
  SERVING_ROLE_EMBEDDING = 4;
  SERVING_ROLE_CUSTOM = 100;
}

// Minimum accelerator interconnect required by a profile.
//
// Example:
//   TP8 H100 profile -> INTERCONNECT_CLASS_NVLINK
enum InterconnectClass {
  INTERCONNECT_CLASS_UNSPECIFIED = 0;
  INTERCONNECT_CLASS_PCIE = 1;
  INTERCONNECT_CLASS_NVLINK = 2;
  INTERCONNECT_CLASS_NVSWITCH = 3;
  INTERCONNECT_CLASS_HIGH_SPEED_FABRIC = 4;
}

// CUDA Graph behavior qualified by Model Optimization.
//
// This is separate from compilation; a compiled runtime may still use
// CUDA Graph capture/replay.
enum CudaGraphMode {
  CUDA_GRAPH_MODE_UNSPECIFIED = 0;
  CUDA_GRAPH_MODE_DISABLED = 1;
  CUDA_GRAPH_MODE_FULL = 2;
  CUDA_GRAPH_MODE_PIECEWISE = 3;
  CUDA_GRAPH_MODE_DECODE_ONLY = 4;
  CUDA_GRAPH_MODE_RUNTIME_DEFAULT = 5;
}


// Provenance of the traffic distribution used to qualify a ServingProfile.
//
// EXPECTED:
//   Product-team supplied workload expectation available before initial
//   production deployment.
//
// OBSERVED:
//   Immutable snapshot derived from canary/production telemetry and supplied
//   to a later Model Optimization run.
//
// Example:
//   Initial profile -> TRAFFIC_DISTRIBUTION_SOURCE_EXPECTED
//   Reoptimized profile after production drift
//       -> TRAFFIC_DISTRIBUTION_SOURCE_OBSERVED
enum TrafficDistributionSource {
  TRAFFIC_DISTRIBUTION_SOURCE_UNSPECIFIED = 0;
  TRAFFIC_DISTRIBUTION_SOURCE_EXPECTED = 1;
  TRAFFIC_DISTRIBUTION_SOURCE_OBSERVED = 2;
}

// Semantic operation the canonical model is qualified to perform.
//
// These are model-level capabilities and are independent of the selected
// ServingProfile hardware/runtime realization.
//
// Example:
//   assistant LLM -> CHAT_COMPLETION + TEXT_GENERATION
//   embedding model -> TEXT_EMBEDDING
enum ModelOperation {
  MODEL_OPERATION_UNSPECIFIED = 0;
  MODEL_OPERATION_CHAT_COMPLETION = 1;
  MODEL_OPERATION_TEXT_GENERATION = 2;
  MODEL_OPERATION_TEXT_EMBEDDING = 3;
  MODEL_OPERATION_RERANKING = 4;
  MODEL_OPERATION_CLASSIFICATION = 5;
  MODEL_OPERATION_REGRESSION = 6;
  MODEL_OPERATION_MULTIMODAL_GENERATION = 7;
  MODEL_OPERATION_CUSTOM = 100;
}

// Tool-choice semantics the canonical model is qualified to support.
//
// Example:
//   AUTO lets the model decide whether to call a tool.
//   REQUIRED requires at least one tool call.
enum ToolChoiceMode {
  TOOL_CHOICE_MODE_UNSPECIFIED = 0;
  TOOL_CHOICE_MODE_AUTO = 1;
  TOOL_CHOICE_MODE_NONE = 2;
  TOOL_CHOICE_MODE_REQUIRED = 3;
  TOOL_CHOICE_MODE_NAMED = 4;
}

// Type of approval attached to the immutable bundle.
enum ApprovalType {
  APPROVAL_TYPE_UNSPECIFIED = 0;
  APPROVAL_TYPE_QUALITY = 1;
  APPROVAL_TYPE_SAFETY = 2;
  APPROVAL_TYPE_PRIVACY = 3;
  APPROVAL_TYPE_COMPLIANCE = 4;
  APPROVAL_TYPE_PRODUCTION = 5;
}

// ============================================================================
// ROOT MANIFEST
// ============================================================================

// Single authoritative metadata object for one immutable ModelBundleVersion.
//
// IMPORTANT:
// - This manifest intentionally does NOT contain its own SHA256.
// - Model Registry stores the manifest digest externally, because embedding a
//   digest of the manifest inside the manifest would be self-referential.
//
// Example immutable registry identity:
//   assistant-70b / bundle_version=42 / manifest_sha256=abc123...
message BundleManifest {
  // Schema major version understood by bundle readers.
  // Example: 1
  uint32 schema_version = 1;

  // Stable identity and ownership of this immutable bundle version.
  BundleIdentity identity = 2;

  // Canonical trained/exported model. Always present.
  // Includes model-level CapabilityContract and QualityEvaluation.
  CanonicalModel canonical_model = 3;

  // Optional preprocessing / postprocessing modules.
  // Example:
  //   component_id="feature-normalizer", type=PREPROCESSING.
  repeated AuxiliaryModel auxiliary_models = 4;

  // Only published/qualified serving realizations plus their
  // primary/alternative ordering.
  ServingProfileCatalog serving_profiles = 5;

  // Training provenance for the canonical model.
  ModelLineage lineage = 6;

  // Required human/platform approvals.
  // Example: safety review "approved-2026-08-20".
  repeated Approval approvals = 7;

  // Supply-chain and artifact attestation references.
  SupplyChainMetadata supply_chain = 8;
}

// ============================================================================
// BUNDLE IDENTITY
// ============================================================================

message BundleIdentity {
  // Stable model family/name.
  // Example: "assistant-70b"
  string model_name = 1;

  // Immutable bundle version allocated by Model Registry/build pipeline.
  // Example: "42"
  string bundle_version = 2;

  // Creation timestamp of this immutable publication.
  google.protobuf.Timestamp created_at = 3;

  // User/service principal that finalized the bundle.
  // Example: "model-optimization-controller"
  string created_by = 4;

  // Owning team / use case metadata.
  TeamMetadata owner = 5;
}

message TeamMetadata {
  // Owning organization/team.
  // Example: "feed-ml"
  string team = 1;

  // Product/use-case identifier.
  // Example: "connection_suggestion"
  string use_case = 2;

  // Operational contact or owning group.
  // Example: "feed-ml-oncall"
  string owner = 3;
}

// ============================================================================
// CANONICAL MODEL
// ============================================================================

// Immutable model truth produced by training/export.
//
// This message intentionally contains no serving runtime, GPU, TP, or
// quantization decision.
message CanonicalModel {
  // High-level model behavior.
  // Example: INFERENCE_TYPE_TEXT_GENERATION.
  InferenceType inference_type = 1;

  // Typed architecture description; detailed architectural hyperparameters
  // remain authoritative in model_config.
  // Example: family="llama", revision="llama-3.1".
  ModelArchitecture architecture = 2;

  // Optional parent/base model from which this model was fine-tuned.
  // Absent for a model trained from scratch.
  // Example: meta-llama/Llama-3.1-70B-Instruct @ immutable commit SHA.
  optional ModelReference base_model = 3;

  // Storage dtype of the canonical exported weights.
  // Example: DATA_TYPE_BFLOAT16.
  //
  // This is independent from ServingProfile precision; an FP8 serving profile
  // may derive from canonical BF16 weights.
  DataType weight_dtype = 4;

  // Canonical model weights or sharded weight directory.
  // Example: bundle_path="canonical/model/".
  ArtifactRef weights = 5;

  // Architecture/configuration artifact needed to reconstruct the model.
  // Example: bundle_path="canonical/model/config.json".
  ArtifactRef model_config = 6;

  // Tokenizer directory for text models.
  // Absent for models that do not tokenize text.
  // Example: bundle_path="canonical/tokenizer/".
  optional ArtifactRef tokenizer = 7;

  // Technical input/output and model-interface semantics.
  // Example: TEXT -> TEXT, max_context_tokens=131072, streaming=true.
  ModelInterfaceContract interface_contract = 8;

  // Deterministic semantic capabilities of this model revision.
  // These are independent of H100/H200, TP degree, runtime, or quantization.
  //
  // Example:
  //   supported_operations=[CHAT_COMPLETION, TEXT_GENERATION]
  //   tools.supports_tool_calling=true
  //   structured_output.supports_json_schema=true
  CapabilityContract capability_contract = 9;

  // Empirical quality measurements for this canonical model revision.
  // These characterize model quality independently of which qualified
  // ServingProfile is ultimately selected.
  //
  // Example:
  //   MMLU=0.791
  //   HumanEval pass@1=0.742
  //   tool_call_success=0.913
  QualityEvaluation quality_evaluation = 10;

  // Optional online feature lookup contracts for classic ML models.
  // Typically empty for pure prompt-based LLMs.
  repeated FeatureSchema input_features = 11;

  // Optional named model-output feature contracts.
  repeated FeatureSchema output_features = 12;
}

// Structured architecture identity.
//
// "family" is intentionally broader/stabler than one Python class.
// "implementation_class" records the canonical loader implementation.
//
// Example:
//   family="llama"
//   implementation_class="transformers.LlamaForCausalLM"
//   config_class="transformers.LlamaConfig"
//   parameter_count=70553706496
message ModelArchitecture {
  // Stable model family used for compatibility/search.
  // Example: "llama"
  string family = 1;

  // Concrete canonical model implementation expected by export/load tooling.
  // Example: "transformers.LlamaForCausalLM"
  string implementation_class = 2;

  // Concrete configuration type.
  // Example: "transformers.LlamaConfig"
  string config_class = 3;

  // Optional architecture generation/revision.
  // Example: "llama-3.1"
  optional string architecture_revision = 4;

  // Number of learned parameters, if known.
  // Example: 70553706496 for a ~70B model.
  optional uint64 parameter_count = 5;
}

// Typed reference to the parent model used for fine-tuning.
//
// This replaces a free-form base_model string.
message ModelReference {
  // Registry/source namespace.
  // Example: "huggingface" or "internal-model-registry".
  string source = 1;

  // Source model identifier.
  // Example: "meta-llama/Llama-3.1-70B-Instruct"
  string name = 2;

  // Immutable source revision.
  // Prefer a commit SHA/digest rather than "main".
  // Example: "8f3c2a..."
  string revision = 3;

  // Optional artifact/content digest when source provides one.
  // Example: "sha256:abcd..."
  optional string digest = 4;
}

// Replaces the vague ModelSemanticConstraints type.
//
// This describes stable model-interface semantics that serving/training must
// preserve. It does NOT describe production traffic distributions or serving
// SLOs; those belong to ServingRequirements / Model Optimization.
message ModelInterfaceContract {
  // Modalities accepted by the model.
  // Example: [MODALITY_TEXT, MODALITY_IMAGE] for a VLM.
  repeated Modality input_modalities = 1;

  // Modalities produced by the model.
  // Example: [MODALITY_TEXT].
  repeated Modality output_modalities = 2;

  // Architectural maximum context length, when meaningful.
  // Example: 131072 tokens.
  // This is NOT a deployment-specific max_model_len chosen for performance.
  optional uint32 max_context_tokens = 3;

  // Whether the model semantically supports incremental/streaming output.
  // Example: true for autoregressive text generation.
  bool supports_streaming_output = 4;

  // Optional chat-template artifact that defines conversation serialization.
  // Example: bundle_path="canonical/interface/chat_template.jinja".
  optional ArtifactRef chat_template = 5;

  // Named tensor inputs for tensor-oriented models.
  // Example: input_ids:int64[batch,seq].
  repeated TensorSchema input_tensors = 6;

  // Named tensor outputs for tensor-oriented models.
  // Example: logits:float16[batch,seq,vocab].
  repeated TensorSchema output_tensors = 7;
}

// Deterministic semantic capability contract for the canonical model.
//
// This contract is suitable for deterministic candidate filtering in a
// model-routing system. It answers "may this model be considered?" rather than
// "how likely is this model to succeed on this particular request?".
//
// Empirical quality belongs to QualityEvaluation below.
message CapabilityContract {
  // Operations for which this model revision is explicitly qualified.
  // Example:
  //   [MODEL_OPERATION_CHAT_COMPLETION, MODEL_OPERATION_TEXT_GENERATION].
  repeated ModelOperation supported_operations = 1;

  // Tool/function-calling semantics.
  ToolCapabilities tools = 2;

  // Structured-output semantics.
  StructuredOutputCapabilities structured_output = 3;

  // Explicitly qualified language tags using BCP-47 where possible.
  // Empty means the bundle does not declare a deterministic language allowlist;
  // it does NOT mean "all languages are equally good".
  // Example: ["en", "es", "fr", "ja"].
  repeated string supported_language_tags = 4;

  // Explicitly qualified application domains, when domain gating is required.
  // Quality within each domain remains empirical and belongs to
  // QualityEvaluation.
  // Example: ["general", "code", "customer_support"].
  repeated string qualified_domains = 5;

  // Extension point for model capabilities not yet promoted to typed fields.
  // Values must be stable contract metadata, not dynamic serving state.
  // Example: {"supports_reasoning_mode":"true"}.
  map<string, string> attributes = 6;
}

message ToolCapabilities {
  // Whether the model is qualified to emit tool/function calls.
  // Example: true.
  bool supports_tool_calling = 1;

  // Whether multiple tool calls may be emitted in one assistant turn.
  // Example: true.
  bool supports_parallel_tool_calls = 2;

  // Supported tool-choice semantics.
  // Example:
  //   [AUTO, NONE, REQUIRED, NAMED].
  repeated ToolChoiceMode supported_tool_choice_modes = 3;
}

message StructuredOutputCapabilities {
  // Whether the model is qualified to produce JSON-object mode.
  // Example: true.
  bool supports_json_object = 1;

  // Whether the model is qualified for schema-constrained JSON output.
  // Example: true.
  bool supports_json_schema = 2;

  // Whether grammar/CFG-constrained generation is part of the canonical
  // model capability contract.
  // Example: false.
  bool supports_grammar_constraints = 3;
}

// Model-level empirical quality for the canonical model revision.
//
// This is intentionally independent of any selected ServingProfile. The
// evaluation itself still records immutable provenance/report artifacts, but
// these metrics describe semantic model quality rather than H100/H200/vLLM/TRT
// serving performance.
message QualityEvaluation {
  // Immutable canonical evaluation execution.
  // Example: "canonical-eval-553".
  string evaluation_run_id = 1;

  // Stable evaluation suite/version.
  // Example: "assistant-quality-suite-v12".
  string suite_id = 2;

  // Time the evaluation was finalized.
  google.protobuf.Timestamp evaluated_at = 3;

  // Evaluator service/tool identity.
  // Example: "llm-quality-evaluator".
  string evaluated_by = 4;

  // Empirical model-quality metrics.
  //
  // MetricResult.dimensions can identify slices such as:
  //   {"capability":"code", "dataset":"humaneval"}
  //   {"capability":"tool_calling"}
  //   {"language":"fr"}
  repeated MetricResult metrics = 5;

  // Optional immutable detailed evaluation report.
  // Example: bundle_path="reports/evaluation/canonical-quality.pb".
  optional ArtifactRef report = 6;
}

message TensorSchema {
  // Tensor name expected by the canonical model interface.
  // Example: "input_ids".
  string name = 1;

  // Element dtype.
  // Example: DATA_TYPE_INT64.
  DataType dtype = 2;

  // Ordered dimensions.
  // Example:
  //   batch: dynamic 1..256
  //   sequence: dynamic 1..131072
  repeated TensorDimension dimensions = 3;
}

message TensorDimension {
  // Human-readable dimension name.
  // Example: "batch", "sequence", "hidden".
  string name = 1;

  // Fixed size when this dimension is statically known.
  // Example: hidden=8192.
  optional uint64 fixed_size = 2;

  // Minimum semantically valid dynamic size.
  // Example: sequence min=1.
  optional uint64 min_size = 3;

  // Maximum semantically valid dynamic size.
  // Example: sequence max=131072.
  optional uint64 max_size = 4;
}

// Online feature-store lookup plus tensor contract.
//
// This preserves the useful FeatureSchema concept from the original bundle.
message FeatureSchema {
  // Team/product namespace to avoid name collisions.
  // Example: "feed".
  string namespace = 1;

  // Logical feature identifier.
  // Example: "member_embedding_v3".
  string feature_key = 2;

  // Feature store implementation.
  // Example: "venice".
  string feature_store_name = 3;

  // Serving endpoint/service name.
  // Example: "feed-feature-service.prod".
  string feature_service_name = 4;

  // Tensor returned by the feature service.
  TensorSchema tensor = 5;

  // Optional lower validation bound for scalar-compatible values.
  optional double min_value = 6;

  // Optional upper validation bound for scalar-compatible values.
  optional double max_value = 7;
}

// ============================================================================
// OPTIONAL AUXILIARY MODEL COMPONENTS
// ============================================================================

message AuxiliaryModel {
  // Stable component identifier inside this bundle.
  // Example: "query-preprocessor".
  string component_id = 1;

  // PREPROCESSING or POSTPROCESSING.
  ModelComponentType type = 2;

  // Independently meaningful component version.
  // Example: "7".
  string version = 3;

  // Serialized component artifact.
  // Example:
  //   format=TORCHSCRIPT,
  //   bundle_path="auxiliary/preprocessing/model.pt".
  ArtifactRef artifact = 4;

  // Input/output interface for this component.
  ModelInterfaceContract interface_contract = 5;

  // Optional runtime environment when component cannot execute in the main
  // inference container by default.
  optional RuntimeEnvironment runtime = 6;
}

// ============================================================================
// ARTIFACT REFERENCES
// ============================================================================

// Single generic artifact reference used everywhere.
//
// There is intentionally NO separate ModelArtifactRef. The previous draft had
// no meaningful semantic distinction between ModelArtifactRef and ArtifactRef,
// so one type is used consistently.
message ArtifactRef {
  oneof location {
    // Path relative to the root of this immutable Model Bundle.
    // Example: "canonical/model/".
    string bundle_path = 1;

    // External immutable artifact URI, used only when bytes are intentionally
    // not embedded in the bundle.
    // Example: "s3://org-artifacts/calibration/v5/".
    string uri = 2;
  }

  // SHA-256 digest of a file, or of a deterministic directory manifest.
  // Example: "3c7d...".
  string sha256 = 3;

  // Total artifact size in bytes.
  // For directories, total size of referenced files.
  uint64 size_bytes = 4;

  // FILE or DIRECTORY.
  ArtifactKind kind = 5;

  // MIME-like media type for readers/tooling.
  // Example:
  //   "application/x-safetensors"
  //   "application/x-tensorrt-engine"
  //   "application/x-directory"
  string media_type = 6;
}

// ============================================================================
// SERVING PROFILE CATALOG
// ============================================================================

// All qualified runtime realizations shipped with this bundle.
//
// Profiles are grouped by ServingRole so the same bundle can represent either:
// - a normal colocated deployment, or
// - role-specific P/D profiles.
//
// In the common colocated case there is exactly one preference group with
// role=SERVING_ROLE_COLOCATED.
message ServingProfileCatalog {
  // Full immutable profile definitions.
  repeated ServingProfile profiles = 1;

  // Primary + ordered alternatives per serving role.
  repeated ServingProfilePreference preference_groups = 2;
}

message ServingProfilePreference {
  // Role this preference ordering applies to.
  // Example: SERVING_ROLE_COLOCATED.
  ServingRole role = 1;

  // Preferred qualified profile.
  // Example: "h200-vllm-fp8-tp4".
  string primary_profile_id = 2;

  // Ordered qualified fallback profiles.
  // Example:
  //   ["h100-vllm-fp8-tp8", "a100-vllm-awq-tp8"].
  repeated string alternative_profile_ids = 3;
}

// One fully-qualified way to serve the canonical model.
//
// A published profile is immutable and has already passed required quality and
// performance qualification.
message ServingProfile {
  // Bundle-unique stable profile ID.
  // Example: "h200-vllm-fp8-tp4".
  string profile_id = 1;

  // Role this profile performs.
  // Example: SERVING_ROLE_COLOCATED or SERVING_ROLE_DECODE.
  ServingRole role = 2;

  // Optional compatibility key used when multiple role profiles must be paired.
  // Example: prefill and decode profiles both use "pd-layout-v3".
  optional string compatibility_key = 3;

  // Exact runtime/container environment.
  RuntimeEnvironment runtime = 4;

  // Accelerator and host-resource requirements for one logical ServingInstance.
  HardwareRequirement hardware = 5;

  // TP/PP/DP/EP realization.
  ParallelismConfig parallelism = 6;

  // Runtime numerical precision and quantization.
  PrecisionConfig precision = 7;

  // Which artifact format/files this runtime loads.
  ArtifactSelection artifacts = 8;

  // Runtime knobs qualified by Model Optimization.
  RuntimeTuningConfig tuning = 9;

  // Empirical evidence collected by Model Optimization for this exact selected
  // runtime + hardware + precision + parallelism + tuning realization.
  //
  // Only published ServingProfiles (primary + alternatives) carry this object;
  // rejected sweep candidates remain in the Optimization Platform experiment
  // store and do not become bundle ServingProfiles.
  QualificationEvidence qualification = 10;

  // Startup requirements and measured/expected startup characteristics.
  StartupRequirements startup = 11;

  // Provenance back to one optimization experiment.
  OptimizationProvenance provenance = 12;
}

// ============================================================================
// RUNTIME + HARDWARE
// ============================================================================

message RuntimeEnvironment {
  // Runtime implementation.
  // Example: RUNTIME_KIND_VLLM.
  RuntimeKind runtime_kind = 1;

  // Required runtime version.
  // Example: "0.10.2".
  string runtime_version = 2;

  // Optional name when runtime_kind=CUSTOM.
  // Example: "linkedin-inference-runtime".
  optional string custom_runtime_name = 3;

  // Immutable OCI image reference pinned by digest.
  // Example:
  //   "registry/llm-serving@sha256:abcd..."
  string container_image_digest = 4;

  // Optional Python version for Python-based runtimes.
  // Example: "3.11.9".
  optional string python_version = 5;

  // Main framework name, if useful diagnostically.
  // Example: "torch".
  optional string framework_name = 6;

  // Framework version.
  // Example: "2.7.1".
  optional string framework_version = 7;

  // CUDA runtime/toolkit compatibility used during qualification.
  // Example: "12.8".
  optional string cuda_runtime_version = 8;

  // cuDNN version when relevant.
  // Example: "9.7".
  optional string cudnn_version = 9;

  // Additional libraries whose versions materially affect execution.
  // Example:
  //   {"flash-attn":"2.7.4", "transformers":"4.55.0"}.
  map<string, string> library_versions = 10;

  // Process entry point inside the image.
  // Example: "python -m vllm.entrypoints.openai.api_server".
  string entrypoint = 11;

  // Static environment variables required by this qualified profile.
  // Secrets should be references/injected separately, not embedded here.
  map<string, string> environment_variables = 12;
}

message HardwareRequirement {
  // Qualified accelerator SKU/class.
  // Example: "NVIDIA_H200_SXM_141GB".
  string accelerator_model = 1;

  // Accelerators required by one logical ServingInstance.
  // Example: 4 for TP4.
  uint32 accelerator_count = 2;

  // Minimum HBM required on each accelerator.
  // Example: 141000000000.
  uint64 min_accelerator_memory_bytes = 3;

  // Optional CUDA compute capability when toolchain compatibility needs it.
  // Example: "9.0".
  optional string compute_capability = 4;

  // Minimum required intra-instance interconnect.
  // Example: INTERCONNECT_CLASS_NVLINK.
  InterconnectClass min_interconnect = 5;

  // Number of nodes required by one logical ServingInstance.
  // Example: 1 for TP8-on-one-node; 2 for PP/TP multi-node.
  uint32 node_count = 6;

  // Minimum host CPU cores per node required by runtime.
  // Example: 32.
  optional uint32 min_cpu_cores_per_node = 7;

  // Minimum host RAM per node.
  // Example: 536870912000 (~500 GiB).
  optional uint64 min_host_memory_bytes_per_node = 8;

  // Minimum local scratch/cache storage per node.
  // Example: 1000000000000 (1 TB).
  optional uint64 min_local_storage_bytes_per_node = 9;
}

message ParallelismConfig {
  // Tensor-parallel degree.
  // Example: 4.
  uint32 tensor_parallel_size = 1;

  // Pipeline-parallel degree.
  // Example: 1 for no PP.
  uint32 pipeline_parallel_size = 2;

  // Data-parallel degree inside this logical runtime component.
  // Example: 1 for one independent engine.
  uint32 data_parallel_size = 3;

  // Expert-parallel degree for MoE runtimes when applicable.
  // Example: 8.
  optional uint32 expert_parallel_size = 4;
}

// ============================================================================
// PRECISION + ARTIFACT SELECTION
// ============================================================================

message PrecisionConfig {
  // Stored runtime weight dtype.
  // Example: DATA_TYPE_FP8_E4M3.
  DataType weight_dtype = 1;

  // Activation compute dtype.
  // Example: DATA_TYPE_BFLOAT16.
  DataType activation_dtype = 2;

  // KV-cache dtype, if independently configured.
  // Example: DATA_TYPE_FP8_E4M3.
  optional DataType kv_cache_dtype = 3;

  // Quantization algorithm family.
  // Example: QUANTIZATION_TYPE_FP8.
  QuantizationType quantization = 4;

  // Human/tool-specific algorithm identifier.
  // Example: "llm-compressor-fp8-dynamic".
  optional string quantization_algorithm = 5;

  // Representative calibration artifact when required.
  // Example: external URI to immutable calibration dataset sample.
  optional ArtifactRef calibration_artifact = 6;
}

message ArtifactSelection {
  // Runtime-facing artifact format.
  // Example: ARTIFACT_FORMAT_TRTLLM_ENGINE.
  ArtifactFormat format = 1;

  // Main file/directory loaded by runtime.
  // Example:
  //   bundle_path="serving_profiles/h100-trtllm-fp8-tp8/artifacts/engine/".
  ArtifactRef primary_artifact = 2;

  // Supporting artifacts required by runtime.
  // Example:
  //   TRT-LLM config, tokenizer-derived runtime files, auxiliary index.
  repeated ArtifactRef supporting_artifacts = 3;

  // Optional immutable build manifest generated by optimization/compiler.
  // Example:
  //   bundle_path=".../build_manifest.pb".
  optional ArtifactRef build_manifest = 4;
}

// ============================================================================
// RUNTIME TUNING
// ============================================================================

message RuntimeTuningConfig {
  // Deployment-qualified maximum context length.
  // This may be lower than CanonicalModel.interface_contract.max_context_tokens
  // for performance/cost reasons.
  // Example: 32768.
  optional uint32 max_model_len = 1;

  // Maximum requests/sequences handled concurrently by one engine.
  // Example: 256.
  optional uint32 max_num_sequences = 2;

  // Maximum tokens admitted into one scheduler iteration/batch.
  // Example: 8192.
  optional uint32 max_num_batched_tokens = 3;

  // Traditional fixed-batch ceiling when runtime exposes one.
  // Example: 32.
  optional uint32 max_batch_size = 4;

  // Fraction of HBM the runtime is allowed to reserve/use.
  // Example: 0.90.
  optional double gpu_memory_utilization = 5;

  // Whether prefix caching was enabled during qualification.
  bool enable_prefix_caching = 6;

  // Whether chunked prefill was enabled during qualification.
  bool enable_chunked_prefill = 7;

  // CUDA Graph strategy qualified for this profile.
  CudaGraphConfig cuda_graph = 8;

  // Runtime-specific flags not yet promoted to strongly typed fields.
  // Example: {"scheduler-policy":"fcfs"}.
  map<string, string> extra_args = 9;
}

message CudaGraphConfig {
  // Capture/replay strategy.
  // Example: CUDA_GRAPH_MODE_DECODE_ONLY.
  CudaGraphMode mode = 1;

  // Batch sizes captured/qualified when runtime uses batch-size buckets.
  // Example: [1,2,4,8,16,32].
  repeated uint32 capture_batch_sizes = 2;

  // Token-count buckets captured/qualified when runtime exposes token-based
  // graph dispatch.
  // Example: [1,2,4,8,16,32,64,128].
  repeated uint32 capture_token_sizes = 3;
}

// ============================================================================
// PERFORMANCE + QUALITY
// ============================================================================

// Immutable traffic-distribution snapshot used as an input to one profile
// qualification.
//
// The live observed distribution is NOT mutable Model Bundle state. Metrics /
// Serving Platform owns live observations. When a specific expected or
// observed distribution is used by Model Optimization to qualify a published
// profile, that exact immutable snapshot is referenced here.
message TrafficDistributionSnapshot {
  // Stable immutable traffic-profile identifier.
  // Example:
  //   "expected-chat-v3"
  //   "observed-assistant-prod-2026-08-25T18"
  string traffic_profile_id = 1;

  // Whether this was Product-declared expected traffic or telemetry-derived
  // observed traffic.
  // Example: TRAFFIC_DISTRIBUTION_SOURCE_EXPECTED.
  TrafficDistributionSource source = 2;

  // Immutable serialized distribution containing the normalized workload
  // statistics used by Model Optimization.
  //
  // Typical fields inside that artifact include:
  //   prompt-length histogram
  //   output-length histogram
  //   concurrency distribution
  //   request-arrival distribution
  //   streaming mix
  //   tool-call mix
  //   prefix-sharing characteristics
  //
  // Example:
  //   bundle_path="reports/optimization/traffic/expected-chat-v3.pb".
  ArtifactRef distribution = 3;

  // Time this immutable snapshot was finalized.
  google.protobuf.Timestamp created_at = 4;

  // Producer/owner of the snapshot.
  //
  // Expected example:
  //   "assistant-product-team"
  //
  // Observed example:
  //   "serving-traffic-analyzer"
  string produced_by = 5;

  // Deployment/environment from which observed traffic was measured.
  // Normally absent for EXPECTED traffic.
  //
  // Examples:
  //   "canary"
  //   "production"
  optional string observed_environment = 6;

  // Observation window for OBSERVED traffic.
  // Absent for EXPECTED traffic.
  optional google.protobuf.Timestamp observation_window_start = 7;
  optional google.protobuf.Timestamp observation_window_end = 8;

  // Deployment that produced the observation, when source=OBSERVED.
  // Example: "assistant-prod-deployment-42".
  optional string source_deployment_id = 9;
}

// Empirical qualification for one published ServingProfile.
//
// This object is profile-specific. Unlike CanonicalModel.quality_evaluation,
// these measurements are meaningful only for the exact serving realization:
// runtime + environment + hardware + parallelism + precision + artifacts +
// tuning + benchmark workload.
message QualificationEvidence {
  // Optimization experiment selected for publication.
  // Example: "opt-exp-8172-c17".
  string experiment_id = 1;

  // Serving performance measured for this exact profile.
  PerformanceEnvelope performance = 2;

  // Quality validation/regression of the optimized serving artifact relative
  // to CanonicalModel.quality_evaluation.
  QualityValidation quality = 3;

  // Immutable consolidated qualification report for this selected profile.
  // Example:
  //   bundle_path="reports/optimization/h200-vllm-fp8-tp4/qualification.pb".
  ArtifactRef qualification_report = 4;

  // Optional supporting reports such as profiler summaries or stability runs.
  repeated ArtifactRef supporting_reports = 5;

  // Exact immutable traffic distribution under which this profile was
  // optimized/qualified.
  //
  // Initial optimization:
  //   source=EXPECTED, supplied from Product ServingRequirements.
  //
  // Later reoptimization:
  //   source=OBSERVED, snapshot derived from real serving telemetry.
  TrafficDistributionSnapshot traffic_distribution = 6;
}

message PerformanceEnvelope {
  // Immutable benchmark execution identifier.
  // Example: "bench-run-183772".
  string benchmark_run_id = 1;

  // Exact load-generator / benchmark workload derived from
  // QualificationEvidence.traffic_distribution.
  //
  // The traffic distribution describes the statistical workload assumption;
  // this artifact describes the concrete benchmark program/replay used to
  // measure the profile.
  //
  // Example: bundle_path="reports/optimization/workload-prod-chat-v4.pb".
  ArtifactRef benchmark_workload = 2;

  // Hardware fingerprint actually used during benchmark.
  // Example: "8xH100-SXM|NVSwitch|driver-575.57".
  string hardware_fingerprint = 3;

  // Runtime/container fingerprint.
  // Example: OCI image digest + runtime version hash.
  string runtime_fingerprint = 4;

  // p99 time to first token in milliseconds.
  // Example: 285.4.
  optional double ttft_p99_ms = 5;

  // p99 time per output token in milliseconds.
  // Example: 31.7.
  optional double tpot_p99_ms = 6;

  // Maximum sustained request rate while all declared SLOs hold.
  // Example: 18.4 requests/sec.
  optional double sustainable_qps = 7;

  // Aggregate output-token throughput at the qualified operating point.
  // Example: 2860 tokens/sec.
  optional double output_tokens_per_sec = 8;

  // Peak GPU-memory use per accelerator at qualified workload.
  // Example: 132.4 GiB.
  optional double peak_gpu_memory_gib = 9;

  // Measured cold-start-to-ready duration.
  // Example: 95 seconds.
  optional google.protobuf.Duration startup_to_ready = 10;

  // Additional benchmark metrics without changing schema for every metric.
  repeated MetricResult additional_metrics = 11;
}

message QualityValidation {
  // Evaluation run performed on the actual optimized ServingProfile artifact.
  // Example: "profile-eval-9821".
  string evaluation_run_id = 1;

  // Canonical model evaluation used as the regression baseline.
  // Must equal CanonicalModel.quality_evaluation.evaluation_run_id.
  // Example: "canonical-eval-553".
  string baseline_evaluation_run_id = 2;

  // Policy describing acceptable profile-level quality regressions.
  // Example: "prod-llm-quality-policy-v7".
  string validation_policy_id = 3;

  // Overall pass/fail for this ServingProfile.
  // A profile cannot appear in the published preference set unless true.
  bool passed = 4;

  // Quality metrics measured on the actual optimized serving artifact.
  repeated MetricResult metrics = 5;

  // Explicit candidate-vs-canonical comparisons.
  // Example: MMLU 0.791 -> 0.789, delta=-0.002, PASS.
  repeated RegressionCheck regression_checks = 6;

  // Optional detailed quality report for this profile.
  optional ArtifactRef report = 7;
}



message MetricResult {
  // Stable metric name.
  // Example: "mmlu_accuracy", "ttft_p99_ms".
  string name = 1;

  // Numeric value.
  // Example: 0.791.
  double value = 2;

  // Unit or scale.
  // Example: "ratio", "ms", "tokens/sec".
  string unit = 3;

  // Optional threshold enforced by the relevant policy.
  // Example: 0.78.
  optional double threshold = 4;

  // Whether this metric passed its configured requirement.
  optional bool passed = 5;

  // Optional semantic slice/dimension labels.
  //
  // Examples:
  //   {"capability":"code", "dataset":"humaneval"}
  //   {"capability":"tool_calling"}
  //   {"language":"fr"}
  //   {"traffic_class":"long_context"}
  map<string, string> dimensions = 6;
}

message RegressionCheck {
  // Metric being compared.
  // Example: "mmlu_accuracy".
  string metric_name = 1;

  // Canonical/reference value.
  // Example: 0.791.
  double baseline_value = 2;

  // Optimized profile value.
  // Example: 0.788.
  double candidate_value = 3;

  // Candidate - baseline.
  // Example: -0.003.
  double delta = 4;

  // Allowed regression magnitude from policy.
  // Example: 0.005.
  double allowed_regression = 5;

  // True when candidate remains inside allowed quality envelope.
  bool passed = 6;
}

// ============================================================================
// STARTUP + OPTIMIZATION PROVENANCE
// ============================================================================

message StartupRequirements {
  // Expected artifact download time under the benchmarked startup path.
  optional google.protobuf.Duration artifact_download = 1;

  // Expected model/engine load time after artifact is local.
  optional google.protobuf.Duration model_load = 2;

  // Runtime initialization before warmup.
  optional google.protobuf.Duration runtime_initialization = 3;

  // Expected profile-specific warmup duration.
  optional google.protobuf.Duration warmup = 4;

  // Ordered semantic startup requirements.
  // Example:
  //   ["MODEL_LOAD", "DISTRIBUTED_INIT", "CUDA_GRAPH_CAPTURE", "WARMUP"].
  repeated string required_steps = 5;
}

message OptimizationProvenance {
  // Model Optimization job that produced/qualified this profile.
  // Example: "opt-job-8172".
  string optimization_run_id = 1;

  // Immutable source bundle version used as input.
  // Example: "assistant-70b:41".
  string source_bundle_version = 2;

  // Toolchain versions used to build derived artifacts.
  // Example:
  //   {"llm-compressor":"0.7.1", "tensorrt-llm":"1.0.0"}.
  map<string, string> tool_versions = 3;

  // Build time.
  google.protobuf.Timestamp created_at = 4;

  // Service/user principal that produced the profile.
  // Example: "model-optimization-controller".
  string produced_by = 5;

  // Optional detailed build/experiment report.
  optional ArtifactRef report = 6;
}

// ============================================================================
// LINEAGE + APPROVAL + SUPPLY CHAIN
// ============================================================================

message ModelLineage {
  // Training execution that produced canonical weights.
  // Example: "train-run-901".
  string training_run_id = 1;

  // Immutable training-code revision.
  // Example: Git commit SHA.
  string training_code_revision = 2;

  // Immutable training configuration.
  // Example: bundle_path="reports/training/train_config.pb".
  ArtifactRef training_config = 3;

  // Datasets used by training/fine-tuning.
  repeated DatasetReference datasets = 4;
}

message DatasetReference {
  // Dataset registry/storage namespace.
  // Example: "internal-dataset-registry".
  string source = 1;

  // Dataset logical name.
  // Example: "assistant-sft-data".
  string name = 2;

  // Immutable dataset version.
  // Example: "2026-08-20".
  string version = 3;

  // Optional immutable content digest.
  optional string digest = 4;

  // Semantic role in training.
  // Example: "SFT", "RLHF", "CALIBRATION".
  string role = 5;
}

message Approval {
  // Approval category.
  // Example: APPROVAL_TYPE_SAFETY.
  ApprovalType type = 1;

  // External approval/sign-off identifier.
  // Example: "safety-review-2026-8821".
  string approval_id = 2;

  // Principal/team that approved.
  // Example: "ai-safety-review".
  string approved_by = 3;

  // Approval timestamp.
  google.protobuf.Timestamp approved_at = 4;

  // Policy/version under which approval was granted.
  // Example: "genai-safety-policy-v9".
  string policy_id = 5;
}

message SupplyChainMetadata {
  // Software bill of materials for runtime/build dependencies.
  // Example: bundle_path="reports/attestations/sbom.spdx.json".
  optional ArtifactRef sbom = 1;

  // Signature/attestation over the published bundle manifest/artifact set.
  // Example: bundle_path="reports/attestations/bundle.sig".
  optional ArtifactRef bundle_attestation = 2;

  // Additional immutable attestations.
  // Example: provenance/SLSA-style build attestation.
  repeated ArtifactRef additional_attestations = 3;
}
```

---

# 6. Answers to the Earlier Ambiguities

## 6.1 Where is `CanonicalModel`?

It is now explicitly part of the root contract:

```proto
message BundleManifest {
    ...
    CanonicalModel canonical_model = 3;
    ...
}
```

There is no mismatch between the diagram and schema.

---

## 6.2 What is `InferenceType`?

It is now defined as a real enum describing **model semantics**, not serving runtime:

```text
TEXT_GENERATION
TEXT_EMBEDDING
RERANKING
CLASSIFICATION
REGRESSION
IMAGE_INFERENCE
MULTIMODAL_GENERATION
CUSTOM
```

Example:

```text
Llama-3.1-70B-Instruct
    inference_type = TEXT_GENERATION
```

It does **not** mean:

```text
vLLM vs TensorRT-LLM
```

That is `RuntimeKind` inside `ServingProfile`.

---

# 7. What Does `architecture` Mean?

The earlier field:

```text
string architecture
```

was too vague.

It is now:

```proto
message ModelArchitecture {
    string family;
    string implementation_class;
    string config_class;
    optional string architecture_revision;
    optional uint64 parameter_count;
}
```

Example:

```text
family:
    llama

implementation_class:
    transformers.LlamaForCausalLM

config_class:
    transformers.LlamaConfig

architecture_revision:
    llama-3.1

parameter_count:
    70553706496
```

The actual architectural hyperparameters:

```text
hidden_size
num_attention_heads
num_hidden_layers
rope_theta
num_key_value_heads
...
```

remain inside the immutable:

```text
canonical/model/config.json
```

We should **not duplicate every HuggingFace/config field into protobuf**, because that would create two sources of truth.

`ModelArchitecture` provides enough typed metadata for:

```text
search
compatibility
optimization candidate generation
validation
```

while `model_config` remains authoritative for detailed reconstruction.

---

# 8. What Does `base_model` Mean?

The earlier:

```text
string base_model
```

was also too vague.

It is now an optional typed `ModelReference`:

```text
source:
    huggingface

name:
    meta-llama/Llama-3.1-70B-Instruct

revision:
    immutable commit SHA

digest:
    optional content digest
```

For:

```text
fine-tuned model
```

this says exactly what parent was used.

For:

```text
model trained from scratch
```

the field is absent.

Training lineage still records the training run and datasets separately.

---

# 9. What Was `canonical_dtype`?

The term was imprecise.

It is replaced with:

```text
CanonicalModel.weight_dtype
```

Example:

```text
DATA_TYPE_BFLOAT16
```

This means:

> **How are the canonical exported model weights stored?**

It does not imply production serves BF16.

A profile may say:

```text
Canonical:
    BF16

ServingProfile:
    FP8 weights
    BF16 activations
    FP8 KV cache
```

That separation is intentional.

---

# 10. What Was `ModelSemanticConstraints`?

The previous type was undefined and too vague.

It is removed.

The real contract is now:

```text
ModelInterfaceContract
```

which explicitly contains:

```text
input/output modalities

architectural maximum context

streaming capability

chat template

input tensor schemas

output tensor schemas
```

Example for an LLM:

```text
input_modalities:
    TEXT

output_modalities:
    TEXT

max_context_tokens:
    131072

supports_streaming_output:
    true

chat_template:
    canonical/interface/chat_template.jinja
```

The distinction is important:

```text
ModelInterfaceContract.max_context_tokens
    =
architectural / semantic model limit

ServingProfile.tuning.max_model_len
    =
qualified deployment/runtime limit
```

For example:

```text
model supports 128K
```

but optimization may determine:

```text
production profile supports 32K
```

to meet cost/SLO targets.

---

# 11. `ModelArtifactRef` vs `ArtifactRef`

There is no need for two types.

That was an inconsistency in the previous draft.

The production schema uses exactly one:

```text
ArtifactRef
```

for:

```text
weights
tokenizer
config
TorchScript modules
quantized checkpoints
TRT engines
build reports
benchmark workload
SBOM
attestations
```

It supports either:

```text
bundle-relative artifact
```

or:

```text
external immutable URI
```

plus:

```text
sha256
size
file/directory kind
media type
```

Example:

```text
ArtifactRef {
    bundle_path:
        "canonical/model/"

    sha256:
        "3c7d..."

    size_bytes:
        140000000000

    kind:
        DIRECTORY

    media_type:
        "application/x-safetensors-directory"
}
```

---

# 12. Why the Manifest Does Not Contain `bundle_digest`

If:

```text
bundle_manifest.pb
```

contains:

```text
bundle_digest = SHA256(bundle_manifest.pb)
```

then calculating the digest changes the file whose digest we are calculating.

That is self-referential.

Instead:

```text
Model Registry {
    model_name
    bundle_version
    artifact_uri
    manifest_sha256
}
```

stores the digest externally.

Inside the manifest, every referenced artifact has its own digest.

A signed bundle attestation can cover:

```text
manifest digest
+
artifact digests
```

---

# 13. Canonical Quality vs ServingProfile Qualification

The bundle now has two deliberately different empirical layers.

```text
CanonicalModel.quality_evaluation
    =
    How good/capable is this model revision semantically?

ServingProfile.qualification
    =
    How does this exact serving realization behave empirically?
```

Canonical quality is independent of the selected hardware profile:

```text
MMLU
HumanEval
tool-call success
reasoning quality
language/domain quality slices
internal task evaluations
```

ServingProfile qualification is specific to:

```text
runtime
container/environment
GPU SKU
parallelism
precision/quantization
runtime tuning
benchmark workload
```

and contains:

```text
TTFT / TPOT
sustainable QPS
throughput
HBM / KV behavior
startup behavior
profile-level quality regression vs canonical baseline
```

For a bundle with:

```text
Primary:
    P0

Alternatives:
    P1
    P2
```

the bundle contains `QualificationEvidence` only for:

```text
P0
P1
P2
```

The complete rejected-candidate sweep remains in the Model Optimization
Platform's experiment store.

The qualification traffic source is also explicit:

```text
Initial optimization
    |
    +-- expected_traffic_distribution
            Product-team supplied
            source = EXPECTED

Production feedback / reoptimization
    |
    +-- observed_traffic_distribution
            Serving/Metrics telemetry derived
            source = OBSERVED
```

The important ownership rule is:

```text
Live observed traffic
    =
    Serving / Metrics Platform state
    NOT mutable Model Bundle state

Traffic snapshot used for a published profile
    =
    immutable QualificationEvidence inside that ServingProfile
```

Therefore an initial profile can say:

```text
P0.QualificationEvidence
    traffic_distribution:
        source = EXPECTED
        traffic_profile_id = expected-chat-v3
```

while a later reoptimized profile can say:

```text
P3.QualificationEvidence
    traffic_distribution:
        source = OBSERVED
        traffic_profile_id = observed-prod-window-17
        observation_window = [...]
```

This allows every empirical serving metric to answer:

> Under exactly which expected or observed workload distribution was this
> ServingProfile qualified?

---

# 14. Serving Profiles and Scheduler Alternatives

The bundle supports the preferred pattern:

```text
Primary profile
+
ordered alternative profiles
```

For the normal colocated case:

```text
ServingProfilePreference {
    role = COLOCATED

    primary =
        h200-vllm-fp8-tp4

    alternatives =
        h100-vllm-fp8-tp8
        h100-trtllm-fp8-tp8
        a100-vllm-awq-tp8
}
```

Serving Controller turns this into scheduler alternatives.

Example:

```text
Profile P0
    ->
4 x H200

Profile P1
    ->
8 x H100

Profile P2
    ->
8 x H100

Profile P3
    ->
8 x A100
```

Then:

```text
ResourceRequirements {
    primary_requirement:
        P0 resource shape

    alternative_requirements:
        P1
        P2
        P3
}
```

The selected requirement must preserve:

```text
selected_profile_id
```

so the controller knows which runtime/artifacts to start.

---

# 15. Why Preferences Are Grouped by `ServingRole`

A single flat:

```text
primary_profile_id
alternatives[]
```

works for ordinary colocated serving.

It becomes ambiguous for disaggregated P/D.

Therefore the schema supports:

```text
preference_groups:
    COLOCATED
```

or:

```text
preference_groups:
    PREFILL
    DECODE
```

Example:

```text
PREFILL:
    primary = h200-prefill-vllm-fp8
    alternatives = [...]

DECODE:
    primary = h200-decode-vllm-fp8
    alternatives = [...]
```

Profiles that must be composed together may carry:

```text
compatibility_key = "pd-layout-v3"
```

This does not force Serving Platform to use P/D; it merely prevents the bundle schema from assuming every serving topology is one homogeneous replica.

---

# 16. Runtime Environment Ownership

The environment belongs to `ServingProfile`, not `CanonicalModel`.

Example:

```text
ServingProfile:
    h100-trtllm-fp8-tp8

RuntimeEnvironment:
    runtime_kind:
        TENSORRT_LLM

    runtime_version:
        1.x

    container_image_digest:
        registry/...@sha256:abcd...

    cuda_runtime_version:
        12.8

    framework_version:
        ...

    library_versions:
        ...
```

Therefore deployment never has to guess:

```text
Which vLLM version?
Which CUDA?
Which compiler runtime?
Which plugin package?
Which container?
```

The qualified profile specifies it.

---

# 17. Canonical Checkpoint Sharding

Canonical model weights are independent of serving TP.

Do not encode:

```text
canonical/rank0.safetensors
canonical/rank1.safetensors
```

merely because serving TP=2.

Training may have used:

```text
FSDP
ZeRO
TP
PP
```

and serving may choose a completely different topology.

Instead:

```text
Training checkpoint
       |
       v
Canonical export
       |
       v
canonical/model/
```

Then Model Optimization derives:

```text
TP4 layout
TP8 layout
TRT-LLM rank files
FP8 checkpoint
AWQ checkpoint
compiled engine
```

inside specific profiles.

---

# 18. Who Writes What

## Training / Canonical Export

Writes:

```text
BundleIdentity

CanonicalModel:
    architecture
    base model
    canonical weight dtype
    canonical weights
    config
    tokenizer
    interface contract
    initial declared capability contract
    feature contracts

ModelLineage

Auxiliary models
```

Training/export does **not** decide:

```text
H200
TP4
FP8
vLLM
TRT-LLM
```

---

## Evaluation / Quality / Safety

Writes or validates in the mutable build workspace before publication:

```text
CanonicalModel.capability_contract
CanonicalModel.quality_evaluation
approvals[]
```

The capability contract is deterministic model-level eligibility metadata.
The quality evaluation is empirical model-level quality and is independent of
which ServingProfile is ultimately selected.

---

## Model Optimization Platform

Consumes for the initial optimization run:

```text
CanonicalModel
ServingRequirements
expected_traffic_distribution
```

For reoptimization it may instead consume:

```text
CanonicalModel
ServingRequirements
observed_traffic_distribution snapshot
```

Writes:

```text
ServingProfileCatalog

derived artifacts

runtime environments

hardware requirements

parallelism

precision

runtime tuning

QualificationEvidence for each selected/published ServingProfile:
    exact traffic distribution snapshot used for qualification
        EXPECTED for initial optimization
        OBSERVED for traffic-driven reoptimization
    performance envelope
    profile quality regression vs canonical QualityEvaluation
    selected experiment/report references

optimization provenance
```

The full rejected/swept experiment corpus remains in the Model Optimization
Platform and is not published as ServingProfiles in the bundle.
```

---

## Bundle Publisher

Performs:

```text
validate schema
validate profile references
validate artifact existence
verify artifact digests
finalize bundle_manifest.pb
publish immutable directory
```

---

## Model Registry

Stores externally:

```text
model name
bundle version
artifact URI
manifest SHA256
lifecycle state
aliases
index/search metadata
```

Registry state is not rewritten into the immutable bundle.

---

## Serving / Metrics Platform

Owns the continuously changing production observation:

```text
observed_traffic_distribution
observed TTFT / TPOT / queueing / throughput
current deployment/runtime metrics
```

This live telemetry does not mutate an existing Model Bundle.

When reoptimization is triggered, the Serving Platform supplies an immutable
observed-traffic snapshot to a new OptimizationJob. If that run publishes new
ServingProfiles, the snapshot becomes part of those profiles'
`QualificationEvidence`.

---

## Serving Platform

Reads:

```text
BundleManifest

selected profile

runtime environment

resource requirements

artifact selection

startup requirements
```

---

# 19. Alias Resolution

Serving replicas must not independently resolve mutable aliases.

Preferred:

```text
User:
    deploy assistant@champion

Serving Platform API:
    champion
      ->
    bundle version 42
      ->
    manifest SHA abc123

ServingDeployment:
    bundle_ref =
        assistant-70b:42@sha256:abc123
```

Now all replicas in one deployment use exactly the same immutable bundle.

---

# 20. Validation Rules

A production Bundle Validator should enforce at least:

## Root

```text
schema_version supported

identity.model_name non-empty

canonical_model present
```

## Canonical model

```text
weights artifact exists and digest matches

model_config exists

tokenizer required for text inference types

weight_dtype specified

architecture.family specified

interface contract consistent

capability contract structurally valid

canonical quality evaluation present when publication policy requires it
```

## Artifact references

```text
exactly one of bundle_path / uri is set

sha256 present

kind present

bundle_path cannot escape bundle root
```

## Serving profile

```text
profile_id unique

role specified

runtime/container image specified

hardware accelerator/count specified

TP/PP/DP all non-zero

artifacts exist

qualification.quality.passed == true

qualification.performance benchmark evidence present

qualification.quality.baseline_evaluation_run_id
    == CanonicalModel.quality_evaluation.evaluation_run_id

qualification.traffic_distribution.source specified

qualification.traffic_distribution.distribution artifact exists

if traffic source == OBSERVED:
    observation window present

PerformanceEnvelope.benchmark_workload must correspond to the
QualificationEvidence.traffic_distribution used by the optimization experiment
```

## Preference groups

```text
one preference group per role

primary profile exists

every alternative exists

all referenced profiles have same role as preference group

no duplicate alternative IDs

primary not repeated in alternatives
```

## P/D compatibility

When Serving Platform chooses P/D:

```text
chosen PREFILL + DECODE profiles
must have compatible compatibility_key
```

## Integrity

```text
all bundle artifact digests verify

attestation verifies if policy requires it
```

---

# 21. Core Invariants

1. `bundle_manifest.pb` is the only authoritative metadata root.
2. `BundleManifest.canonical_model` always exists.
3. Every schema type referenced by the root is explicitly defined.
4. `InferenceType` describes model semantics, not serving engine.
5. `ModelArchitecture` is typed metadata; detailed hyperparameters remain in the canonical config artifact.
6. `base_model` is a typed immutable `ModelReference`, not a free-form string.
7. Canonical weight dtype is called `weight_dtype`; serving precision belongs to the profile.
8. `ModelInterfaceContract` describes technical I/O semantics; `CapabilityContract` describes deterministic model-level semantic eligibility.
9. `CanonicalModel.quality_evaluation` contains empirical semantic quality for the model revision and is independent of the selected serving hardware/runtime profile.
10. Canonical capability and quality are not duplicated at BundleManifest root.
11. There is one generic `ArtifactRef`; `ModelArtifactRef` is removed.
12. Canonical weights are independent of training or serving shard topology.
13. Published bundle versions are immutable.
14. Model Registry aliases are mutable but resolve to immutable bundle version + manifest digest before deployment.
15. Serving runtime environment belongs to a `ServingProfile`.
16. A ServingProfile binds runtime + hardware + parallelism + precision + artifacts + tuning + empirical `QualificationEvidence`.
17. The bundle contains profile-specific empirical results only for published ServingProfiles (primary + alternatives), not rejected optimization sweep candidates.
18. `QualificationEvidence.performance` is valid only for its exact runtime/hardware/environment/workload realization.
19. `QualificationEvidence.quality` must explicitly reference the canonical quality evaluation used as its regression baseline.
20. Primary/alternative selection is explicit and role-aware.
21. Scheduler may select only a profile already qualified and declared by Serving Platform.
22. Scheduler selection must preserve `selected_profile_id`.
23. Profile fallback occurs before runtime materialization, not as a silent runtime fallback.
24. Initial Model Optimization qualifies profiles against a Product-supplied `expected_traffic_distribution`.
25. Continuously changing `observed_traffic_distribution` belongs to Serving/Metrics state, not the immutable Model Bundle.
26. When observed traffic triggers reoptimization, the exact immutable observed snapshot used by the new OptimizationJob is stored in each resulting selected ServingProfile's `QualificationEvidence`.
27. Every profile-level serving metric is interpretable only together with its exact traffic-distribution snapshot and benchmark workload.
28. Every artifact is digest-verifiable.

---

# 22. Concrete Example

```text
Bundle:
    assistant-70b
    version 42

CanonicalModel:
    inference_type:
        TEXT_GENERATION

    architecture:
        family:
            llama

        implementation_class:
            transformers.LlamaForCausalLM

        config_class:
            transformers.LlamaConfig

        architecture_revision:
            llama-3.1

        parameter_count:
            ~70B

    base_model:
        source:
            huggingface

        name:
            meta-llama/Llama-3.1-70B-Instruct

        revision:
            immutable commit SHA

    weight_dtype:
        BFLOAT16

    weights:
        canonical/model/

    model_config:
        canonical/model/config.json

    tokenizer:
        canonical/tokenizer/

    interface:
        input = TEXT
        output = TEXT
        max_context = 128K
        streaming = true

    capability_contract:
        operations:
            CHAT_COMPLETION
            TEXT_GENERATION

        tools:
            supports_tool_calling = true
            supports_parallel_tool_calls = true

        structured_output:
            supports_json_object = true
            supports_json_schema = true

        supported_languages:
            en
            es
            fr
            ja

    quality_evaluation:
        evaluation_run_id:
            canonical-eval-553

        MMLU:
            0.791

        HumanEval pass@1:
            0.742

        tool_call_success:
            0.913
```

Qualified profiles:

```text
COLOCATED preference:

Primary:
    h200-vllm-fp8-tp4

    runtime:
        vLLM
        pinned image digest

    hardware:
        4 x H200 SXM

    parallelism:
        TP4

    precision:
        FP8 weights
        BF16 activations

    artifact:
        FP8 safetensors

    qualification:
        experiment_id:
            opt-exp-8172-c17

        traffic_distribution:
            source:
                EXPECTED

            traffic_profile_id:
                expected-chat-v3

            produced_by:
                assistant-product-team

        performance:
            TTFT p99:
                285 ms

            TPOT p99:
                31.7 ms

            sustainable QPS:
                18.4

        quality:
            baseline:
                canonical-eval-553

            MMLU:
                canonical = 0.791
                profile = 0.789
                delta = -0.002
                PASS

Alternative 1:
    h100-vllm-fp8-tp8

Alternative 2:
    h100-trtllm-fp8-tp8

Alternative 3:
    a100-vllm-awq-tp8
```

Deployment path:

```text
Model Registry Alias
    |
    v
immutable BundleManifest
    |
    v
ServingProfilePreference
    |
    v
Serving Controller
    |
    v
ResourceRequirements {
    primary
    alternatives
}
    |
    v
GPU Fleet Scheduler
    |
    v
selected_profile_id + ResourceAllocation
    |
    v
Serving Controller loads exact matching
runtime + artifacts
```

---

# 23. Principal-Staff Interview Summary

> **“The Model Bundle is the immutable contract across training, optimization, registry, and serving. Its root manifest has one canonical model definition and a catalog of qualified serving profiles. Canonical model metadata contains intrinsic model truth—architecture identity, parent model, canonical weight dtype, weights/config/tokenizer, interface semantics, a deterministic CapabilityContract, and canonical empirical QualityEvaluation. Runtime-specific decisions live in ServingProfile, which binds an exact runtime image, hardware resource shape, parallelism, precision, optimized artifacts, tuning, and profile-specific QualificationEvidence containing measured serving performance plus quality regression against the canonical baseline. Profiles are organized as a primary plus ordered alternatives per serving role. Each selected profile's QualificationEvidence also records the immutable traffic distribution under which it was qualified—Product-supplied expected traffic for the initial optimization run, or an observed production snapshot for a later reoptimization run. Live observed traffic remains Serving/Metrics state and never mutates an existing bundle. The Serving Controller translates qualified profiles into scheduler resource alternatives, and the selected allocation carries the selected profile identity back so deployment starts the exact qualified runtime. Published bundles are immutable; Registry aliases are the mutable layer.”**
