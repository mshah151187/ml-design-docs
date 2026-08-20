# Model Bundle

A Model Bundle is a standardized directory structure that every component in the
ML pipeline understands. It is the shared contract between the post-training
pipeline, the Model Registry, and the serving system.

Every component — format conversion, optimization, the registry client, the
serving fleet, the compliance emitter, the observability SDK — reads from and
writes to this structure using the same directory layout and protobuf schemas.
No component needs to know how another component works. It only needs to know
the bundle.

---

## Why a Rigid Structure

Without a bundle contract, each component negotiates its own interface:

```
Format conversion step → passes weights path to optimization step via config
Optimization step      → passes quantized path to registry via env variable
Registry               → stores path string, serving system reads it differently
Serving system         → hardcodes expected file names, breaks on next model
```

Every handoff is a bespoke agreement. Each one can break independently.

With a bundle contract, the interface is the directory:

```
Any component that produces a bundle writes to the same structure.
Any component that consumes a bundle reads from the same structure.
Components are decoupled — you can swap out any step without touching others.
```

---

## Directory Structure

The `inference_model/` directory is partitioned into format-specific
subdirectories. Each pipeline step adds a new subdirectory when it produces a
new artifact format. Previous formats are preserved — the serving system reads
`serving_config.artifact_format` to know which subdirectory is active.

```
model_bundle/
  │
  ├── bundle_metadata.pb              ← SINGLE entry point for all metadata
  │                                     contains ModelMetadata for all three
  │                                     sub-models as embedded fields
  │
  ├── inference_model/
  │     │
  │     ├── model_metadata.pb         ← ModelMetadata for the inference model
  │     │                               same data as BundleMetadata.inference_model
  │     │                               allows components to read just this sub-model
  │     │
  │     ├── original/                 ← written by modeler at bundle creation
  │     │     ├── rank0.safetensors   PyTorch weights, training output
  │     │     └── rank1.safetensors   (if TP > 1)
  │     │
  │     ├── trtllm_checkpoint/        ← written by format conversion step (TRT-LLM path)
  │     │     ├── rank0.safetensors   weights restructured into TRT-LLM QKV-fused layout
  │     │     ├── rank1.safetensors   TP-sharded, one file per GPU rank
  │     │     └── config.json         TRT-LLM architecture config (required by trtllm-build)
  │     │
  │     ├── trtllm_engine/            ← written by optimization step (trtllm-build output)
  │     │     ├── rank0.engine        compiled binary — kernels fused, hardware-locked
  │     │     └── rank1.engine        one engine file per GPU rank
  │     │
  │     ├── onnx/                     ← written by format conversion step (ONNX path)
  │     │     └── model.onnx          compute graph + weights, hardware-agnostic
  │     │
  │     └── tokenizer/                ← written by modeler, never modified
  │           ├── tokenizer.json
  │           ├── tokenizer_config.json
  │           ├── vocab.json
  │           ├── special_tokens_map.json
  │           └── merges.txt
  │
  ├── preprocessing_model/            ← optional; written by team at onboarding
  │     ├── model_metadata.pb         ModelMetadata for the preprocessing model
  │     └── model.pt                  serialized TorchScript module
  │
  └── postprocessing_model/           ← optional; written by team at onboarding
        ├── model_metadata.pb         ModelMetadata for the postprocessing model
        └── model.pt                  serialized TorchScript module
```

Not all subdirectories under `inference_model/` are present in every bundle.
Which ones exist depends on which pipeline path was taken:

```
vLLM path:        original/  +  tokenizer/   (trtllm_* and onnx/ absent)
TRT-LLM path:     original/  +  trtllm_checkpoint/  +  trtllm_engine/  +  tokenizer/
ONNX path:        original/  +  onnx/  +  tokenizer/
```

`original/` is always present and never overwritten — it is the source of truth
for rebuilding any other format. `trtllm_checkpoint/` is kept even after the
engine is compiled because rebuilding the engine (for a different GPU or
different max_batch_size) reads from it, not from `original/`.

`bundle_metadata.pb` is the single access point. Any component that needs
metadata about the bundle or any of its sub-models reads this one file.
The sub-model directories contain only the artifact files — weights, tokenizer,
serialized modules. No separate proto files per subdirectory.

Three things worth noting about this design:

**Two-level access pattern.** `bundle_metadata.pb` at the root is the full picture
in one read — bundle-level config (docker_image, inference_type) plus all three
`ModelMetadata` embedded as nested fields. This is what the serving system reads
at startup: one file, full picture.

Each sub-model directory also contains its own `model_metadata.pb` with the same
data as the corresponding embedded field. This matters for components that only
care about one sub-model — a feature validation tool that only needs
`input_features`, or a format conversion step that only needs `serving_config`,
can read `inference_model/model_metadata.pb` directly without parsing the full
bundle proto. The two are kept in sync: any pipeline step that updates
`BundleMetadata.inference_model` also rewrites `inference_model/model_metadata.pb`.

**Sub-models are first-class citizens.** Preprocessing and postprocessing are not
scripts or callables — they are PyTorch modules serialized with TorchScript.
They can be loaded by any PyTorch runtime without the original Python class, have
proper type information, and are tracked with their own `ModelMetadata` including
version and training run ID.

**Protobuf, not JSON.** Schema is enforced at compile time. Breaking changes are
caught early. Forward and backward compatibility is built in via field numbering.

---

## Protobuf Schemas

Two proto definitions. `ModelMetadata` is defined first because `BundleMetadata`
embeds three instances of it. `bundle_metadata.pb` is the single compiled binary
at the root of every bundle — one read gives you everything.

### `model_metadata.proto`

```proto
syntax = "proto3";

enum ModelType {
  INFERENCE       = 0;
  PREPROCESSING   = 1;
  POSTPROCESSING  = 2;
}

enum QuantizationType {
  NONE  = 0;
  FP8   = 1;
  INT8  = 2;
  INT4  = 3;
}

// Tells the serving system which subdirectory under inference_model/ to load from.
// Set by the pipeline step that produces each artifact; serving system reads this
// once at startup and knows exactly which files to load.
enum ArtifactFormat {
  SAFETENSORS       = 0;   // inference_model/original/rank*.safetensors  (default after training)
  TRTLLM_CHECKPOINT = 1;   // inference_model/trtllm_checkpoint/          (after convert_checkpoint.py)
  TRTLLM_ENGINE     = 2;   // inference_model/trtllm_engine/rank*.engine  (after trtllm-build)
  ONNX              = 3;   // inference_model/onnx/model.onnx             (after torch.onnx.export)
}

message ServingConfig {
  string         engine                 = 1;   // "vllm" | "sglang" | "tensorrt_llm"
  ArtifactFormat artifact_format        = 2;   // which subdirectory to load from
  int32          tensor_parallel_size   = 3;
  int32          pipeline_parallel_size = 4;
  int32          max_batch_size         = 5;
  int32          max_model_len          = 6;
  int32          max_num_tokens         = 7;
  float          gpu_memory_utilization = 8;
  bool           enable_prefix_caching  = 9;
  string         dtype                  = 10;  // "float16" | "bfloat16"
}

// Describes one feature tensor — its identity in the feature store
// and its tensor contract (shape, dtype, valid range).
message FeatureSchema {
  // Identity — how to locate this feature in the feature serving system
  string namespace            = 1;   // team/product scope: "feed", "ads", "jobs"
                                     // Feed and Ads can both have "member_embedding"
                                     // namespaced as feed.member_embedding vs
                                     // ads.member_embedding — no collision
  string feature_key          = 2;   // lookup key: "member_embedding_v3"
  string feature_store_name   = 3;   // which store: "venice", "pinnacle"
  string feature_service_name = 4;   // which serving endpoint to call:
                                     // "feed-feature-service.prod.linkedin.com"

  // Tensor contract — what the serving system expects to receive back
  repeated int64 shape        = 5;   // e.g. [768] for embedding, [1] for scalar,
                                     // [128, 64] for 2D feature matrix
  string dtype                = 6;   // "float32", "int64", "bool", "float16"

  // Optional validation bounds — serving system can assert these before
  // passing the tensor to the model; catches feature store drift early
  optional float min_value    = 7;
  optional float max_value    = 8;
}

message ModelMetadata {
  string           model_name       = 1;   // "llama3-8b-instruct"
  string           model_version    = 2;   // "3"
  string           base_model       = 3;   // "meta-llama/Llama-3-8B-Instruct"
  string           fine_tune_data   = 4;   // HDFS path to fine-tune dataset
  string           training_run_id  = 5;
  ModelType        model_type       = 6;
  string           torch_dtype      = 7;   // "bfloat16"
  QuantizationType quantization     = 8;
  float            quant_score_drop = 9;   // MMLU drop from quantization (0.003)
  ServingConfig    serving_config   = 10;  // populated only for INFERENCE type

  // Feature contracts — used by the serving system to fetch and validate
  // features from the Feature Serving system before running inference.
  // Each entry is one tensor the model reads (input) or produces (output).
  repeated FeatureSchema input_features  = 11;
  repeated FeatureSchema output_features = 12;
}
```

**`input_features`** — what the serving system must fetch before calling the model.
At request time, for each `FeatureSchema` in `input_features`, the serving system
calls `feature_service_name` with `namespace + feature_key + entity_id`, receives
a tensor, validates its `shape` and `dtype` against the schema, optionally checks
`[min_value, max_value]`, then assembles all tensors into the model's input.

**`output_features`** — what the model produces. Downstream consumers (rankers,
other models, logging pipelines) read `output_features` to know what tensors to
expect and how to interpret them — shape, dtype, and valid range.

**`namespace`** disambiguates features across teams. Feed and Ads may both define
a feature named `member_embedding` — stored in the same feature store under
different namespaces (`feed.member_embedding` vs `ads.member_embedding`). The
serving system always passes `namespace + feature_key` together as the lookup key.

`ServingConfig` and feature schemas are only populated in the `INFERENCE` type
`ModelMetadata`. Preprocessing and postprocessing carry `model_type`,
`model_version`, and `training_run_id` — enough to track and version them —
but no engine config or feature contracts.

---

### `bundle_metadata.proto`

```proto
syntax = "proto3";

import "model_metadata.proto";

enum InferenceType {
  TEXT_GENERATION    = 0;
  SENTENCE_EMBEDDING = 1;
  CUSTOM_EMBEDDING   = 2;
  IMAGE_INFERENCE    = 3;
}

message Environment {
  string python        = 1;
  string torch         = 2;
  string transformers  = 3;
  string vllm          = 4;
  string cuda          = 5;
  string cudnn         = 6;
}

message TeamMetadata {
  string team      = 1;   // "feed-ml"
  string use_case  = 2;   // "connection_suggestion"
  string owner     = 3;   // "mshah@linkedin.com"
}

message EvalScores {
  float  mmlu            = 1;
  float  task_f1         = 2;
  float  ttft_p99_ms     = 3;
  float  throughput_tps  = 4;
}

message BundleMetadata {
  // Bundle-level identity and routing
  string        bundle_name     = 1;   // "llama3-8b-instruct"
  string        version         = 2;   // "3"
  InferenceType inference_type  = 3;   // routes serving system to correct engine
  string        docker_image    = 4;   // mandatory: serving system pulls this first
  Environment   environment     = 5;   // full package snapshot at benchmarking time
  TeamMetadata  team_metadata   = 6;
  EvalScores    eval_scores     = 7;
  string        safety_sign_off = 8;   // "approved-2026-06-27"
  string        created_at      = 9;
  string        created_by      = 10;
  string        status          = 11;  // "staging" | "production" | "deprecated"

  // Sub-model metadata — all three embedded directly
  ModelMetadata inference_model      = 12;  // always present
  ModelMetadata preprocessing_model  = 13;  // empty if no preprocessing declared
  ModelMetadata postprocessing_model = 14;  // empty if no postprocessing declared
}
```

`BundleMetadata` is the single access point. Fields 12–14 embed `ModelMetadata`
directly — not as file paths, not as references, but as nested messages.

A component that needs the inference engine's TP degree reads:
```
bundle.inference_model.serving_config.tensor_parallel_size
```

A component that needs to check if preprocessing is present reads:
```
bundle.preprocessing_model.model_name != ""
```

No directory traversal. No secondary file reads. One proto load, full picture.

---

## Who Writes What

The bundle starts at training completion and is progressively enriched by each
pipeline stage. All metadata writes go to `bundle_metadata.pb` — each stage
deserializes, updates its fields, and writes back. Artifact files are additive:
each stage adds its own subdirectory, never overwriting a previous stage's output.

### Stage 0 — Modeler (bundle creation)

The modeler runs training, packages the result, and registers the initial bundle.
This is the only stage that creates `bundle_metadata.pb` from scratch.

```
Writes to bundle_metadata.pb (creates):
  bundle_name, version, inference_type, created_at, created_by
  team_metadata: { team, use_case, owner }
  inference_model:
    model_name, model_version, base_model, fine_tune_data, training_run_id
    torch_dtype: "bfloat16"
    input_features, output_features  ← feature contracts defined here
    serving_config:                  ← partially populated
      engine: "vllm"                 ← tentative; confirmed in eval
      artifact_format: SAFETENSORS   ← default at creation
      tensor_parallel_size: 2        ← from training config
      (max_batch_size, max_model_len left empty — benchmarked later)
  preprocessing_model:  (empty if no preprocessing needed)
  postprocessing_model: (empty if no postprocessing needed)

docker_image: ""        ← NOT set yet; set after benchmarking environment locked
environment: {}         ← NOT set yet; set after benchmarking environment locked

Writes artifact files:
  inference_model/original/rank0.safetensors   training weights
  inference_model/original/rank1.safetensors   (if TP > 1)
  inference_model/tokenizer/                   tokenizer files from HF hub

Writes pre/post TorchScript (if team declares them):
  preprocessing_model/model.pt
  postprocessing_model/model.pt
  ↳ bundle_metadata.pb updated with preprocessing_model/postprocessing_model fields
```

### Stage 1 — Evaluation

Loads the safetensors weights with vLLM, runs benchmarking and quality evals.
This stage locks the environment: the docker image used here must match production.

```
Updates bundle_metadata.pb:
  eval_scores:
    mmlu: 0.68
    task_f1: 0.74
    ttft_p99_ms: 180.0      ← baseline TTFT before optimization
    throughput_tps: 420.0   ← baseline tokens/sec
  docker_image: "registry.linkedin.com/llm-serving:cuda12.1-torch2.3-vllm0.4.2"
  environment: { python: "3.11.4", torch: "2.3.0", vllm: "0.4.2", cuda: "12.1" }
  inference_model.serving_config:
    engine: "vllm"          ← confirmed (or changed to "tensorrt_llm" if TRT path chosen)
    max_batch_size: 32      ← determined from benchmarking
    max_model_len: 4096

Writes no artifact files — reads from inference_model/original/
```

After evaluation, a decision is made on the serving path:
- **vLLM path**: original/ weights are sufficient; skip to Canary
- **TRT-LLM path**: proceed to Format Conversion → Optimization
- **ONNX path**: proceed to Format Conversion only (no trtllm-build needed)

### Stage 2 — Format Conversion

Runs only if TRT-LLM or ONNX path was chosen. Reads from `original/`, writes a
new subdirectory. Does NOT modify weights — only restructures their layout.

**TRT-LLM path** (`convert_checkpoint.py`):
```
Writes artifact files:
  inference_model/trtllm_checkpoint/config.json       TRT-LLM architecture config
  inference_model/trtllm_checkpoint/rank0.safetensors QKV-fused, TP-sharded layout
  inference_model/trtllm_checkpoint/rank1.safetensors

Updates bundle_metadata.pb:
  inference_model.serving_config:
    engine: "tensorrt_llm"
    artifact_format: TRTLLM_CHECKPOINT   ← updated; trtllm-build reads this dir
```

**ONNX path** (`torch.onnx.export`):
```
Writes artifact files:
  inference_model/onnx/model.onnx    compute graph + weights as protobuf

Updates bundle_metadata.pb:
  inference_model.serving_config:
    engine: "onnxruntime"   (or "tensorrt" if ONNX → TRT core path)
    artifact_format: ONNX
```

**vLLM path**: no format conversion needed — skip this stage entirely.

### Stage 3 — Optimization

Runs only on TRT-LLM path. Reads from `trtllm_checkpoint/`, compiles the engine,
writes to `trtllm_engine/`. This is the most expensive stage (can take 30–90 min
for a 70B model on H100).

```
Writes artifact files:
  inference_model/trtllm_engine/rank0.engine   compiled binary, hardware-locked
  inference_model/trtllm_engine/rank1.engine   one per GPU rank

Updates bundle_metadata.pb:
  inference_model:
    quantization: FP8           (if quantization applied during trtllm-build)
    quant_score_drop: 0.003     (MMLU drop measured after quantization)
  inference_model.serving_config:
    artifact_format: TRTLLM_ENGINE   ← updated; serving system loads .engine files now
    max_batch_size: 32               ← finalized from trtllm-build flags
    max_model_len: 4096
```

After this stage, the bundle is complete. All three artifact subdirectories exist:
`original/` (always kept for rebuild), `trtllm_checkpoint/` (kept for engine
rebuild with different shapes), and `trtllm_engine/` (what production loads).

### Summary table

| Stage | Reads from | Adds to bundle |
|-------|-----------|----------------|
| Modeler | — (training output) | `original/`, `tokenizer/`, full `BundleMetadata` skeleton |
| Evaluation | `original/` | `eval_scores`, `docker_image`, `environment`, `serving_config` parameters |
| Format Conversion (TRT-LLM) | `original/` | `trtllm_checkpoint/`, `artifact_format = TRTLLM_CHECKPOINT` |
| Format Conversion (ONNX) | `original/` | `onnx/model.onnx`, `artifact_format = ONNX` |
| Optimization | `trtllm_checkpoint/` | `trtllm_engine/`, `artifact_format = TRTLLM_ENGINE`, `quantization` |

---

## Who Reads What

Every component reads `bundle_metadata.pb` first — one file, one read.
Sub-model directories are accessed only for the artifact files (weights, modules).

```
Model Registry client
  reads:   bundle_metadata.pb
             → bundle_name, version, status, team_metadata (for indexing)
             → inference_model.serving_config (for search/filter)
             → inference_model.quantization, eval_scores (for dashboards)
  stores:  entire directory → blob store at registry/{bundle_name}/{version}/

Serving system — startup
  reads:   bundle_metadata.pb
             → docker_image                        pull exact image before anything else
             → inference_type                      select which engine to initialize
             → inference_model.serving_config      configure TP, batch size, dtype
             → inference_model.input_features      register feature fetch plan
             → inference_model.output_features     register output schema for downstream
             → preprocessing_model.model_name      check if preprocessing present
             → postprocessing_model.model_name     check if postprocessing present
  reads:   inference_model/rank*.safetensors       load weights into GPU VRAM
  reads:   inference_model/tokenizer/              initialize tokenizer
  reads:   preprocessing_model/model.pt            load TorchScript module (if present)
  reads:   postprocessing_model/model.pt           load TorchScript module (if present)

Serving system — per request (online inference)
  for each FeatureSchema in inference_model.input_features:
    call feature_service_name
      with namespace + feature_key + entity_id from request
    receive tensor
    assert tensor.shape == schema.shape
    assert tensor.dtype == schema.dtype
    if schema.min_value set: assert tensor.min() >= schema.min_value
    if schema.max_value set: assert tensor.max() <= schema.max_value
  assemble validated tensors → model input → run inference
  map output tensors → output_features schema → return to caller

Compliance emission client
  reads:   bundle_metadata.pb
             → bundle_name, version, team_metadata, inference_model.training_run_id
             → populate Kafka compliance signal fields

Observability SDK
  reads:   bundle_metadata.pb
             → bundle_name, team_metadata
             → tag all emitted metrics with model + team identity
```

---

## Versioning in the Registry

Each bundle version lives at its own immutable path in blob storage:

```
hdfs://model-registry/
  llama3-8b-instruct/
    v1/
      bundle_metadata.pb
      inference_model/
      preprocessing_model/    ← present in v1
      postprocessing_model/
    v2/
      bundle_metadata.pb      ← docker_image updated, eval_scores updated
      inference_model/        ← FP8 quantized weights, updated model_metadata.pb
      preprocessing_model/    ← same module, new model_metadata.pb (version bump)
      postprocessing_model/
```

The Model Registry alias (`@champion`) points to a version directory.
Promoting to production = moving the alias, not copying files.
Each sub-model can be versioned independently within a bundle version if needed
— the `model_version` field in `model_metadata.pb` tracks this.

---

## The Contract Principle

The bundle is not a file format — it is a **contract**.

**Mandatory** (always present, serving system will not start without these):
- `bundle_metadata.pb` with `docker_image` and `inference_type` populated
- `inference_model/model_metadata.pb` (must match embedded field in `bundle_metadata.pb`)
- `inference_model/original/rank0.safetensors` (source of truth; always preserved)
- `inference_model/tokenizer/`

**Optional** (absent means feature is not used — not an error):
- `preprocessing_model/`
- `postprocessing_model/`

A bundle validation step runs after each pipeline stage. It deserializes the
proto files, checks all mandatory fields are populated, and verifies that
`rank0.safetensors` (or `.engine`) exists. Validation failure blocks the bundle
from advancing to the next pipeline stage or being registered in the Model Registry.
