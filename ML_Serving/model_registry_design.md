# Model Registry — System Design

The Model Registry is the central artifact store for the post-training pipeline.
Every step that produces or modifies a model artifact writes to it. Every
component that consumes a model reads from it.

---

## What It Actually Is

```
Model Registry
      │
      ├── Blob Store (HDFS / S3 / GCS)
      │     Stores model bundles as directory trees.
      │     Immutable once written — no overwriting a version's files.
      │     Content-addressed by path: registry/llama3-8b/v2/
      │
      └── Metadata Store (MySQL / PostgreSQL)
            Stores structured data about each bundle:
            model name, version, artifact path, status,
            eval scores, safety sign-off, tags, aliases.
            The registry IS this metadata layer.
            The blob store is just storage it points into.
```

The registry does not store the weights. It stores a **pointer** (artifact_path)
to where the bundle lives in blob storage, plus all the structured metadata
that describes that bundle.

---

## Data Model

```sql
-- One row per named model (the "family")
CREATE TABLE registered_models (
    model_name      VARCHAR(255) PRIMARY KEY,   -- e.g. "llama3-8b-instruct"
    description     TEXT,
    created_at      TIMESTAMP,
    created_by      VARCHAR(255),               -- team or user
    tags            JSON                        -- arbitrary key-value labels
);

-- One row per version of a model
CREATE TABLE model_versions (
    model_name      VARCHAR(255) REFERENCES registered_models(model_name),
    version         INT,                        -- auto-incremented per model_name
    artifact_path   VARCHAR(1024),              -- hdfs://registry/llama3-8b/v2/
    status          ENUM(
                        'pending',              -- conversion / optimization in progress
                        'staging',              -- passed eval + safety, in benchmarking
                        'production',           -- serving live traffic
                        'deprecated',           -- superseded, kept for rollback
                        'failed'                -- did not pass a gate
                    ),
    run_id          VARCHAR(255),               -- training run that produced base weights
    docker_image    VARCHAR(1024),              -- exact image used for benchmarking + serving
                                               -- e.g. "registry.linkedin.com/llm-serving:cuda12.1-torch2.3-vllm0.4.2"
    environment     JSON,                      -- snapshot of all package versions at registration time
                                               -- {python: "3.11.4", torch: "2.3.0", vllm: "0.4.2", cuda: "12.1"}
    created_at      TIMESTAMP,
    created_by      VARCHAR(255),
    PRIMARY KEY (model_name, version)
);

-- Flexible key-value metadata attached to a version
-- Each pipeline step writes its outputs here
CREATE TABLE model_version_metadata (
    model_name      VARCHAR(255),
    version         INT,
    key             VARCHAR(255),               -- e.g. "mmlu", "ttft_p99", "safety_sign_off"
    value           TEXT,                       -- scalar or JSON string for nested data
    written_by      VARCHAR(255),               -- which pipeline step wrote this
    written_at      TIMESTAMP,
    PRIMARY KEY (model_name, version, key),
    FOREIGN KEY (model_name, version) REFERENCES model_versions(model_name, version)
);

-- Mutable named pointers to specific versions
-- Serving code references aliases, not version numbers
CREATE TABLE model_aliases (
    model_name      VARCHAR(255),
    alias           VARCHAR(255),               -- e.g. "champion", "shadow", "canary"
    version         INT,                        -- which version this alias currently points to
    updated_at      TIMESTAMP,
    updated_by      VARCHAR(255),
    PRIMARY KEY (model_name, alias),
    FOREIGN KEY (model_name, version) REFERENCES model_versions(model_name, version)
);
```

---

## API Surface

### Registration — pipeline steps write here

```
POST /models
  Body: { model_name, description, created_by }
  Creates a new registered model (the family).
  Returns: { model_name }

POST /models/{model_name}/versions
  Body: {
    artifact_path,
    run_id,
    created_by,
    docker_image,    -- "registry.linkedin.com/llm-serving:cuda12.1-torch2.3-vllm0.4.2"
    environment: {   -- full package snapshot at benchmarking time
      python:       "3.11.4",
      torch:        "2.3.0",
      transformers: "4.41.0",
      vllm:         "0.4.2",
      cuda:         "12.1",
      cudnn:        "8.9.2"
    }
  }
  Registers a new version. Auto-assigns next version number.
  Initial status: "pending"
  Returns: { model_name, version, artifact_path, status }

PUT /models/{model_name}/versions/{version}/status
  Body: { status: "staging" | "production" | "deprecated" | "failed" }
  Transitions the version lifecycle status.
  Returns: { model_name, version, status }

POST /models/{model_name}/versions/{version}/metadata
  Body: { key, value, written_by }
  Attach a metadata entry to this version (eval score, safety sign-off, etc.)
  Can be called multiple times — one call per pipeline step output.
  Returns: { model_name, version, key, value }

PUT /models/{model_name}/aliases/{alias}
  Body: { version, updated_by }
  Point alias to a specific version. Overwrites previous alias target.
  This is how you promote: move @champion from v2 to v3.
  Returns: { model_name, alias, version }
```

### Resolution — serving system reads here

```
GET /models/{model_name}/versions/{version}
  Returns full version record: artifact_path, status, all metadata.

GET /models/{model_name}/aliases/{alias}
  Resolves alias → version → artifact_path.
  Returns: { model_name, alias, version, artifact_path, status, metadata }
  This is the primary call a serving system makes at startup.

GET /models/{model_name}/versions
  List all versions of a model, optionally filtered by status.
  Query params: ?status=production&limit=10
  Returns: [ { version, artifact_path, status, created_at }, ... ]

GET /models
  List all registered model families.
  Returns: [ { model_name, latest_version, production_alias_version }, ... ]
```

### Querying — auditing, dashboards

```
GET /models/{model_name}/versions/{version}/metadata
  All metadata entries for this version.
  Returns: { mmlu: 0.68, ttft_p99: 140, safety_sign_off: "approved", ... }

GET /models/{model_name}/versions/{version}/lineage
  Trace back to training run: run_id, dataset version, training config.
  Returns: { version, run_id, base_model, fine_tune_data, training_params }
```

---

## How the Serving System Uses It

```
Serving system starts up (e.g. vLLM server, or your Flyte Ray Job)
        │
        ▼
GET /models/llama3-8b-instruct/aliases/champion
        │
        Response:
        {
          "alias":         "champion",
          "version":       3,
          "artifact_path": "hdfs://registry/llama3-8b-instruct/v3/",
          "status":        "production",
          "docker_image":  "registry.linkedin.com/llm-serving:cuda12.1-torch2.3-vllm0.4.2",
          "environment": {
            "python":       "3.11.4",
            "torch":        "2.3.0",
            "vllm":         "0.4.2",
            "cuda":         "12.1"
          },
          "metadata": {
            "mmlu":          "0.677",
            "quantization":  "fp8",
            "tp_degree":     "2",
            "max_batch":     "32"
          }
        }
        │
        ▼
Download bundle from artifact_path (HDFS / S3 client)
  hdfs://registry/llama3-8b-instruct/v3/
    ├── model/rank0.safetensors
    ├── model/rank1.safetensors
    ├── tokenizer/
    ├── config.json
    ├── serving_config.json
    └── metadata.json          ← includes docker_image + environment snapshot
        │
        ▼
Read metadata.json → docker_image = "registry.linkedin.com/llm-serving:cuda12.1-torch2.3-vllm0.4.2"
Pull and run inside that exact image  ← same environment as benchmarking
Read config.json → inference_type = TEXT_GENERATION
Read serving_config.json → tp=2, max_batch=32, max_tokens=512
Load weights from model/ → start vLLM engine
```

**The serving code never hardcodes a version:**

```python
# Serving system startup
REGISTRY_URL = "https://model-registry.linkedin.com"
MODEL_NAME   = "llama3-8b-instruct"
ALIAS        = "champion"

resp = requests.get(f"{REGISTRY_URL}/models/{MODEL_NAME}/aliases/{ALIAS}")
bundle_path  = resp.json()["artifact_path"]
serving_cfg  = resp.json()["metadata"]

# Download bundle and start engine
download_bundle(bundle_path, local_path="/tmp/model_bundle")
start_vllm_engine(
    model_path=f"/tmp/model_bundle/model/",
    tp_degree=int(serving_cfg["tp_degree"]),
    max_batch=int(serving_cfg["max_batch"]),
)
```

To promote v3 to production and have all serving replicas pick it up:

```bash
# One API call — no serving code deploy needed
curl -X PUT https://model-registry.linkedin.com/models/llama3-8b-instruct/aliases/champion \
     -d '{ "version": 3, "updated_by": "ml-platform-oncall" }'

# Serving replicas read @champion on next restart / reload cycle
# Or: push a signal to serving fleet to hot-reload the alias
```

---

## How the Post-Training Pipeline Writes to It

Each step is a writer. The pipeline assembles the version record incrementally:

```
Step 1 — Evaluation
  POST /models/llama3-8b-instruct/versions
    { artifact_path: "hdfs://registry/llama3-8b/v3/", run_id: "run_abc" }
  → version 3 created, status=pending

  POST /models/llama3-8b-instruct/versions/3/metadata
    { key: "mmlu",          value: "0.68",  written_by: "eval-pipeline" }
    { key: "ttft_p99_base", value: "180ms", written_by: "eval-pipeline" }

Step 2 — Safety
  POST /models/llama3-8b-instruct/versions/3/metadata
    { key: "safety_sign_off", value: "approved-2026-06-27", written_by: "safety-team" }
  PUT /models/llama3-8b-instruct/versions/3/status
    { status: "staging" }

Step 3 — Format Conversion
  (bundle updated in blob store at same artifact_path)
  POST /models/llama3-8b-instruct/versions/3/metadata
    { key: "format", value: "trtllm_fp16", written_by: "conversion-pipeline" }

Step 4 — Optimization
  POST /models/llama3-8b-instruct/versions/3/metadata
    { key: "quantization",   value: "fp8",   written_by: "optimization-pipeline" }
    { key: "mmlu_quantized", value: "0.677", written_by: "optimization-pipeline" }
    { key: "ttft_p99_opt",   value: "140ms", written_by: "optimization-pipeline" }

Step 5 — Canary passes, promote to production
  PUT /models/llama3-8b-instruct/versions/3/status
    { status: "production" }
  PUT /models/llama3-8b-instruct/aliases/champion
    { version: 3, updated_by: "oncall" }
  PUT /models/llama3-8b-instruct/versions/2/status
    { status: "deprecated" }   ← kept warm for rollback window
```

---

## Key Design Decisions

**Versions are immutable, aliases are mutable.**
Once version 3 is registered with artifact_path X, that path never changes.
Aliases (@champion, @shadow) are the only thing that moves. This means
serving code always references an alias, never a version number.

**Blob store is append-only, no overwrites.**
Each version gets its own path: `registry/{model}/{version}/`.
Optimization step writes quantized weights to the same version path
(same version, updated files) OR creates a new version (cleaner audit trail).
LinkedIn's practice: new version for each meaningful artifact change.

**Metadata is additive, not a single JSON blob.**
Each pipeline step appends its own metadata keys independently.
This avoids write conflicts when steps run in parallel and allows
each step to own its slice of metadata cleanly.

**Soft deletes only.**
Versions are never deleted — only marked `deprecated`.
Required for rollback: if v3 has a production incident, flip @champion
back to v2 (which is deprecated but still has its bundle intact).
Decommission (actual deletion from blob store) happens only after
a long retention window (30–90 days).

**Environment snapshot + Docker image are first-class fields, not optional metadata.**
Benchmarking numbers are only meaningful in the context of the environment they
were measured in. A QPS of 1200 tok/s measured on Python 3.10 + vLLM 0.3.0 is
not the same as 1200 tok/s on Python 3.11 + vLLM 0.4.2. If the serving system
runs in a different environment, the numbers are not reproducible and production
behavior will silently diverge from benchmarking results.

The fix: store the exact Docker image URI at registration time and mandate that
the serving system pulls and runs inside that image. The image pins every
dependency transitively — Python version, CUDA version, torch, vLLM, cuDNN,
all library versions — in one immutable artifact.

Real incident: benchmarking QPS did not match production QPS. Root cause was
a Python minor version difference (3.10 vs 3.11) between the benchmarking
environment and the production serving fleet. The asyncio event loop scheduler
behavior changed between versions, affecting how vLLM's async request handling
batched concurrent requests. The serving system was running a slower path
without anyone realizing it. Resolved by pinning docker_image in the bundle
and enforcing that production pulls the same image benchmarking ran in.
