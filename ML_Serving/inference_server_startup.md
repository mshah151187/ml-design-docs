# Inference Server Startup

When Kubernetes schedules a new serving pod, a sequence of one-time initialization
processes runs before the pod is allowed to receive any traffic. These processes
are sequential where required by dependency and parallel where possible.

This doc covers the full startup sequence, the parallelization opportunities within
it, and the techniques used in production today to reduce startup time.

---

## The Five Startup Processes

### Process 1 — Pod Initialized

Kubernetes schedules the pod onto a GPU node. The container starts using the exact
image from `bundle_metadata.docker_image` — the same image used during benchmarking,
guaranteeing the execution environment matches the environment in which `eval_scores`
were measured.

### Process 2 — Model Bundle Loader

Reads a **ConfigMap** mounted into the pod by the platform at deploy time:

```yaml
# ConfigMap created by platform when deploying the service
model_name:  "llama3-8b-instruct"
model_alias: "champion"
```

Uses the Model Registry Client to resolve alias → artifact path in blob store.
Downloads `bundle_metadata.pb` first (small proto file), then the full bundle:

```
/pod/model_bundle/
  bundle_metadata.pb
  inference_model/<artifact_format>/rank*.safetensors  (or rank*.engine)
  inference_model/tokenizer/
  preprocessing_model/model.pt    (if present)
  postprocessing_model/model.pt   (if present)
```

### Process 3 — Model Loader

Reads weights from local pod disk into GPU VRAM. Which subdirectory is loaded
depends on `serving_config.artifact_format` in the bundle:

```
artifact_format = SAFETENSORS    → load inference_model/original/rank*.safetensors
artifact_format = TRTLLM_ENGINE  → load inference_model/trtllm_engine/rank*.engine
```

For tensor-parallel deployments, each GPU rank loads its own shard independently
and in parallel — vLLM handles this internally. Also loads
`preprocessing_model/model.pt` and `postprocessing_model/model.pt` into GPU if
present in the bundle.

### Process 4 — Warm Process (CUDA Graph Capture)

Runs dummy inference at each fixed bucket size to capture CUDA graphs:

```
bucket sizes: [1, 2, 4, 8, 16, 32, ...]  (configured by platform per model)

for each bucket_size:
    dummy_input = zeros(batch=bucket_size, seq=max_model_len)
    with cuda.graph():
        output = model.forward(dummy_input)   # GPU records exact kernel sequence
    store graph in GPU VRAM
```

At request time: if the incoming batch fits a bucket, the pre-captured graph is
replayed instead of launching individual kernels — significantly lower per-request
GPU dispatch overhead.

**CUDA graph cache lives in GPU VRAM** for the lifetime of the pod. The graphs are
GPU constructs and cannot be executed from CPU.

**TRT-LLM path skips this entirely.** The `.engine` file IS the pre-compiled,
pre-fused graph — kernel selection, fusion, and scheduling are all baked in at
`trtllm-build` time. There is no runtime graph capture for TRT-LLM.

### Process 5 — Feature Store Initializer (co-located deployments only)

Reads `inference_model.input_features` from `bundle_metadata.pb` to know exactly
which feature partitions this model needs. Bootstraps the Venice Da Vinci Client
with those specific partitions onto local node disk/memory:

```
input_features → namespace + feature_store_name + feature_key
              → Da Vinci Client downloads those partitions from Venice cluster
              → features resident locally on the node
              → request-time lookup = local read (microseconds vs 5–20ms network)
```

Starts a background sync thread to keep the local partition current as upstream
Venice updates. Skipped entirely when the serving config uses remote feature store
calls.

### Readiness Gate

After all processes complete, the `readinessProbe` passes:

```yaml
readinessProbe:
  httpGet:
    path: /health/ready    # vLLM / Triton expose this; returns 200 only after
                           # weights loaded + CUDA graphs captured
  initialDelaySeconds: 60
  periodSeconds: 10
```

Pod is added to the Service endpoint list only after the probe passes. Raw traffic
never reaches a pod that has not completed all startup processes.

---

## Parallelization Within the Startup Sequence

Steps 3 and 4 are strictly sequential — CUDA graph capture requires weights already
in GPU VRAM. But the weight download (Process 2) and feature store bootstrap
(Process 5) are both I/O-bound and independent of each other.

Downloading `bundle_metadata.pb` first (tiny proto) unlocks parallel execution of
the two largest I/O operations:

```
[2a] Download bundle_metadata.pb   (small — done in seconds)
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
[2b] Download weights from          [5] Feature Store Init
     blob store to local disk           Read input_features from
     (large, I/O bound —               bundle_metadata.pb
     10–100 GB depending               Bootstrap Da Vinci Client
     on model size)                    (I/O bound — downloads
                                        feature partitions)
        │                                        │
        └─────────────────┬──────────────────────┘
                          ▼
                  [3] Model Loader
                  (weights → GPU VRAM)
                          │
                          ▼
                  [4] Warm Process
                  (CUDA graph capture)
                          │
                          ▼
                  readinessProbe passes
```

This cuts startup time by running the two largest I/O operations concurrently.

---

## Startup Optimization Techniques

### Node-Local Model Caching

**Problem:** Every pod restart re-downloads the full model bundle from blob store.
For a 70B model this is 140 GB — downloading takes minutes even on fast internal
network links.

**Solution:** Cache the model bundle on node-local NVMe SSD. The first pod on a
node downloads the bundle and writes it to a node-local cache directory
(e.g. `/node-cache/model-bundles/<model_name>/<version>/`). Subsequent pods on
the same node skip the download entirely and read from local disk.

```
Pod 1 startup on node A:
  cache miss → download from blob store (slow) → write to /node-cache/

Pod 2 startup on node A (same model):
  cache hit → read from /node-cache/ (fast NVMe read) → skip download
```

Cache invalidation: keyed by `model_name + version`. When a new version is
deployed the key changes, cache miss triggers a fresh download.

### Docker Image Pre-Pull

**Problem:** The serving container image (cuda12.1-torch2.3-vllm0.4.2) can be
10–20 GB. Pulling it at pod scheduling time adds minutes to startup.

**Solution:** A DaemonSet runs on every GPU node and pre-pulls known serving
images before any pod is scheduled there. By the time Kubernetes schedules the
pod, the image is already on the node — container start is near-instant.

```yaml
# DaemonSet that pre-pulls serving images onto every GPU node
containers:
- name: image-puller
  image: busybox
  command: ["docker", "pull", "registry.linkedin.com/llm-serving:cuda12.1-torch2.3-vllm0.4.2"]
```

### Feature Partition Pre-Loading (Co-located Deployments)

**Problem:** Da Vinci Client bootstrap downloads large feature partitions from
Venice cluster at pod startup. This can take minutes for high-cardinality feature
sets.

**Solution:** Same DaemonSet pattern — a background process pre-downloads the
feature partitions needed by each model onto the node before pods are scheduled.
When Process 5 runs, it finds the partitions already on local disk and bootstrap
time drops to near-zero.

### Parallel GPU Rank Loading (Tensor Parallel)

For TP > 1 deployments, each GPU rank loads its own weight shard independently.
vLLM handles this internally — ranks do not wait on each other. A 4-GPU TP=4
deployment loads 4 shards concurrently; effective weight loading time is
`total_weight_size / 4`, not `total_weight_size`.

### TRT-LLM Engine as Pre-Compiled Graph

The compiled `.engine` file bakes in kernel fusion, scheduling, and quantization
at `trtllm-build` time (run once, offline, as part of the optimization pipeline
step). Pod startup for TRT-LLM skips Process 4 entirely — there is no runtime
graph capture. The engine is loaded directly and is immediately ready to serve.

This is the production-grade solution to CUDA graph caching: instead of capturing
and serializing graphs at runtime, compile them once offline and store the result
as a first-class bundle artifact.

### CUDA Graph Serialization Cache (Emerging)

**For vLLM:** For the same model version + GPU type + CUDA version + bucket sizes,
the captured CUDA graph is deterministic — identical across every pod and every
restart. Recapturing from scratch on every pod start is wasted work.

CUDA 12.3+ added graph serialization APIs. The approach: after first capture,
serialize the graph binary and store in a remote cache (S3 / Redis) keyed by:

```
model_name + model_version + gpu_type + cuda_version + bucket_sizes
→ serialized graph binary
```

On next pod startup: cache hit → deserialize → skip warmup.

Not yet natively supported in vLLM (as of mid-2026), but the principle is the
same as what TRT-LLM already does with its `.engine` artifact. This would be the
vLLM-equivalent of pre-compiled graph storage.

---

## Startup Time Summary

```
Process                     Time driver           Optimization
─────────────────────────────────────────────────────────────────────
Pod init + image pull       image size (10–20 GB) DaemonSet pre-pull
Bundle download             model size (14–140 GB) Node-local cache
Feature store bootstrap     partition size         Node-local pre-load
                            + parallelized with bundle download
Model load to GPU           NVMe → VRAM bandwidth  GPU Direct Storage
                            parallel across TP ranks
CUDA graph capture          number of buckets ×    TRT-LLM: skip entirely
                            forward pass time       vLLM: serialize cache (future)
```
