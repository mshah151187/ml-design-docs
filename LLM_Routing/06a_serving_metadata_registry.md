# Component — ServingMetadataRegistry

## Responsibility

Provide static/slowly changing execution metadata required by Planner and ReplicaScheduler: `ServingProfile`, KV representation compatibility, and cost priors.

This is control-plane metadata. It is materialized locally for request-time reads; the request path should not synchronously call a remote metadata database for every candidate.

## API — `GetServingProfile`

### Input

```text
GetServingProfileRequest {
    serving_profile_id: string
}
```

### Output — `ServingProfile`

```text
ServingProfile {
    serving_profile_id: string

    model_id: string
    model_revision: string

    engine: string
    engine_version: string

    tensor_parallel_degree: int
    pipeline_parallel_degree: int

    precision: string
    quantization: string?

    max_context_tokens: int
    max_batch_tokens: int?

    kv_compatibility_id: string
    kv_block_size_tokens: int
    kv_dtype: string

    chunked_prefill_enabled: bool
    prefix_caching_enabled: bool
    speculative_decoding_enabled: bool

    attention_backend: string?
    cuda_graph_mode: string?

    hardware: HardwareProfile

    baseline_prefill_tokens_per_sec: float?
    baseline_decode_tokens_per_sec: float?

    metadata_version: string
}
```

### `HardwareProfile`

```text
HardwareProfile {
    accelerator_type: string
    accelerator_count: int
    accelerator_memory_gb: float

    cpu_count: int
    host_memory_gb: float

    interconnect: string?
    network_class: string?
}
```

## API — `GetKVCompatibilityProfile`

### Input

```text
GetKVCompatibilityProfileRequest {
    kv_compatibility_id: string
}
```

### Output — `KVCompatibilityProfile`

```text
KVCompatibilityProfile {
    kv_compatibility_id: string

    model_id: string
    model_revision: string

    adapter_id: string?

    kv_block_size_tokens: int
    kv_dtype: string
    kv_layout_version: string
    attention_layout_id: string?

    hash_schema_version: string

    transferable_across_replicas: bool
    supported_transfer_tiers: set<GPU | CPU | SHARED>

    metadata_version: string
}
```

Two replicas may use cross-replica `TRANSFER_KV` only when their compatibility contract permits it. Hardware/engine names alone are not sufficient.

## API — `GetCostProfile`

### Input

```text
GetCostProfileRequest {
    serving_profile_id: string
    cost_policy_id: string
}
```

### Output — `ServingCostProfile`

```text
ServingCostProfile {
    serving_profile_id: string
    cost_policy_id: string

    accelerator_cost_per_second: float
    host_cost_per_second: float?

    transfer_cost_per_gb: float?
    memory_occupancy_cost_per_gb_second: float?

    accounting_currency_or_unit: string
    cost_profile_version: string
}
```

The generation of these priors is outside Model Router; Scheduler consumes them.

## Invariants

- `ServingProfile` describes execution configuration, not current load.
- `KVCompatibilityProfile` is the explicit high-level contract for transferable cache representation.
- Static priors assist cold/sparse performance state but trustworthy empirical performance can dominate latency prediction.
- Metadata updates are versioned and locally materialized before request-time use.
