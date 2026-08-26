# Component — ContextMaterializationCoordinator (KV-Tier Updated)

## 1. Why This Component Exists

The model executioner should not know:

```text
how to pin remote worker KV
how to read shared DRAM
how to restore NVMe
how to perform TP2->TP4 transformation
how to import cache pages
when to fall back to recompute
```

It should ask one question:

> **Make the selected compute stage's context usable.**

The ContextMaterializationCoordinator owns that protocol.

---

# 2. Expanded Responsibility

Earlier V5 modes:

```text
LOCAL_KV_REUSE
TRANSFER_KV
RECOMPUTE_FROM_CONTEXT
```

Updated modes:

```text
LOCAL_KV_REUSE

TRANSFER_WORKER_KV

RESTORE_SHARED_GPU_KV
RESTORE_CPU_KV
RESTORE_NVME_KV

EXACT_TRANSFORM_KV

RECOMPUTE_FROM_TARGET_CONTEXT
```

Optional future:

```text
SEMANTIC_MAP_KV
```

only if separately quality-qualified.

---

# 3. What It Does Not Decide

It does not decide:

```text
which quality model to use
which P/D topology is globally best
which tier should be preferred
which request wins a terminal execution lease
```

Those are planning/execution-coordination responsibilities.

It receives an **ordered materialization chain** for an already selected stage/destination.

---

# 4. Input

```text
MaterializeContextRequest {
    request_id
    stage_id

    destination_worker
    destination_epoch

    target_model_context_artifact

    required_prefix_tokens

    ordered_plans[]

    execution_lease

    deadline
}
```

---

# 5. Output

```text
MaterializedContext {
    status

    destination_worker
    destination_epoch

    materialized_prefix_tokens
    destination_kv_manifest_ref

    method_used
    source_residency_id?

    transform_id?

    materialization_latency_ms
}
```

Or terminal stage-local failure after exhausting plans.

---

# 6. Local KV Reuse

Fastest path:

```text
destination already has exact compatible prefix
```

Protocol:

```text
validate destination epoch
validate context digest
validate block manifest/version
validate compatibility
mark/reuse existing VALID blocks
```

No external data movement.

---

# 7. Worker-to-Worker Transfer

```text
source worker
    |
    | pin exact source blocks
    v
KVTransferService / NIXL
    |
    v
destination IMPORTING blocks
```

Protocol:

```text
1. JIT allocate destination import blocks.
2. Pin source exact manifest.
3. Transfer.
4. Validate completion.
5. Atomically publish destination blocks as VALID.
6. Unpin source.
```

On failure:

```text
abort partial import
unpin
next fallback
```

---

# 8. KVTier Restore

Difference from worker transfer:

```text
source is a storage object/residency,
not a live model-server block table
```

Protocol:

```text
1. JIT allocate destination KV import.
2. Acquire versioned KVTier read handle.
3. Validate object generation / logical identity.
4. Transfer object blocks to destination.
5. Apply exact representation transform if required.
6. Validate checksum/manifest as appropriate.
7. Finalize destination import -> VALID.
8. Release read handle.
```

Directory lookup is not sufficient; step 2 closes the race.

---

# 9. CPU / NVMe Restore

The coordinator hides tier mechanics from execution logic.

CPU path may be:

```text
pinned DRAM
    |
    | RDMA / GPUDirect capable path
    v
destination GPU
```

NVMe path may be:

```text
NVMe
   |
   | storage read / GPUDirect Storage / staging
   v
GPU/DRAM
   |
   v
destination cache
```

Concrete transport is delegated to KVTier/transfer backend.

Coordinator owns protocol correctness and fallback.

---

# 10. Exact Transform

Suppose source:

```text
same Llama-70B logical KV
TP2 / HND
```

destination:

```text
TP4 / NHD
```

If a trusted transform exists:

```text
source representation
      |
      | TP reshard + transpose
      v
destination representation
```

The coordinator invokes a transform service/kernel either:

```text
before transfer
during transfer/pipeline
after transfer into staging
```

depending on backend.

Important invariant:

> **Exact transform is legal only when semantic logical KV identity matches.**

---

# 11. Semantic Mapping Is Different

Cross-model mapper:

```text
Llama-8B KV
    ->
approximate Llama-70B KV
```

must never be represented as an ordinary exact transform.

If supported later:

```text
MaterializationMode = SEMANTIC_MAP_KV
```

with:

```text
mapper_id
quality qualification
uncertainty
supported context range
```

Its use belongs behind the same quality contract as model selection.

Base V5 can omit this mode.

---

# 12. Recompute

Universal fallback:

```text
TargetModelContextArtifact
    |
    | exact target tokens
    v
destination or Prefill worker
    |
    v
fresh target-model Prefill
```

Recompute is slower but strongest for correctness because it reconstructs exactly what the target model expects.

---

# 13. P/D Stage Materialization

For topology:

```text
KVTier CPU -> P3 -> D7
```

there are two stage-level materializations.

### Prefill stage

```text
materialize old prefix on P3
    ->
P3 computes missing suffix
```

### P->D handoff

```text
materialize updated P3 KV on D7
```

The ExecutionCoordinator/topology runner invokes the coordinator for each prerequisite boundary.

The coordinator itself remains stage-local.

---

# 14. Destination Allocation Rule

Retain the earlier invariant:

> **Do not reserve destination KV while a request waits in the terminal worker queue.**

Only after a topology candidate wins execution ownership:

```text
JIT allocate/import
```

This prevents queued future work from hoarding current GPU capacity.

For a remote Prefill prerequisite, temporary P resources are similarly acquired only when the selected topology is actually executing.

---

# 15. Ordered Fallback Chain

Example D7:

```text
1. LOCAL_KV_REUSE
2. RESTORE_CPU_KV(Node9)
3. TRANSFER_WORKER_KV(D2)
4. RECOMPUTE_FROM_TARGET_CONTEXT
```

Failure at one method does not automatically rerun global planning.

The coordinator tries the next pre-approved method for the same stage/destination while:

```text
deadline remains
compatibility remains valid
```

If every method fails:

```text
stage materialization failure
```

is returned to execution coordination.

---

# 16. Partial Transfer Safety

Destination state:

```text
FREE
 -> ALLOCATED
 -> IMPORTING
 -> VALID
```

Only `VALID` blocks are visible to attention execution.

Any failure during:

```text
restore
transfer
transform
checksum
```

returns imported blocks to a non-visible/free state.

No partial context becomes semantically usable in base V5.

---

# 17. Source Read Stability

Worker source:

```text
pin_count / source lease
```

KVTier source:

```text
versioned read handle / immutable object generation
```

Both provide the same logical guarantee:

> **The source bytes cannot change/recycle underneath an active transfer.**

---

# 18. Failure Classification

## Stale directory residency

```text
AcquireRead / Pin fails
    -> next fallback
```

## Destination capacity failure

```text
cannot JIT allocate
    -> stage failure
    -> possibly release topology candidate
```

## Transfer timeout

```text
abort partial import
next fallback if deadline allows
```

## Exact transform unsupported at runtime

```text
next fallback
repair metadata/compatibility registration
```

## KVTier unavailable

```text
worker source or recompute
```

None of these should corrupt SessionStore state.

---

# 19. Observability

Emit:

```text
ContextMaterializationObservation {
    request_id
    stage_id

    attempted_methods[]
    method_used

    source_tier?
    source_location?

    prefix_tokens
    bytes

    restore_ms
    transform_ms
    transfer_ms
    import_commit_ms

    fallback_count
    failure_reason?

    directory_version
    source_generation?
}
```

This data trains/calibrates future materialization-cost predictors.

---

# 20. Design Principles

1. Executioner asks for ready context, not a transport primitive.
2. Coordinator executes plans; Planner selects them.
3. Directory observation is revalidated JIT.
4. Worker pin and KVTier read-handle are parallel concepts.
5. Destination memory is JIT allocated.
6. Partial imports are never visible.
7. Exact transform preserves semantic KV identity.
8. Semantic mapping, if ever used, is a distinct quality-sensitive mode.
9. Ordered same-stage fallback avoids unnecessary replanning.
10. Recompute remains the universal safe fallback.
11. KVTier failures degrade performance, not semantic correctness.
12. Materialization metrics are first-class feedback.

---

# 21. Interview Answer

> **I keep context movement behind a ContextMaterializationCoordinator. The Planner may choose a CPU-tier restore, worker-to-worker transfer, local reuse, exact TP/layout transform or recompute, but the Executioner only asks for a usable context. At execution time the coordinator JIT-allocates destination KV, acquires a stable source—either a worker pin or an immutable KVTier read handle—moves and transforms the data, and only publishes destination blocks after the entire import is valid. Stale directory entries or tier failures simply advance the pre-approved fallback chain, with target-model re-Prefill as the universal safe fallback.**
