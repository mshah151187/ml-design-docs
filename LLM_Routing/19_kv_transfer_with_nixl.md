# KV Transfer with NIXL and RDMA

**Status:** V5 companion design document  
**Scope:** Cross-replica KV-cache transfer for stateful/session-aware LLM inference  
**Primary components:** `ContextMaterializationCoordinator`, `KVTransferService`, `ModelServerContextAPI`

> **Version note:** The serving-system architecture in this document is the stable design contract. Concrete NIXL / Dynamo API snippets are illustrative and version-sensitive. Verify exact API names against the deployed NIXL/Dynamo release.

---

# 1. Why Cross-Replica KV Transfer Exists

In session-driven inference, the ideal replica for the next turn is often the replica that already holds the conversation prefix in its KV cache.

```text
Session S123

R1
    reusable KV prefix = 60K tokens
    heavily loaded

R3
    reusable KV prefix = 0
    lightly loaded
```

Without KV transfer, the router has two obvious choices:

```text
Option A
--------
Stay on R1
    + almost no prefix recomputation
    - large queue/backlog delay


Option B
--------
Move to R3
    + low queue/backlog delay
    - recompute the entire 60K-token prefix
```

For long contexts, both can be poor.

A third path is:

```text
Option C
--------
Choose R3
    +
transfer compatible KV blocks from R1 -> R3
```

Conceptually:

```text
R1
60K-token KV cache
        |
        | high-speed transfer
        v
R3
        |
        v
continue inference using imported KV
```

The scheduling question becomes:

> **Which destination replica + context-materialization strategy provides the best SLO/cost outcome?**

For one destination replica, the main strategies are:

```text
LOCAL_KV_REUSE
TRANSFER_KV
RECOMPUTE_FROM_CONTEXT
```

---

# 2. Correctness Hierarchy

Cross-replica KV transfer is important, but it is still an optimization.

```text
Tier 1 — Durable semantic truth
--------------------------------
SessionStore
conversation events / committed session state


Tier 2 — Reconstructible model artifact
----------------------------------------
TargetModelContextArtifact
exact target-model serialization
token IDs
prefix block identities


Tier 3 — Ephemeral execution artifact
-------------------------------------
KV cache on R1 / R3
```

Therefore:

```text
KV transfer fails
        |
        v
recompute from target-model context

target-model artifact unavailable
        |
        v
reconstruct from durable SessionStore
```

### Core Invariant

> **Session state is correctness state. KV state is acceleration state.**

---

# 3. RDMA vs GPUDirect RDMA vs NIXL

## 3.1 RDMA

RDMA is a data-movement mechanism that allows one machine to access registered memory associated with another machine without requiring the remote CPU to copy every byte through an application socket loop.

Conceptually:

```text
WRITE

source memory
     |
     v
local NIC
     |
     | network fabric
     v
remote NIC
     |
     v
remote registered memory
```

Raw RDMA still leaves application responsibilities such as:

```text
memory registration
remote memory metadata
connection establishment
addresses / lengths
access permissions
transfer completion
transport details
```

---

## 3.2 GPUDirect RDMA

A host-staged transfer can look like:

```text
R1 GPU
   |
   v
R1 CPU DRAM
   |
   v
R1 NIC
   |
   v
network
   |
   v
R3 NIC
   |
   v
R3 CPU DRAM
   |
   v
R3 GPU
```

With GPUDirect RDMA, a capable NIC can directly exchange data with GPU memory:

```text
R1 GPU HBM
    |
    | DMA
    v
R1 NIC
    |
    | InfiniBand / RoCE
    v
R3 NIC
    |
    | DMA
    v
R3 GPU HBM
```

The CPUs still participate in the **control plane**:

```text
select blocks
pin blocks
allocate destination blocks
exchange metadata
start transfer
observe completion
handle failure
```

but they do not need to copy the bulk KV bytes themselves.

---

## 3.3 NIXL

NIXL sits above the raw transport and provides an inference-oriented transfer abstraction across memory and storage types.

```text
Our serving protocol
        |
        v
KVTransferService
        |
        v
NIXL
        |
        +--> UCX / RDMA-capable backend
        +--> other supported transfer backend
        |
        v
hardware / network
```

NIXL's job is primarily:

```text
describe source/destination memory
manage transfer metadata
select/use an available transfer backend
initiate asynchronous transfer
report completion/failure
```

NIXL does **not** know:

```text
which session version is correct
whether a KV block belongs to request X
whether R1 may evict the block
whether R3 should accept this context
whether model layouts are semantically compatible
which fallback should run after transfer failure
```

### Core Invariant

> **NIXL moves bytes. The serving platform decides whether those bytes are the correct KV state.**

---

# 4. V5 Component Responsibilities

## 4.1 `ReplicaScheduler`

Chooses an execution plan such as:

```text
destination = R3

context_fallback_chain:
    1. TRANSFER_KV(source=R1)
    2. RECOMPUTE_FROM_CONTEXT
```

It predicts latency/cost but does not perform transfer.

## 4.2 `ContextMaterializationCoordinator`

Owns the materialization protocol:

```text
execute ordered materialization plans
perform deadline checks
coordinate source/destination Model Server APIs
invoke KVTransferService
fallback from TRANSFER -> RECOMPUTE
clean partial state
return a valid MaterializedContext
```

## 4.3 `KVTransferService`

Owns the bulk-copy abstraction:

```text
translate KV manifests into transfer descriptors
exchange NIXL/RDMA metadata
start transfer
monitor completion
report bytes/time/backend/failure
```

## 4.4 `ModelServerContextAPI`

Owns KV memory lifecycle:

```text
locate source blocks
pin/unpin source blocks
allocate destination blocks just in time
mark destination blocks IMPORTING
publish blocks as VALID only after success
abort/free partial imports
return block manifests
```

---

# 5. Destination KV Is Not Reserved While Waiting

V5 deliberately does **not** reserve destination KV space while a request is still sitting in R3's queue.

```text
Req42 plans to transfer 30 GB to R3
but many older requests are ahead of it.
```

Early reservation could harm those older requests.

Therefore:

```text
request waits in R3 queue
        |
        v
R3 wins TryClaim / execution lease
        |
        v
ContextMaterializationCoordinator starts
        |
        v
JIT destination capacity check/allocation
```

If R3 no longer has enough capacity, the transfer path fails and the coordinator tries the next materialization plan.

### Core Invariant

> **No speculative destination KV reservation before execution ownership.**

---

# 6. Source KV Must Be Pinned

Source behavior is different. Once transfer starts, the source memory must remain stable.

Without pinning:

```text
1. Coordinator finds source block B on R1
2. Coordinator prepares R3 destination
3. R1 engine evicts/reuses B
4. DMA reads the old address
```

This can cause failure or silent corruption.

The source Model Server therefore exposes a conceptual API:

```text
PinKV(
    context_digest,
    source_epoch,
    block_manifest,
    lease_ttl
)
```

Successful pinning means:

> The specified blocks are still valid and may not be evicted/reused until the transfer pin is released or expires.

### Core Invariant

> **A source address may not participate in KV transfer unless the engine holds a temporary pin/lease protecting that block.**

---

# 7. Transfer Compatibility

NIXL can move bytes; it does not automatically reconcile incompatible KV layouts.

Direct transfer should require:

```text
source.kv_compatibility_id
    ==
destination.kv_compatibility_id
```

The compatibility contract may encode:

```text
model revision
attention/KV representation
KV dtype
tensor-parallel layout
pipeline ownership where relevant
block/page size
layer layout
head partitioning
adapter / LoRA identity
KV format revision
```

For V5:

```text
compatible layouts
    -> direct transfer allowed

incompatible layouts
    -> TRANSFER_KV not generated
    -> RECOMPUTE_FROM_CONTEXT
```

---

# 8. Exact Transfer Protocol: R1 -> R3

Suppose:

```text
Session S123
Context version = 27
Context digest = H123

R1 owns reusable KV
R3 won execution ownership

Materialization chain:
    1. TRANSFER_KV(R1 -> R3)
    2. RECOMPUTE_FROM_CONTEXT
```

Protocol:

```text
R3 Executioner
      |
      v
ContextMaterializationCoordinator
      |
      v
1. JIT allocate destination KV on R3
      |
      v
2. Validate + pin source KV on R1
      |
      v
3. Build destination writable descriptors
      |
      v
4. Exchange small transfer metadata
      |
      v
5. Initiate NIXL WRITE
      |
      v
6. NICs move KV bytes
      |
      v
7. Wait for completion
      |
      v
8. Validate/import destination KV
      |
      v
9. Mark destination blocks VALID
      |
      v
10. Unpin source
      |
      v
EXECUTING_PRE_STREAM
```

### Step 1 — JIT Destination Allocation

R3 allocates blocks in an `IMPORTING` state.

```text
SUCCESS
    allocation_id
    destination manifest

INSUFFICIENT_CAPACITY
    -> fallback path
```

### Step 2 — Source Validation + Pin

R1 verifies:

```text
source replica epoch still matches
context digest matches
requested blocks still exist
KV compatibility matches
blocks are valid
```

Then prevents eviction for the transfer lease.

### Step 3 — Destination Descriptors

R3's allocated blocks map to GPU memory ranges:

```text
block 0:
    gpu = 0
    address = 0x...
    length = ...

block 1:
    gpu = 1
    address = 0x...
    length = ...
```

NIXL wraps memory locations in descriptors.

### Step 4 — Control-Plane Metadata

R3 exposes writable-memory metadata.

Only small metadata crosses the control plane:

```text
transfer id
remote worker identity
registered-region information
descriptor information
context/version fencing metadata
```

### Step 5 — R1 Initiates WRITE

R1 has:

```text
local descriptors
    = pinned R1 KV blocks

remote metadata
    = R3 writable buffers
```

It starts the NIXL-backed transfer.

### Step 6 — Bulk Data Plane

```text
CONTROL PLANE
=============

ContextMaterializationCoordinator
      |
      +--> allocate
      +--> pin
      +--> exchange metadata
      +--> start
      +--> completion/failure


DATA PLANE
==========

R1 GPU ===================================> R3 GPU
                 NIXL / RDMA
```

The router never carries the KV payload.

### Step 7 — Completion

Possible states:

```text
IN_PROGRESS
DONE
ERROR
TIMEOUT/CANCELLED
```

The transfer obeys the materialization/execution deadline.

### Step 8 — Destination Validation

DMA completion only proves bytes moved.

R3 still validates:

```text
allocation belongs to request
context digest matches
expected block count/length matches
destination engine epoch still matches
allocation has not been revoked
```

### Step 9 — Publish KV

Only after full validation:

```text
IMPORTING -> VALID
```

The engine may now reference the imported blocks.

### Step 10 — Unpin Source

Source pins are released on:

```text
success
transfer failure
destination validation failure
timeout
cancellation
```

using `finally`-style cleanup.

---

# 9. Transfer Failure and Fallback

For R3:

```text
1. TRANSFER_KV(R1 -> R3)
2. RECOMPUTE_FROM_CONTEXT
```

A source-pin failure can become:

```text
R1 pin failed
      |
      v
TRANSFER_KV failed
      |
      v
R3 RECOMPUTE_FROM_CONTEXT
      |
      v
success
```

Only when all R3 materialization plans fail should R3 release execution ownership so parked sibling replicas can compete.

---

# 10. Partial Transfer Safety

Destination lifecycle:

```text
FREE
  |
  v
ALLOCATED
  |
  v
IMPORTING
  |
  +-------- success --------> VALID
  |
  +-------- failure --------> FREE
```

The engine cannot read `IMPORTING` blocks.

For base V5:

> **KV import is atomic at the context-materialization contract boundary.**

Partial descriptor success is not published as partially valid context.

---

# 11. Replica Epoch Fencing

Replica IDs survive process restarts conceptually; memory does not.

Therefore transfer plans include:

```text
source_replica_id = R1
source_epoch = 14

destination_replica_id = R3
destination_epoch = 8
```

Every Model Server call validates the expected epoch.

If either side restarts, the plan becomes stale and materialization fails safely.

---

# 12. Multi-GPU / Tensor-Parallel Transfer

For `TP=4`, KV may be sharded across four GPUs.

```text
R1.GPU0 -> R3.GPU0
R1.GPU1 -> R3.GPU1
R1.GPU2 -> R3.GPU2
R1.GPU3 -> R3.GPU3
```

A manifest can contain multiple shard transfers.

```text
KVTransferManifest {
    shards: [
        {source_gpu: 0, destination_gpu: 0, blocks: [...]},
        {source_gpu: 1, destination_gpu: 1, blocks: [...]},
        ...
    ]
}
```

These can execute concurrently, but they remain one logical materialization transaction.

If one shard fails, base V5 aborts the entire imported context.

NIXL moves buffers; it does not automatically reshard transformer KV from one incompatible TP layout into another.

---

# 13. Memory Registration Strategy

A poor design is:

```text
for every request:
    register block
    transfer
    deregister block
```

For stable KV arenas, long-lived registration should be considered where the chosen NIXL integration permits it:

```text
Model Server startup
      |
      v
allocate large KV arena
      |
      v
register arena with transfer subsystem
      |
      v
serve many request transfers using subranges
```

Then request-time work mostly identifies:

```text
offset/address
length
device
block mapping
```

Higher-level NIXL/Dynamo abstractions may intentionally use dynamic registration for simplicity, so the exact choice should be benchmarked.

---

# 14. Transfer Direction: WRITE vs READ

Both patterns are possible:

```text
WRITE:
    R1 pushes into R3

READ:
    R3 pulls from R1
```

This design uses WRITE:

```text
R3 creates writable destination memory
R1 owns/pins the source
R1 initiates write
```

The architecture does not require WRITE forever; `KVTransferService` hides that implementation choice.

---

# 15. Transfer Cost Model

A simple planning approximation:

```text
T_transfer
    =
      setup_cost
    + KV_bytes / effective_bandwidth
    + destination_import_cost
```

Useful empirical features:

```text
bytes
source/destination topology
same-node vs cross-node
NIC type
fabric type
concurrent transfer load
descriptor count
GPU topology
backend selected
historical bandwidth
historical setup latency
```

The scheduler compares:

```text
LOCAL_KV_REUSE
RECOMPUTE
TRANSFER_KV
```

not just replica identities.

---

# 16. Router Latency Budget Interaction

The router itself has a deadline.

Normal path:

```text
PreciseKVIndex
    |
    v
source candidate discovery
    |
    v
transfer cost prediction
    |
    v
TRANSFER_KV candidate
```

When router budget is nearly exhausted:

```text
skip optional transfer planning
        |
        v
emit RECOMPUTE_FROM_CONTEXT
```

The router should not spend excessive time trying to save downstream prefill.

---

# 17. Observability

Per-transfer:

```text
request_id
context_digest
source_replica
source_epoch
destination_replica
destination_epoch
kv_compatibility_id

block_count
bytes_requested
bytes_transferred

transport/backend
setup_duration_ms
transfer_duration_ms
import_duration_ms
source_pin_duration_ms

status
failure_reason
fallback_used
remaining_execution_budget_ms
```

Aggregates:

```text
transfer success rate
effective bandwidth
P50/P95/P99 transfer latency
pin failure rate
destination-capacity failure rate
epoch mismatch rate
partial-transfer abort rate
TRANSFER -> RECOMPUTE fallback rate
prefill tokens saved
SLO gain from transfer
```

---

# 18. Code — Serving-System Layer

The most important code is the code **we own**. NIXL stays behind `KVTransferService`.

## 18.1 Data Types

```python
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ContextMode(Enum):
    LOCAL_KV_REUSE = "local_kv_reuse"
    TRANSFER_KV = "transfer_kv"
    RECOMPUTE_FROM_CONTEXT = "recompute_from_context"


@dataclass(frozen=True)
class KVBlockRef:
    block_id: str
    logical_block_index: int
    size_bytes: int


@dataclass(frozen=True)
class TransferKVPlan:
    source_replica_id: str
    source_epoch: int

    destination_replica_id: str
    destination_epoch: int

    context_digest: str
    kv_compatibility_id: str

    source_blocks: Sequence[KVBlockRef]
    required_destination_blocks: int


@dataclass(frozen=True)
class MaterializationPlan:
    mode: ContextMode
    transfer: TransferKVPlan | None = None


@dataclass(frozen=True)
class MaterializationRequest:
    request_id: str
    context_digest: str
    target_context_artifact_ref: str

    destination_replica_id: str
    destination_epoch: int

    plans: Sequence[MaterializationPlan]

    deadline_at_monotonic: float
```

---

## 18.2 Coordinator

```python
class RetryableMaterializationError(Exception):
    pass


class ContextMaterializationFailed(Exception):
    pass


class ContextMaterializationCoordinator:

    def __init__(
        self,
        model_server_api,
        kv_transfer_service,
        context_recompute_service,
        clock,
    ):
        self.model_server_api = model_server_api
        self.kv_transfer_service = kv_transfer_service
        self.context_recompute_service = context_recompute_service
        self.clock = clock

    async def materialize(
        self,
        req: MaterializationRequest,
        execution_lease,
    ):
        errors = []

        for plan in req.plans:
            if self.clock.monotonic() >= req.deadline_at_monotonic:
                raise ContextMaterializationFailed(
                    f"materialization deadline exceeded; errors={errors}"
                )

            try:
                if plan.mode is ContextMode.LOCAL_KV_REUSE:
                    return await self._local_reuse(
                        req=req,
                        execution_lease=execution_lease,
                    )

                if plan.mode is ContextMode.TRANSFER_KV:
                    return await self._transfer_kv(
                        req=req,
                        plan=plan.transfer,
                        execution_lease=execution_lease,
                    )

                if plan.mode is ContextMode.RECOMPUTE_FROM_CONTEXT:
                    return await self._recompute(
                        req=req,
                        execution_lease=execution_lease,
                    )

                raise ValueError(f"unsupported context mode: {plan.mode}")

            except RetryableMaterializationError as exc:
                errors.append((plan.mode.value, str(exc)))
                continue

        raise ContextMaterializationFailed(
            f"all materialization plans failed: {errors}"
        )
```

---

## 18.3 Transfer Path

```python
async def _transfer_kv(
    self,
    req: MaterializationRequest,
    plan: TransferKVPlan,
    execution_lease,
):
    source = plan.source_replica_id
    destination = plan.destination_replica_id

    # 1. JIT destination allocation.
    dst = await self.model_server_api.allocate_import_kv(
        replica_id=destination,
        expected_replica_epoch=plan.destination_epoch,
        request_id=req.request_id,
        execution_lease_id=execution_lease.lease_id,
        num_blocks=plan.required_destination_blocks,
        kv_compatibility_id=plan.kv_compatibility_id,
    )

    if not dst.success:
        raise RetryableMaterializationError(
            f"destination allocation failed: {dst.reason}"
        )

    source_pin = None

    try:
        # 2. Validate + pin source KV.
        source_pin = await self.model_server_api.pin_kv(
            replica_id=source,
            expected_replica_epoch=plan.source_epoch,
            context_digest=plan.context_digest,
            kv_compatibility_id=plan.kv_compatibility_id,
            block_refs=plan.source_blocks,
            pin_deadline_at=req.deadline_at_monotonic,
        )

        if not source_pin.success:
            raise RetryableMaterializationError(
                f"source pin failed: {source_pin.reason}"
            )

        # 3. Bulk transfer.
        transfer_result = await self.kv_transfer_service.transfer_kv(
            KVTransferRequest(
                request_id=req.request_id,

                source_replica_id=source,
                source_epoch=plan.source_epoch,
                source_manifest=source_pin.manifest,

                destination_replica_id=destination,
                destination_epoch=plan.destination_epoch,
                destination_manifest=dst.manifest,

                context_digest=plan.context_digest,
                kv_compatibility_id=plan.kv_compatibility_id,

                deadline_at_monotonic=req.deadline_at_monotonic,
            )
        )

        if not transfer_result.success:
            raise RetryableMaterializationError(
                f"KV transfer failed: {transfer_result.reason}"
            )

        # 4. Publish imported KV only after full validation.
        commit = await self.model_server_api.commit_imported_kv(
            replica_id=destination,
            expected_replica_epoch=plan.destination_epoch,
            request_id=req.request_id,
            execution_lease_id=execution_lease.lease_id,
            allocation_id=dst.allocation_id,
            context_digest=plan.context_digest,
            kv_compatibility_id=plan.kv_compatibility_id,
            transferred_manifest=transfer_result.destination_manifest,
        )

        if not commit.success:
            raise RetryableMaterializationError(
                f"destination import rejected: {commit.reason}"
            )

        return MaterializedContext(
            replica_id=destination,
            mode=ContextMode.TRANSFER_KV,
            context_digest=plan.context_digest,
            kv_manifest=commit.manifest,
        )

    except Exception:
        # Partial destination bytes never become valid KV.
        await self.model_server_api.abort_import_kv(
            replica_id=destination,
            expected_replica_epoch=plan.destination_epoch,
            allocation_id=dst.allocation_id,
        )
        raise

    finally:
        if source_pin is not None and source_pin.success:
            await self.model_server_api.unpin_kv(
                replica_id=source,
                expected_replica_epoch=plan.source_epoch,
                pin_id=source_pin.pin_id,
            )
```

---

# 19. Code — KVTransferService

```python
@dataclass(frozen=True)
class KVTransferRequest:
    request_id: str

    source_replica_id: str
    source_epoch: int
    source_manifest: object

    destination_replica_id: str
    destination_epoch: int
    destination_manifest: object

    context_digest: str
    kv_compatibility_id: str

    deadline_at_monotonic: float


@dataclass(frozen=True)
class KVTransferResult:
    success: bool
    reason: str | None

    bytes_transferred: int
    transfer_duration_ms: float

    destination_manifest: object | None

    backend: str | None


class KVTransferService:
    async def transfer_kv(
        self,
        req: KVTransferRequest,
    ) -> KVTransferResult:
        raise NotImplementedError
```

A NIXL-backed implementation can remain simple:

```python
class NixlKVTransferService(KVTransferService):

    def __init__(self, replica_clients):
        self.replica_clients = replica_clients

    async def transfer_kv(self, req):
        source = self.replica_clients[
            req.source_replica_id
        ]
        destination = self.replica_clients[
            req.destination_replica_id
        ]

        # R3 exposes its already-allocated IMPORTING buffers.
        remote_write_metadata = (
            await destination.prepare_writable_kv(
                req.destination_manifest
            )
        )

        # R1 writes pinned KV buffers into R3.
        operation = await source.begin_kv_write(
            source_manifest=req.source_manifest,
            remote_write_metadata=remote_write_metadata,
        )

        try:
            await operation.wait_until(
                deadline_at=req.deadline_at_monotonic
            )
        except TimeoutError:
            await operation.cancel()

            return KVTransferResult(
                success=False,
                reason="transfer deadline exceeded",
                bytes_transferred=operation.bytes_transferred,
                transfer_duration_ms=operation.elapsed_ms,
                destination_manifest=None,
                backend=operation.backend,
            )

        if not operation.success:
            return KVTransferResult(
                success=False,
                reason=operation.error,
                bytes_transferred=operation.bytes_transferred,
                transfer_duration_ms=operation.elapsed_ms,
                destination_manifest=None,
                backend=operation.backend,
            )

        return KVTransferResult(
            success=True,
            reason=None,
            bytes_transferred=operation.bytes_transferred,
            transfer_duration_ms=operation.elapsed_ms,
            destination_manifest=req.destination_manifest,
            backend=operation.backend,
        )
```

---

# 20. Code — Model Server KV State

The engine-side state machine matters at least as much as NIXL.

```python
from enum import Enum


class KVBlockState(Enum):
    FREE = "free"
    VALID = "valid"
    IMPORTING = "importing"
```

Destination:

```python
class KVAllocator:

    def allocate_for_import(self, num_blocks):
        blocks = self._allocate_free_blocks(num_blocks)

        for block in blocks:
            assert block.state is KVBlockState.FREE
            block.state = KVBlockState.IMPORTING

        return blocks

    def commit_import(self, blocks, context_digest):
        for block in blocks:
            if block.state is not KVBlockState.IMPORTING:
                raise RuntimeError("invalid import state")

        self._validate_import(
            blocks,
            context_digest,
        )

        for block in blocks:
            block.context_digest = context_digest
            block.state = KVBlockState.VALID

    def abort_import(self, blocks):
        for block in blocks:
            if block.state is KVBlockState.IMPORTING:
                block.reset()
                block.state = KVBlockState.FREE
```

Source pinning should normally be modeled by a pin count rather than converting `VALID` into a mutually exclusive state:

```python
class KVCache:

    def pin_for_transfer(
        self,
        block_ids,
        expected_context_digest,
        pin_expiry,
    ):
        blocks = []

        for block_id in block_ids:
            block = self.lookup(block_id)

            if block is None:
                raise KVNotPresent(block_id)

            if block.context_digest != expected_context_digest:
                raise KVVersionMismatch(block_id)

            if block.state is not KVBlockState.VALID:
                raise KVNotTransferable(block_id)

            blocks.append(block)

        # Validate all first, then publish pins.
        for block in blocks:
            block.pin_count += 1

        return SourceKVPin(
            pin_id=new_pin_id(),
            blocks=blocks,
            expiry=pin_expiry,
        )

    def unpin(self, pin):
        for block in pin.blocks:
            block.pin_count -= 1
            assert block.pin_count >= 0
```

Eviction:

```python
def may_evict(block):
    return (
        block.state is KVBlockState.VALID
        and block.pin_count == 0
    )
```

This distinction is useful because a valid source block remains valid while temporarily pinned; the pin is an eviction constraint, not a new semantic KV state.

---

# 21. Code — Current Dynamo NIXL Connect Style

The current Dynamo NIXL Connect abstraction exposes concepts such as:

```text
Connector
Descriptor
create_writable(...)
WritableOperation.metadata
begin_write(...)
wait_for_completion()
```

Exact APIs are version-sensitive.

## 21.1 Destination R3 Creates Writable Memory

```python
from dynamo.nixl_connect import Connector, Descriptor


class R3NixlEndpoint:

    def __init__(self, worker_id: str):
        self.connector = Connector(
            worker_id=worker_id
        )
        self.pending_writable = {}

    def prepare_writable(
        self,
        destination_tensors,
    ):
        descriptors = [
            Descriptor(tensor)
            for tensor in destination_tensors
        ]

        writable = self.connector.create_writable(
            descriptors
        )

        metadata = writable.metadata

        transfer_id = create_transfer_id()

        self.pending_writable[
            transfer_id
        ] = writable

        return transfer_id, metadata
```

The destination tensors point into R3's `IMPORTING` KV allocations.

The metadata is small control-plane state sent to R1.

---

## 21.2 Source R1 Begins WRITE

```python
from dynamo.nixl_connect import Connector, Descriptor


class R1NixlEndpoint:

    def __init__(self, worker_id: str):
        self.connector = Connector(
            worker_id=worker_id
        )

    async def write_kv(
        self,
        pinned_source_tensors,
        remote_metadata,
    ):
        source_descriptors = [
            Descriptor(tensor)
            for tensor in pinned_source_tensors
        ]

        write_op = self.connector.begin_write(
            local_descriptors=source_descriptors,
            remote_metadata=remote_metadata,
        )

        await write_op.wait_for_completion()

        return write_op
```

Conceptually:

```text
R3:
    create_writable(dst tensors)
          |
          v
    remote metadata
          |
          v
R1:
    begin_write(src tensors, remote metadata)
          |
          v
    await completion
```

---

## 21.3 Destination Observes Completion

```python
async def wait_for_remote_write(
    self,
    transfer_id,
):
    writable = self.pending_writable[
        transfer_id
    ]

    try:
        await writable.wait_for_completion()
    finally:
        self.pending_writable.pop(
            transfer_id,
            None,
        )
```

Only after transfer completion **and** Model Server validation does:

```text
IMPORTING -> VALID
```

---

# 22. Code — Lower-Level NIXL Style

At a lower level, the pattern is:

```text
build descriptors
initialize transfer handle
post transfer
poll/check completion
```

Illustrative sketch:

```python
def transfer_with_nixl_agent(
    agent,
    source_tensors,
    remote_descriptors,
    remote_agent_name,
):
    local_descs = agent.get_xfer_descs(
        source_tensors
    )

    handle = agent.initialize_xfer(
        "WRITE",
        local_descs,
        remote_descriptors,
        remote_agent_name,
        b"kv-transfer",
    )

    try:
        status = agent.transfer(handle)

        while status == "PROC":
            status = agent.check_xfer_state(
                handle
            )

        if status != "DONE":
            raise RuntimeError(
                f"NIXL transfer failed: {status}"
            )

    finally:
        handle.release()
```

The current lower-level API uses statuses equivalent to:

```text
DONE
PROC
ERR
```

and can expose transfer telemetry and the chosen backend.

All of this remains an implementation detail of `KVTransferService`.

---

# 23. Long-Lived Registration

Where appropriate for the chosen integration, a Model Server can register a large stable KV arena once rather than registering every block per request.

Conceptually:

```python
class ModelServer:

    def __init__(
        self,
        kv_arena,
        nixl_agent,
    ):
        self.kv_arena = kv_arena
        self.nixl_agent = nixl_agent

        # Illustrative low-level model.
        self.registration = (
            self.nixl_agent.register_memory(
                self.kv_arena
            )
        )
```

Requests then identify subranges:

```python
def descriptor_for_block(block):
    return (
        block.gpu_address,
        block.size_bytes,
        block.gpu_id,
    )
```

Whether this is preferable to dynamic registration should be benchmarked against the actual NIXL/Dynamo integration.

---

# 24. Multi-GPU Transfer Code

For TP=4:

```python
@dataclass(frozen=True)
class TransferShard:
    source_gpu: int
    destination_gpu: int
    source_tensors: list
    destination_tensors: list
```

Transfers can be concurrent:

```python
async def transfer_tp_shards(
    service,
    shards,
    deadline_at,
):
    await asyncio.gather(
        *[
            service.transfer_shard(
                shard,
                deadline_at=deadline_at,
            )
            for shard in shards
        ]
    )
```

But all shards remain one logical transaction.

If one shard fails, base V5 aborts the whole imported context.

---

# 25. What the Code Must Defend Against

The transfer path must handle:

```text
1. Source KV evicted before pin
2. Source replica restarted
3. Destination replica restarted
4. Destination lost capacity before JIT allocation
5. Destination allocation revoked
6. Transfer timed out
7. Partial descriptor transfer
8. RDMA/backend failure
9. Context digest mismatch
10. KV compatibility mismatch
11. Source pin TTL expired
12. Execution lease expired
13. Request cancelled
14. Destination import commit failed
```

Normal failure behavior:

```text
abort partial destination state
release source pin
record failure
try next materialization plan
```

not:

```text
salvage ambiguous KV bytes
```

---

# 26. Final End-to-End Code Flow

```python
execution_lease = await registry.try_claim(
    request_id=req.id,
    replica_id="R3",
)

materialized = await context_coordinator.materialize(
    MaterializationRequest(
        request_id=req.id,
        context_digest=context.digest,
        target_context_artifact_ref=context.artifact_ref,

        destination_replica_id="R3",
        destination_epoch=8,

        plans=[
            MaterializationPlan(
                mode=ContextMode.TRANSFER_KV,
                transfer=TransferKVPlan(
                    source_replica_id="R1",
                    source_epoch=14,

                    destination_replica_id="R3",
                    destination_epoch=8,

                    context_digest=context.digest,
                    kv_compatibility_id="KVC-42",

                    source_blocks=matched_blocks,
                    required_destination_blocks=len(
                        matched_blocks
                    ),
                ),
            ),

            MaterializationPlan(
                mode=ContextMode.RECOMPUTE_FROM_CONTEXT,
            ),
        ],

        deadline_at_monotonic=deadline,
    ),

    execution_lease=execution_lease,
)

await model_server.execute(
    request=req,
    materialized_context=materialized,
)
```

Notice what the application-level flow does **not** expose:

```text
RDMA queue pairs
remote keys
InfiniBand verbs
UCX internals
GPU BAR mappings
NIXL backend internals
```

Those stay below `KVTransferService`.

---

# 27. Staff+ Interview Summary

> For a stateful request, the router may choose a destination replica that does not currently own the best KV prefix. If cross-replica transfer is predicted to beat recomputation, the destination first wins execution ownership and then performs a just-in-time KV allocation. The source Model Server validates and temporarily pins the exact compatible KV blocks to prevent eviction races. The destination exposes the allocated GPU buffers through the transfer subsystem, and the source initiates an asynchronous NIXL-backed transfer, ideally using GPUDirect RDMA so the NIC moves data directly between GPU memories. The destination keeps imported blocks in an `IMPORTING` state and publishes them atomically as valid KV only after transfer completion and metadata validation. On any failure we free partial destination state, release the source pin, and fall back to recomputation. NIXL owns byte movement; the serving platform owns context identity, compatibility, memory lifecycle, leases, deadlines, and fallback.

---

# 28. Design Principles Summary

1. **NIXL moves bytes; the serving platform owns KV correctness.**
2. **Session state remains durable even when all KV is lost.**
3. **No destination KV reservation while a request waits in the queue.**
4. **Destination allocation happens just in time after execution ownership.**
5. **Source KV must be pinned before DMA begins.**
6. **Direct transfer requires strict KV-layout compatibility.**
7. **Replica epochs fence stale source/destination metadata.**
8. **Imported KV remains `IMPORTING` until the entire import validates.**
9. **Partial transfer never becomes partially valid execution state.**
10. **Transfer failure falls back to recomputation before abandoning the destination replica.**
11. **Large KV payloads stay on the data plane; routers exchange only metadata.**
12. **Transfer planning respects router and execution latency budgets.**
13. **Long-lived memory registration should be considered to reduce registration overhead.**
14. **TP/multi-GPU transfer is a set of shard transfers under one logical materialization transaction.**
15. **NIXL APIs remain an implementation detail behind `KVTransferService`.**

---

# 29. API Reference Note

The concrete API examples in this document were checked against current primary documentation/code for:

```text
NVIDIA Dynamo NIXL Connect API
NIXL Python API
NIXL expanded two-peer Python example
NVIDIA GPUDirect RDMA documentation
```

Because NIXL/Dynamo are actively evolving, verify exact imports, metadata types, registration behavior, and transfer APIs against the version deployed in the serving environment.
