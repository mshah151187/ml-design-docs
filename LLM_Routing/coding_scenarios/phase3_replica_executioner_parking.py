"""
Staff+ ML Infra Interview Exercise
==================================

Scenario 6: ReplicaExecutioner + Thread-Safe Parked Requests

Latest architecture
-------------------
Each replica has one logical FIFO candidate queue.

The same request may appear in Top-K=3 queues, but RequestExecutionRegistry
allows only one execution owner.

If another replica currently owns the request, a sibling candidate is PARKED,
not permanently discarded, because the owner may fail before streaming.

A shared background poller revisits parked entries.

This exercise focuses on:
- asyncio.Queue
- multiple worker coroutines
- one shared parked map
- lock discipline
- polling/backoff
- stale phase fencing
- expired lease recovery
- graceful shutdown
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Awaitable, Callable, Dict, Optional, Protocol


class ExecutionState(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    EXECUTING_PRE_STREAM = "EXECUTING_PRE_STREAM"
    STREAMING = "STREAMING"

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REQUEST_INVALID = "REQUEST_INVALID"


TERMINAL_STATES = {
    ExecutionState.SUCCESS,
    ExecutionState.FAILED,
    ExecutionState.CANCELLED,
    ExecutionState.REQUEST_INVALID,
}


@dataclass(frozen=True, slots=True)
class RequestExecutionState:
    request_id: str
    phase: int
    state: ExecutionState

    owner_replica: Optional[str] = None
    owner_executioner: Optional[str] = None

    lease_id: Optional[str] = None
    lease_expires_at_monotonic: Optional[float] = None

    version: int = 0


class ClaimStatus(str, Enum):
    CLAIMED = "CLAIMED"
    ALREADY_OWNED_BY_CALLER = "ALREADY_OWNED_BY_CALLER"

    OWNED_BY_OTHER = "OWNED_BY_OTHER"
    STALE_PHASE = "STALE_PHASE"
    TERMINAL = "TERMINAL"
    STREAMING = "STREAMING"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class ClaimResult:
    status: ClaimStatus
    state: Optional[RequestExecutionState]

    @property
    def lease_id(self) -> Optional[str]:
        return self.state.lease_id if self.state is not None else None


class TransitionStatus(str, Enum):
    APPLIED = "APPLIED"
    ALREADY_APPLIED = "ALREADY_APPLIED"

    STALE_PHASE = "STALE_PHASE"
    NOT_OWNER = "NOT_OWNER"
    INVALID_STATE = "INVALID_STATE"
    TERMINAL = "TERMINAL"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class TransitionResult:
    status: TransitionStatus
    state: Optional[RequestExecutionState]


@dataclass(frozen=True, slots=True)
class ReplicaQueueEntry:
    request_id: str
    phase: int

    replica_id: str
    deployment_id: str
    candidate_rank: int

    enqueue_time_monotonic: float
    admission_deadline_monotonic: Optional[float]

    estimated_prefill_tokens: int
    estimated_decode_tokens: int


@dataclass(slots=True)
class ParkedEntry:
    entry: ReplicaQueueEntry

    parked_at_monotonic: float
    next_poll_at_monotonic: float

    poll_count: int = 0
    last_observed_version: Optional[int] = None


class ExecutionRegistryClient(Protocol):
    async def try_claim(
        self,
        *,
        request_id: str,
        phase: int,
        replica_id: str,
        executioner_id: str,
        lease_ttl_seconds: float,
    ) -> ClaimResult:
        ...

    async def get_state(
        self,
        request_id: str,
    ) -> Optional[RequestExecutionState]:
        ...

    async def mark_executing(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        ...

    async def release_for_retry(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        ...

    async def mark_streaming(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        ...

    async def mark_success(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        ...


class ModelServerResultType(str, Enum):
    SUCCESS = "SUCCESS"
    RETRYABLE_PRE_STREAM_FAILURE = "RETRYABLE_PRE_STREAM_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


@dataclass(frozen=True, slots=True)
class ModelServerResult:
    result_type: ModelServerResultType
    stream_started: bool = False
    response: Optional[object] = None
    message: Optional[str] = None


class ModelServerClient(Protocol):
    async def execute(
        self,
        *,
        request_id: str,
        replica_id: str,
        on_stream_started: Callable[[], Awaitable[None]],
    ) -> ModelServerResult:
        ...


class FakeExecutionRegistryClient:
    """
    GIVEN FAKE for this exercise only.

    It intentionally abstracts production Redis details.
    """

    def __init__(self) -> None:
        self._states: Dict[str, RequestExecutionState] = {}
        self._lock = asyncio.Lock()
        self._lease_counter = 0

    async def set_state(
        self,
        state: RequestExecutionState,
    ) -> None:
        async with self._lock:
            self._states[state.request_id] = state

    async def get_state(
        self,
        request_id: str,
    ) -> Optional[RequestExecutionState]:
        async with self._lock:
            return self._states.get(request_id)

    async def try_claim(
        self,
        *,
        request_id: str,
        phase: int,
        replica_id: str,
        executioner_id: str,
        lease_ttl_seconds: float,
    ) -> ClaimResult:
        async with self._lock:
            now = time.monotonic()
            state = self._states.get(request_id)

            if state is None:
                return ClaimResult(ClaimStatus.MISSING, None)

            if state.phase != phase:
                return ClaimResult(ClaimStatus.STALE_PHASE, state)

            if state.state in TERMINAL_STATES:
                return ClaimResult(ClaimStatus.TERMINAL, state)

            if state.state == ExecutionState.STREAMING:
                return ClaimResult(ClaimStatus.STREAMING, state)

            live_lease = (
                state.lease_id is not None
                and state.lease_expires_at_monotonic is not None
                and state.lease_expires_at_monotonic > now
            )

            if (
                live_lease
                and state.owner_replica == replica_id
                and state.owner_executioner == executioner_id
            ):
                return ClaimResult(
                    ClaimStatus.ALREADY_OWNED_BY_CALLER,
                    state,
                )

            if live_lease:
                return ClaimResult(ClaimStatus.OWNED_BY_OTHER, state)

            # PENDING or expired recoverable pre-stream lease.
            self._lease_counter += 1
            lease_id = f"lease-{self._lease_counter}"

            claimed = replace(
                state,
                state=ExecutionState.CLAIMED,
                owner_replica=replica_id,
                owner_executioner=executioner_id,
                lease_id=lease_id,
                lease_expires_at_monotonic=now + lease_ttl_seconds,
                version=state.version + 1,
            )
            self._states[request_id] = claimed

            return ClaimResult(ClaimStatus.CLAIMED, claimed)

    async def mark_executing(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        async with self._lock:
            state = self._states.get(request_id)

            if state is None:
                return TransitionResult(TransitionStatus.MISSING, None)

            if state.phase != phase:
                return TransitionResult(TransitionStatus.STALE_PHASE, state)

            if state.lease_id != lease_id:
                return TransitionResult(TransitionStatus.NOT_OWNER, state)

            if state.state == ExecutionState.EXECUTING_PRE_STREAM:
                return TransitionResult(
                    TransitionStatus.ALREADY_APPLIED,
                    state,
                )

            if state.state != ExecutionState.CLAIMED:
                return TransitionResult(
                    TransitionStatus.INVALID_STATE,
                    state,
                )

            updated = replace(
                state,
                state=ExecutionState.EXECUTING_PRE_STREAM,
                version=state.version + 1,
            )
            self._states[request_id] = updated
            return TransitionResult(TransitionStatus.APPLIED, updated)

    async def release_for_retry(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        async with self._lock:
            state = self._states.get(request_id)

            if state is None:
                return TransitionResult(TransitionStatus.MISSING, None)

            if state.phase != phase:
                return TransitionResult(TransitionStatus.STALE_PHASE, state)

            if state.lease_id != lease_id:
                return TransitionResult(TransitionStatus.NOT_OWNER, state)

            if state.state not in {
                ExecutionState.CLAIMED,
                ExecutionState.EXECUTING_PRE_STREAM,
            }:
                return TransitionResult(
                    TransitionStatus.INVALID_STATE,
                    state,
                )

            updated = replace(
                state,
                state=ExecutionState.PENDING,
                owner_replica=None,
                owner_executioner=None,
                lease_id=None,
                lease_expires_at_monotonic=None,
                version=state.version + 1,
            )
            self._states[request_id] = updated
            return TransitionResult(TransitionStatus.APPLIED, updated)

    async def mark_streaming(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        async with self._lock:
            state = self._states.get(request_id)

            if state is None:
                return TransitionResult(TransitionStatus.MISSING, None)

            if state.phase != phase:
                return TransitionResult(TransitionStatus.STALE_PHASE, state)

            if state.lease_id != lease_id:
                return TransitionResult(TransitionStatus.NOT_OWNER, state)

            if state.state == ExecutionState.STREAMING:
                return TransitionResult(
                    TransitionStatus.ALREADY_APPLIED,
                    state,
                )

            if state.state != ExecutionState.EXECUTING_PRE_STREAM:
                return TransitionResult(
                    TransitionStatus.INVALID_STATE,
                    state,
                )

            updated = replace(
                state,
                state=ExecutionState.STREAMING,
                version=state.version + 1,
            )
            self._states[request_id] = updated
            return TransitionResult(TransitionStatus.APPLIED, updated)

    async def mark_success(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        async with self._lock:
            state = self._states.get(request_id)

            if state is None:
                return TransitionResult(TransitionStatus.MISSING, None)

            if state.phase != phase:
                return TransitionResult(TransitionStatus.STALE_PHASE, state)

            if state.state == ExecutionState.SUCCESS:
                if state.lease_id == lease_id:
                    return TransitionResult(
                        TransitionStatus.ALREADY_APPLIED,
                        state,
                    )
                return TransitionResult(
                    TransitionStatus.NOT_OWNER,
                    state,
                )

            if state.lease_id != lease_id:
                return TransitionResult(TransitionStatus.NOT_OWNER, state)

            if state.state not in {
                ExecutionState.EXECUTING_PRE_STREAM,
                ExecutionState.STREAMING,
            }:
                return TransitionResult(
                    TransitionStatus.INVALID_STATE,
                    state,
                )

            updated = replace(
                state,
                state=ExecutionState.SUCCESS,
                version=state.version + 1,
            )
            self._states[request_id] = updated
            return TransitionResult(TransitionStatus.APPLIED, updated)



class ReplicaExecutioner:
    """
    MAIN CLASS TO IMPLEMENT.

    One instance belongs to one replica. Several instances can consume the same
    logical replica queue.

    Lock rule:
        Never hold `_parked_lock` across registry/model-server I/O.
    """

    def __init__(
        self,
        *,
        replica_id: str,
        executioner_id: str,

        ready_queue: asyncio.Queue[ReplicaQueueEntry],
        registry: ExecutionRegistryClient,
        model_server: ModelServerClient,

        worker_count: int = 4,

        lease_ttl_seconds: float = 5.0,

        parked_poll_interval_seconds: float = 0.050,
        max_parked_poll_interval_seconds: float = 0.500,

        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._replica_id = replica_id
        self._executioner_id = executioner_id

        self._ready_queue = ready_queue
        self._registry = registry
        self._model_server = model_server

        self._worker_count = worker_count
        self._lease_ttl_seconds = lease_ttl_seconds

        self._parked_poll_interval_seconds = parked_poll_interval_seconds
        self._max_parked_poll_interval_seconds = (
            max_parked_poll_interval_seconds
        )

        self._clock = clock

        self._parked: Dict[tuple[str, int], ParkedEntry] = {}
        self._parked_lock = asyncio.Lock()

        self._tasks: list[asyncio.Task[None]] = []
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """
        Start:
          worker_count ready-queue consumers
          one shared parked-request poller

        Reject double-start or deliberately make it idempotent.
        """
        raise NotImplementedError

    async def stop(self) -> None:
        """
        Graceful shutdown.

        Discuss:
        - cancel vs drain
        - workers blocked in queue.get()
        - active model-server calls
        - parked entries
        - task cleanup
        """
        raise NotImplementedError

    async def enqueue(
        self,
        entry: ReplicaQueueEntry,
    ) -> None:
        """
        Validate replica_id and enqueue to FIFO.
        """
        raise NotImplementedError

    async def _worker_loop(
        self,
        worker_index: int,
    ) -> None:
        """
        Consume:
            entry = await queue.get()
            try:
                await _process_ready_entry(entry)
            finally:
                queue.task_done()
        """
        raise NotImplementedError

    async def _process_ready_entry(
        self,
        entry: ReplicaQueueEntry,
    ) -> None:
        """
        1. Cheap local deadline check.
        2. registry.try_claim()
        3. branch:

        CLAIMED / ALREADY_OWNED_BY_CALLER:
            _execute_claimed()

        OWNED_BY_OTHER:
            _park()

        STALE_PHASE / TERMINAL / STREAMING / MISSING:
            discard according to policy

        Registry state, not queue presence, determines correctness.
        """
        raise NotImplementedError

    async def _execute_claimed(
        self,
        entry: ReplicaQueueEntry,
        *,
        lease_id: str,
    ) -> None:
        """
        Suggested flow:
        1. mark_executing
        2. call model server
        3. callback on first visible output -> mark_streaming
        4. success -> mark_success
        5. retryable pre-stream rejection/failure -> release_for_retry
        6. terminal failure -> production registry terminal-failure API

        Model-server OVER_CAPACITY/THROTTLE is typically a retryable pre-stream
        release so parked sibling candidates can compete.
        """
        raise NotImplementedError

    async def _park(
        self,
        entry: ReplicaQueueEntry,
        *,
        observed_state: Optional[RequestExecutionState],
    ) -> None:
        """
        Key by (request_id, phase).

        Deduplicate local copies.

        Initialize next poll and record observed registry version.
        """
        raise NotImplementedError

    async def _parked_poller_loop(self) -> None:
        """
        One shared polling loop for all parked requests.

        Do NOT allocate one timer/task per parked request.

        For each due parked entry:
            state = await registry.get_state(...)
            await _reconcile_parked(...)
        """
        raise NotImplementedError

    async def _take_due_parked_snapshot(
        self,
        now: float,
    ) -> list[ParkedEntry]:
        """
        Copy due items under lock, then release lock before I/O.
        """
        raise NotImplementedError

    async def _reconcile_parked(
        self,
        parked: ParkedEntry,
        state: Optional[RequestExecutionState],
    ) -> None:
        """
        MISSING / terminal:
            discard

        state.phase != parked.phase:
            discard stale entry

        PENDING:
            remove parked record and requeue

        CLAIMED / EXECUTING_PRE_STREAM with LIVE lease:
            remain parked and back off

        CLAIMED / EXECUTING_PRE_STREAM with EXPIRED lease:
            remove parked record and requeue so try_claim() can perform the
            atomic fenced takeover

        STREAMING:
            remain logically blocked until terminal; polling can back off more
            aggressively because V1 does not fail over after streaming

        Important race:
        reading PENDING then requeueing is safe even if another replica claims in
        between. The later try_claim() simply loses and parks again.
        """
        raise NotImplementedError

    async def _remove_parked_if_same(
        self,
        parked: ParkedEntry,
    ) -> bool:
        """
        Prevent stale poll results from deleting a newer parked replacement.
        """
        raise NotImplementedError

    async def _reschedule_parked(
        self,
        parked: ParkedEntry,
        *,
        observed_version: Optional[int],
    ) -> None:
        """
        Bounded exponential backoff; production may add jitter.
        """
        raise NotImplementedError

    async def parked_count(self) -> int:
        async with self._parked_lock:
            return len(self._parked)


@dataclass(frozen=True, slots=True)
class ReplicaQueueLoad:
    router_queued_requests: int = 0
    router_queued_prefill_tokens: int = 0
    router_queued_decode_tokens: int = 0

    executioner_inflight_requests: int = 0
    executioner_inflight_prefill_tokens: int = 0
    executioner_inflight_decode_tokens: int = 0


class QueueLoadTracker:
    """
    OPTIONAL LEVEL-2 EXTENSION.

    Maintain the router-owned terms in:

        EffectiveLoad
          = EngineReportedLoad
          + RouterQueueLoad
          + ExecutionerInflightLoad

    Physical queue length is not enough because stale/terminal records can remain
    until lazy consumption.
    """

    async def on_logically_enqueued(
        self,
        entry: ReplicaQueueEntry,
    ) -> None:
        raise NotImplementedError

    async def on_claimed(
        self,
        entry: ReplicaQueueEntry,
    ) -> None:
        raise NotImplementedError

    async def on_execution_finished(
        self,
        entry: ReplicaQueueEntry,
    ) -> None:
        raise NotImplementedError

    async def on_logically_discarded(
        self,
        entry: ReplicaQueueEntry,
    ) -> None:
        raise NotImplementedError

    async def snapshot(self) -> ReplicaQueueLoad:
        raise NotImplementedError


def lease_is_live(
    state: RequestExecutionState,
    *,
    now: float,
) -> bool:
    return (
        state.lease_id is not None
        and state.lease_expires_at_monotonic is not None
        and state.lease_expires_at_monotonic > now
    )


# Interview scenarios:
#
# 1. R5 owns Req1; R1 parks. R5 releases -> R1 poller requeues exactly once.
# 2. Poller reads PENDING; R3 claims before R1's requeue -> safe claim loss.
# 3. phase 0 -> phase 1 while phase-0 parked entry exists -> discard.
# 4. owner lease expires before streaming -> parked sibling requeues for takeover.
# 5. request enters STREAMING -> V1 does not let sibling take over.
# 6. 100K parked requests -> explain why one task per request is a poor design.
