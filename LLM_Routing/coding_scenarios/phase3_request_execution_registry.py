"""
Staff+ ML Infra Interview Exercise
==================================

Scenario 5: Lease-Based RequestExecutionRegistry

The registry is authoritative for REQUEST-SCOPED execution coordination.

It stores:
    phase
    state
    owner
    lease
    attempt count
    deadlines
    version

It does NOT store:
    Top-K candidate map
    replica scheduling state
    queue load
    ServingProfile

Those belong elsewhere.

The backing store is abstracted as FakeRedisClient.atomic_update() so the
interview tests state-machine reasoning, not Redis command memorization.
"""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Dict, Optional, TypeVar


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


RECOVERABLE_PRE_STREAM_STATES = {
    ExecutionState.CLAIMED,
    ExecutionState.EXECUTING_PRE_STREAM,
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

    attempt_count: int = 0

    admission_deadline_monotonic: Optional[float] = None
    request_deadline_monotonic: Optional[float] = None

    version: int = 0
    updated_at_monotonic: float = 0.0


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


T = TypeVar("T")


class FakeRedisClient:
    """
    GIVEN SUPPORTING COMPONENT.

    `atomic_update()` represents the capability needed from a distributed
    coordination store: conditional read-modify-write on one request key.

    Do not infer a specific production Redis API from this fake.
    """

    def __init__(self) -> None:
        self._data: Dict[str, RequestExecutionState] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        key: str,
    ) -> Optional[RequestExecutionState]:
        async with self._lock:
            return copy.deepcopy(self._data.get(key))

    async def put_if_absent(
        self,
        key: str,
        value: RequestExecutionState,
    ) -> bool:
        async with self._lock:
            if key in self._data:
                return False

            self._data[key] = copy.deepcopy(value)
            return True

    async def atomic_update(
        self,
        key: str,
        update_fn: Callable[
            [Optional[RequestExecutionState]],
            tuple[Optional[RequestExecutionState], T],
        ],
    ) -> T:
        async with self._lock:
            current = copy.deepcopy(self._data.get(key))
            new_value, result = update_fn(current)

            if new_value is not None:
                self._data[key] = copy.deepcopy(new_value)

            return result


class RequestExecutionRegistry:
    """
    MAIN CLASS TO IMPLEMENT.

    Never implement ownership transitions as separate GET + SET operations.
    """

    def __init__(
        self,
        redis: FakeRedisClient,
        *,
        clock: Callable[[], float] = time.monotonic,
        lease_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._redis = redis
        self._clock = clock
        self._lease_id_factory = lease_id_factory

    @staticmethod
    def _key(request_id: str) -> str:
        return f"request-execution:{request_id}"

    async def create_request(
        self,
        *,
        request_id: str,
        initial_phase: int,
        admission_deadline_monotonic: Optional[float],
        request_deadline_monotonic: Optional[float],
    ) -> bool:
        """
        Create:
            phase=initial_phase
            state=PENDING
            no owner/lease

        Return False if key already exists.
        """
        raise NotImplementedError

    async def get_state(
        self,
        request_id: str,
    ) -> Optional[RequestExecutionState]:
        return await self._redis.get(self._key(request_id))

    async def try_claim(
        self,
        *,
        request_id: str,
        phase: int,
        replica_id: str,
        executioner_id: str,
        lease_ttl_seconds: float,
    ) -> ClaimResult:
        """
        Atomic ownership acquisition.

        Rules:
        - missing -> MISSING
        - wrong phase -> STALE_PHASE
        - terminal -> TERMINAL
        - STREAMING -> STREAMING; never takeover in V1
        - PENDING -> create new lease and CLAIMED
        - same live owner identity -> idempotent
        - another live owner -> OWNED_BY_OTHER
        - expired owner in CLAIMED/EXECUTING_PRE_STREAM:
            allow fenced takeover with a NEW lease_id

        The stale old owner must later fail every mutation because its lease_id
        no longer matches.
        """
        raise NotImplementedError

    async def renew_lease(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
        lease_ttl_seconds: float,
    ) -> TransitionResult:
        """
        Only current, still-live owner can renew.

        Do not resurrect a lease after it expired; another Executioner may have
        already become eligible to take over.
        """
        raise NotImplementedError

    async def mark_executing(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        """
        CLAIMED -> EXECUTING_PRE_STREAM

        Identical retry under same lease can be ALREADY_APPLIED.
        """
        raise NotImplementedError

    async def release_for_retry(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        """
        Recoverable pre-stream state -> PENDING.

        Clear owner and lease.

        This enables parked same-phase sibling candidates to compete again.
        """
        raise NotImplementedError

    async def mark_streaming(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        """
        EXECUTING_PRE_STREAM -> STREAMING.

        Once visible output begins, V1 does not transparently fail over.
        """
        raise NotImplementedError

    async def mark_success(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: str,
    ) -> TransitionResult:
        """
        Mark SUCCESS.

        Permit success from STREAMING and, for non-streaming APIs,
        EXECUTING_PRE_STREAM if you choose that policy explicitly.
        """
        raise NotImplementedError

    async def mark_terminal_failure(
        self,
        *,
        request_id: str,
        phase: int,
        lease_id: Optional[str],
        terminal_state: ExecutionState,
    ) -> TransitionResult:
        """
        terminal_state must be one of FAILED/CANCELLED/REQUEST_INVALID.

        Define which terminal transitions require current lease ownership.
        """
        raise NotImplementedError

    async def advance_phase(
        self,
        *,
        request_id: str,
        expected_phase: int,
        next_phase: int,
        new_admission_deadline_monotonic: Optional[float],
    ) -> TransitionResult:
        """
        ExecutionCoordinator transition.

        Allowed only when:
            current phase == expected_phase
            current state == PENDING
            request is not terminal

        Reject CLAIMED / EXECUTING_PRE_STREAM / STREAMING:
        V1 does not hedge active execution.

        This must race atomically with try_claim().
        """
        raise NotImplementedError

    async def cancel_request(
        self,
        *,
        request_id: str,
    ) -> TransitionResult:
        """
        Cancel non-terminal request.

        Discuss how active backend execution is notified after registry state
        changes.
        """
        raise NotImplementedError


def lease_is_live(
    state: RequestExecutionState,
    *,
    now: float,
) -> bool:
    """
    True only if a lease exists and expiry is strictly in the future.
    """
    raise NotImplementedError


def next_version(
    state: RequestExecutionState,
    *,
    now: float,
    **changes: object,
) -> RequestExecutionState:
    """
    Helper around dataclasses.replace:
      version += 1
      updated_at_monotonic = now
    """
    raise NotImplementedError


# Interview races:
#
# 1. R1 and R3 try_claim simultaneously -> one live lease.
# 2. try_claim(phase=0) races advance_phase(0->1) -> one atomic winner.
# 3. E1 lease L1 expires; E2 obtains L2; E1 mark_success(L1) must fail.
# 4. pre-stream failure release -> PENDING -> parked sibling can claim.
# 5. STREAMING -> no takeover in V1.
# 6. old phase-0 queue entry after phase=1 -> STALE_PHASE.
