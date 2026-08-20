"""
Staff+ ML Infra Interview Exercise
==================================

Scenario 1: Replica Scheduling State Cache

Latest architecture alignment
-----------------------------
The old exercise used Planner-local PendingAssignment state. That is obsolete.

The locked design uses shared per-replica router queues, so the Planner's
effective load is:

    EffectiveLoad(replica)
      = EngineReportedLoad
      + RouterQueueLoad
      + ExecutionerInflightLoad

This cache combines versioned backend telemetry with versioned router-owned
queue/executioner load. It does NOT store request ownership; request-scoped
phase/lease/owner state belongs to RequestExecutionRegistry.

Problem Statement
-----------------
Build an in-process, concurrency-safe cache that:

1. Accepts asynchronous backend model-server snapshots.
2. Rejects stale replica-process updates using epoch + sequence fencing.
3. Accepts router queue/executioner load snapshots from the shared queue layer.
4. Detects stale backend/router state independently.
5. Produces a cheap scorer-facing EffectiveReplicaState.
6. Removes a replica cleanly when discovery says it disappeared.

Staff+ focus
------------
- Multiple independently versioned state sources.
- Immutable read snapshots.
- Hot read path vs background writes.
- Staleness semantics.
- No accidental double counting.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional


class AdmissionState(str, Enum):
    OPEN = "OPEN"
    PRESSURED = "PRESSURED"
    THROTTLED = "THROTTLED"
    REJECTING = "REJECTING"
    UNKNOWN = "UNKNOWN"


class Freshness(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class ModelRevisionKey:
    model_id: str
    model_revision: str

    def stable_key(self) -> str:
        return f"{self.model_id}|{self.model_revision}"


@dataclass(frozen=True, slots=True)
class ModelDeploymentKey:
    model: ModelRevisionKey
    deployment_id: str
    worker_role: str = "unified"

    def stable_key(self) -> str:
        return (
            f"{self.model.stable_key()}|{self.deployment_id}|{self.worker_role}"
        )


@dataclass(frozen=True, slots=True)
class ReplicaKey:
    deployment: ModelDeploymentKey
    replica_id: str


@dataclass(frozen=True, slots=True)
class BackendReplicaState:
    """
    Dynamic state reported by the model server / telemetry pipeline.
    """

    key: ReplicaKey

    replica_epoch: int
    sequence: int
    observed_at_monotonic: float

    queued_requests: int = 0
    running_requests: int = 0

    queued_prefill_tokens: int = 0
    active_prefill_tokens: int = 0

    active_decode_sequences: int = 0
    estimated_remaining_decode_tokens: int = 0

    admission_state: AdmissionState = AdmissionState.UNKNOWN

    kv_cache_utilization: float = 0.0
    available_kv_blocks: int = 0

    ewma_ttft_ms: Optional[float] = None
    ewma_tpot_ms: Optional[float] = None
    prefill_tokens_per_sec: Optional[float] = None
    decode_tokens_per_sec: Optional[float] = None


@dataclass(frozen=True, slots=True)
class RouterExecutionLoad:
    """
    Router-owned aggregate state for one replica.

    This comes from the shared replica-queue / Executioner subsystem rather than
    Planner-local provisional assignments.
    """

    key: ReplicaKey

    # Versioned independently from backend telemetry.
    sequence: int
    observed_at_monotonic: float

    router_queued_requests: int = 0
    router_queued_prefill_tokens: int = 0
    router_queued_decode_tokens: int = 0

    executioner_inflight_requests: int = 0
    executioner_inflight_prefill_tokens: int = 0
    executioner_inflight_decode_tokens: int = 0


@dataclass(frozen=True, slots=True)
class EffectiveReplicaState:
    backend: BackendReplicaState
    router: Optional[RouterExecutionLoad]

    backend_freshness: Freshness
    router_freshness: Freshness

    @property
    def effective_queued_prefill_tokens(self) -> int:
        router_queued = (
            self.router.router_queued_prefill_tokens
            if self.router is not None
            else 0
        )
        inflight = (
            self.router.executioner_inflight_prefill_tokens
            if self.router is not None
            else 0
        )
        return (
            self.backend.queued_prefill_tokens
            + router_queued
            + inflight
        )

    @property
    def effective_decode_work_tokens(self) -> int:
        router_queued = (
            self.router.router_queued_decode_tokens
            if self.router is not None
            else 0
        )
        inflight = (
            self.router.executioner_inflight_decode_tokens
            if self.router is not None
            else 0
        )
        return (
            self.backend.estimated_remaining_decode_tokens
            + router_queued
            + inflight
        )


class UpdateDecision(str, Enum):
    APPLIED = "APPLIED"
    DUPLICATE_OR_OUT_OF_ORDER = "DUPLICATE_OR_OUT_OF_ORDER"
    OLD_EPOCH = "OLD_EPOCH"


class ReplicaStateCache:
    """
    MAIN CLASS TO IMPLEMENT.

    Backend telemetry and router queue state have different producers and clocks,
    so keep their version/freshness handling separate.

    V1 skeleton uses one asyncio.Lock for clarity. In a production discussion,
    consider:
      - immutable model snapshots
      - per-model lock striping
      - copy-on-write maps
      - single-writer materializer + lock-free-ish readers
    """

    def __init__(
        self,
        *,
        backend_stale_after_seconds: float = 1.0,
        router_stale_after_seconds: float = 0.250,
    ) -> None:
        self._backend: Dict[ReplicaKey, BackendReplicaState] = {}
        self._router: Dict[ReplicaKey, RouterExecutionLoad] = {}

        self._backend_stale_after_seconds = backend_stale_after_seconds
        self._router_stale_after_seconds = router_stale_after_seconds

        self._lock = asyncio.Lock()

    async def apply_backend_update(
        self,
        state: BackendReplicaState,
    ) -> UpdateDecision:
        """
        Required semantics:
        - first update -> accept
        - incoming epoch < current epoch -> OLD_EPOCH
        - incoming epoch > current epoch -> replace
        - same epoch and sequence <= current sequence -> reject
        - otherwise accept

        When epoch advances, decide whether router load should also be cleared or
        whether queue identity is independent of model-server process identity.
        Explain the boundary.
        """
        raise NotImplementedError

    async def apply_router_load_update(
        self,
        state: RouterExecutionLoad,
    ) -> UpdateDecision:
        """
        Router queue state does not use the model-server replica epoch.

        Required semantics:
        - first update -> accept
        - sequence <= current sequence -> reject
        - otherwise accept

        Staff+ discussion:
        If the queue service itself restarts, would you add a queue_epoch?
        """
        raise NotImplementedError

    async def get_effective_state(
        self,
        key: ReplicaKey,
        *,
        now: Optional[float] = None,
    ) -> Optional[EffectiveReplicaState]:
        """
        Return the scorer-facing state.

        - backend state is required; without it return None
        - compute backend/router freshness independently
        - missing router state is allowed but visible as MISSING
        - do not mutate shared structures on read
        """
        raise NotImplementedError

    async def get_model_snapshot(
        self,
        model: ModelRevisionKey,
        *,
        now: Optional[float] = None,
    ) -> Dict[ReplicaKey, EffectiveReplicaState]:
        """
        Return all deployment/replica states for one model revision.

        Important latest-design point:
        one model revision can span multiple compatible deployments / ServingProfiles.
        """
        raise NotImplementedError

    async def remove_replica(self, key: ReplicaKey) -> None:
        """
        Called after discovery removes a replica.

        Remove both backend and router aggregate state for that replica.
        """
        raise NotImplementedError

    async def replace_backend_model_snapshot(
        self,
        model: ModelRevisionKey,
        states: Iterable[BackendReplicaState],
    ) -> None:
        """
        Bootstrap/recovery helper.

        Discuss snapshot-version / live-stream catch-up so updates are not lost
        between snapshot load and subscription.
        """
        raise NotImplementedError


def _freshness(
    observed_at: Optional[float],
    *,
    now: float,
    stale_after_seconds: float,
) -> Freshness:
    if observed_at is None:
        return Freshness.MISSING

    if now - observed_at > stale_after_seconds:
        return Freshness.STALE

    return Freshness.FRESH


# Interview scenarios:
#
# 1. Backend epoch 12 seq 100 arrives, then epoch 11 seq 1000. Reject old epoch.
# 2. Router queue state is fresh but backend telemetry is stale.
# 3. Same model revision has H100/vLLM and H100/TRT-LLM deployments.
#    get_model_snapshot() must expose both.
# 4. Explain why RequestExecutionRegistry state does not belong in this cache.
# 5. Explain why physical queue depth may differ from logical active queue load.
