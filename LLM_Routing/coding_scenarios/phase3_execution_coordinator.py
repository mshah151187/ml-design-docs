"""
Staff+ ML Infra Interview Exercise
==================================

Scenario 3: Phase-Based ExecutionCoordinator

This file SUPERSEDES the old phase3_execution_dispatcher.py exercise.

Latest architecture
-------------------
The Planner returns Top-K=3 candidates for each quality-approved model phase.

The ExecutionCoordinator does following:

1. creates RequestExecutionState(phase=0, PENDING)
2. publishes Top-K queue entries for the active phase
3. observes RequestExecutionRegistry
4. advances to the next model phase only if the current phase remains PENDING
   past its admission/start budget
5. stops when request becomes terminal

Current phase is authoritative in RequestExecutionRegistry.

V1 explicitly does NOT implement hedged inference.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Protocol, Tuple


TOP_K = 3


class ModelPlanStatus(str, Enum):
    PLANNING = "PLANNING"
    READY = "READY"
    FAILED = "FAILED"


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
class ModelRevisionKey:
    model_id: str
    model_revision: str


@dataclass(frozen=True, slots=True)
class ReplicaCandidate:
    replica_id: str
    deployment_id: str
    serving_profile_id: str

    score: float
    planned_at_monotonic: float
    state_version: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ModelReplicaPlan:
    phase: int
    model: ModelRevisionKey
    candidates: Tuple[ReplicaCandidate, ...]

    def top_k(self, k: int = TOP_K) -> Tuple[ReplicaCandidate, ...]:
        return self.candidates[:k]


@dataclass(slots=True)
class ProgressivePhasePlan:
    """
    One quality-approved model phase.

    phase=0: primary
    phase=1+: ordered fallbacks
    """

    phase: int
    model: ModelRevisionKey

    status: ModelPlanStatus = ModelPlanStatus.PLANNING
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)

    plan: Optional[ModelReplicaPlan] = None
    error: Optional[BaseException] = None

    def set_ready(self, plan: ModelReplicaPlan) -> None:
        if plan.phase != self.phase:
            raise ValueError("plan phase mismatch")
        if plan.model != self.model:
            raise ValueError("plan model mismatch")

        self.plan = plan
        self.status = ModelPlanStatus.READY
        self.ready_event.set()

    def set_failed(self, error: BaseException) -> None:
        self.error = error
        self.status = ModelPlanStatus.FAILED
        self.ready_event.set()


@dataclass(slots=True)
class ExecutionRouteContext:
    """
    Progressively materialized plans.

    IMPORTANT:
    There is intentionally NO mutable `current_model_index` here.

    `RequestExecutionRegistry.phase` is authoritative.

    phase_plans[0] is primary; phase_plans[1:] are fallbacks.
    """

    request_id: str
    phase_plans: List[ProgressivePhasePlan]


@dataclass(frozen=True, slots=True)
class RequestExecutionState:
    request_id: str
    phase: int
    state: ExecutionState

    owner_replica: Optional[str] = None
    lease_id: Optional[str] = None

    admission_deadline_monotonic: Optional[float] = None
    request_deadline_monotonic: Optional[float] = None

    version: int = 0


@dataclass(frozen=True, slots=True)
class ReplicaQueueEntry:
    request_id: str
    phase: int

    replica_id: str
    deployment_id: str
    candidate_rank: int

    enqueue_time_monotonic: float
    admission_deadline_monotonic: Optional[float]

    # Compact work estimates used by queue load tracking.
    estimated_prefill_tokens: int
    estimated_decode_tokens: int


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


class ExecutionRegistryClient(Protocol):
    async def create_request(
        self,
        *,
        request_id: str,
        initial_phase: int,
        admission_deadline_monotonic: Optional[float],
        request_deadline_monotonic: Optional[float],
    ) -> bool:
        ...

    async def get_state(
        self,
        request_id: str,
    ) -> Optional[RequestExecutionState]:
        ...

    async def advance_phase(
        self,
        *,
        request_id: str,
        expected_phase: int,
        next_phase: int,
        new_admission_deadline_monotonic: Optional[float],
    ) -> TransitionResult:
        ...


class ReplicaQueuePublisher(Protocol):
    async def enqueue(
        self,
        entry: ReplicaQueueEntry,
    ) -> None:
        ...


class RequestPayloadMetadata(Protocol):
    @property
    def estimated_prefill_tokens(self) -> int:
        ...

    @property
    def estimated_decode_tokens(self) -> int:
        ...


@dataclass(frozen=True, slots=True)
class ExecutionCoordinatorPolicy:
    top_k: int = TOP_K

    # Default phase admission budget. A real system may choose this from request
    # SLO/model profile rather than a single constant.
    phase_admission_budget_seconds: float = 0.050

    # Polling abstraction for exercise. Production could use notifications/watch.
    registry_poll_interval_seconds: float = 0.005


class RouteExhaustedError(RuntimeError):
    pass


class RequestExecutionFailed(RuntimeError):
    pass


class ExecutionCoordinator:
    """
    MAIN CLASS TO IMPLEMENT.

    It coordinates model phases, not replicas.

    Replica-local ownership/retry happens in ReplicaExecutioner.
    """

    def __init__(
        self,
        *,
        registry: ExecutionRegistryClient,
        queue_publishers: dict[str, ReplicaQueuePublisher],
        policy: ExecutionCoordinatorPolicy = ExecutionCoordinatorPolicy(),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._queue_publishers = queue_publishers
        self._policy = policy
        self._clock = clock

    async def execute(
        self,
        *,
        context: ExecutionRouteContext,
        payload: RequestPayloadMetadata,
        request_deadline_monotonic: Optional[float],
    ) -> RequestExecutionState:
        """
        High-level V1 state machine.

        Suggested flow:
        1. validate phase_plans are contiguous: 0..N-1
        2. wait for primary plan only
        3. create registry state at phase=0 PENDING
        4. activate phase 0 by enqueuing Top-K
        5. observe registry:
             SUCCESS -> return
             terminal failure -> raise/return policy
             PENDING before admission deadline -> keep waiting
             PENDING after admission deadline -> try advance to next phase
             CLAIMED/EXECUTING/STREAMING -> do NOT hedge; keep observing
        6. when phase advances, wait for that phase's plan if still PLANNING,
           then enqueue its Top-K
        7. if no fallback remains -> RouteExhaustedError

        Crucial race:
        Executioner try_claim(phase=0) can race with coordinator
        advance_phase(0->1). Registry atomicity decides one winner.
        """
        phase_index=0
        request_id=context.request_id
        phase_plans=context.phase_plans

        while True:
            if phase_index == len(phase_plans):
                break
            model_replica_plan=await self._wait_for_phase_plan(phase_plans[phase_index])
            if phase_index==0:
                await self._registry.create_request(
                    request_id=request_id,
                    initial_phase=0,
                    admission_deadline_monotonic=self._phase_deadline(),
                    request_deadline_monotonic=request_deadline_monotonic,
                )
            await self._activate_phase(
                request_id=request_id,
                plan=model_replica_plan,
                payload=payload,
                admission_deadline_monotonic=self._phase_deadline(),
            )
            phase_advanced=False
            while not phase_advanced:
                request_execution_state=await self._observe_phase(
                    request_id=request_id,
                    expected_phase=phase_index,
                )
                if self._need_to_advance(request_execution_state):
                    phase_advanced=await self._try_advance_to_next_phase(context,request_execution_state)
                if phase_advanced:
                    phase_index=phase_index+1
                elif self._can_terminate_request(request_execution_state):
                    return request_execution_state
            
        raise RouteExhaustedError()

    def _can_terminate_request(self,request_execution_state: RequestExecutionState):
            if request_execution_state.state in [ExecutionState.CLAIMED,ExecutionState.EXECUTING_PRE_STREAM,ExecutionState.STREAMING,ExecutionState.SUCCESS]:
                return True
            return False

    def _need_to_advance(self, request_execution_state: RequestExecutionState) -> bool:
        state = request_execution_state.state
        if state in (ExecutionState.REQUEST_INVALID, ExecutionState.CANCELLED):
            raise RequestExecutionFailed(state)
        return state in (ExecutionState.PENDING, ExecutionState.FAILED)


    async def _wait_for_phase_plan(
        self,
        phase_plan: ProgressivePhasePlan,
    ) -> ModelReplicaPlan:
        """
        If READY, return immediately.
        If PLANNING, wait for ready_event.
        If FAILED, raise controlled planning failure.

        Primary plan is critical-path; fallback plans may already be running
        concurrently.
        """
        if phase_plan.status == ModelPlanStatus.FAILED:
            raise RequestExecutionFailed(f"Model plan for phase {phase_plan.phase} is failed")
        if phase_plan.status == ModelPlanStatus.PLANNING:
            await phase_plan.ready_event.wait()
        if phase_plan.status == ModelPlanStatus.FAILED:
            raise RequestExecutionFailed(f"Model plan for phase {phase_plan.phase} is failed")
        return phase_plan.plan
        

    async def _activate_phase(
        self,
        *,
        request_id: str,
        plan: ModelReplicaPlan,
        payload: RequestPayloadMetadata,
        admission_deadline_monotonic: Optional[float],
    ) -> None:
        """
        Fan a compact request reference into Top-K=3 replica queues.

        Requirements:
        - preserve candidate rank
        - do not duplicate full request payload
        - fail explicitly if a queue publisher is missing
        - decide partial-fanout behavior if one enqueue succeeds and another fails

        Staff+ discussion:
        Would you require all 3 queue writes before considering phase activated?
        """
        request_queue_dict={}
        for rank,replica_candidate in enumerate(plan.top_k(3)):
            if replica_candidate.replica_id not in self._queue_publishers:
                raise RequestExecutionFailed(f"no queue publisher for replica {replica_candidate.replica_id}")
            queue_entry=ReplicaQueueEntry(
                request_id=request_id,
                phase=plan.phase,
                replica_id=replica_candidate.replica_id,
                deployment_id=replica_candidate.deployment_id,
                candidate_rank=rank,
                enqueue_time_monotonic=self._clock(),
                admission_deadline_monotonic=admission_deadline_monotonic,
                estimated_prefill_tokens=payload.estimated_prefill_tokens,
                estimated_decode_tokens=payload.estimated_decode_tokens,
            )

            request_queue_publishers=self._queue_publishers[replica_candidate.replica_id]
            request_queue_dict[replica_candidate.replica_id]=(queue_entry,request_queue_publishers)


        async def publish_to_replica_queue(queue_entry:ReplicaQueueEntry,request_queue_publisher:ReplicaQueuePublisher):
            await request_queue_publisher.enqueue(entry=queue_entry)

        tasks=[]
        for _, (queue_entry, request_queue_publisher) in request_queue_dict.items():
            task = asyncio.create_task(publish_to_replica_queue(queue_entry,request_queue_publisher))
            tasks.append(task)

        done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        succeeded = [t for t in done if t.exception() is None]
        if not succeeded:
            raise RequestExecutionFailed("None of the replica is executing the request")
        return

    async def _observe_phase(
        self,
        *,
        request_id: str,
        expected_phase: int,
    ) -> RequestExecutionState:
        """
        Poll/watch registry until one meaningful transition requires coordinator
        action.

        Return when:
        - terminal
        - PENDING and admission deadline expired
        - phase changed underneath us

        Do NOT return simply because CLAIMED/EXECUTING/STREAMING is observed;
        V1 does not hedge active execution.
        """
        while True:
            await asyncio.sleep(self._policy.registry_poll_interval_seconds)
            state = await self._registry.get_state(request_id)

            if state.state in (ExecutionState.CANCELLED, ExecutionState.REQUEST_INVALID):
                raise RequestExecutionFailed(state.state)

            if state.state in TERMINAL_STATES:   # SUCCESS, FAILED
                return state

            if state.state == ExecutionState.PENDING:
                deadline = state.admission_deadline_monotonic
                if deadline is not None and self._clock() >= deadline:
                    return state   # caller will try to advance phase

    async def _try_advance_to_next_phase(
        self,
        *,
        context: ExecutionRouteContext,
        current: RequestExecutionState,
    ) -> bool:
        """
        Atomically advance current.phase -> current.phase + 1 only when current
        registry state is still PENDING.

        Return True only if this coordinator won the transition.

        If an Executioner claimed concurrently, return False and continue
        observing the current active execution.
        """
        current_phase=current.phase
        next_phase=current_phase+1
        if next_phase >= len(context.phase_plans):
            raise RouteExhaustedError()

        phase_plan = context.phase_plans[next_phase]
        await self._wait_for_phase_plan(phase_plan)
        transition_result=await self._registry.advance_phase(request_id=current.request_id,
                                     expected_phase=current_phase,
                                     next_phase=next_phase,
                                     new_admission_deadline_monotonic=self._phase_deadline())

        if transition_result.status in [TransitionStatus.STALE_PHASE,TransitionStatus.INVALID_STATE,
                               TransitionStatus.TERMINAL,TransitionStatus.NOT_OWNER]:
            return False
        if transition_result.status == TransitionStatus.MISSING:
            raise RequestExecutionFailed()
        return True
        
        

    def _phase_deadline(self) -> float:
        return self._clock() + self._policy.phase_admission_budget_seconds


class ProgressivePlanPublisher:
    """
    Supporting helper used by async fallback Planner RPCs.
    """

    def __init__(self, context: ExecutionRouteContext) -> None:
        self._context = context

    def publish_plan(
        self,
        *,
        phase: int,
        plan: ModelReplicaPlan,
    ) -> None:
        self._context.phase_plans[phase].set_ready(plan)

    def publish_failure(
        self,
        *,
        phase: int,
        error: BaseException,
    ) -> None:
        self._context.phase_plans[phase].set_failed(error)


# Interview scenarios:
#
# 1. Phase 0 timer expires exactly as R3 claims. Only one atomic transition wins.
# 2. Phase 0 becomes EXECUTING at 45ms; 50ms timer must NOT start phase 1.
# 3. R3 fails pre-stream and Executioner releases to PENDING at 70ms.
#    If the phase deadline is already expired, coordinator may then advance.
# 4. Fallback plan is still PLANNING when phase 1 becomes necessary.
# 5. One of three queue publishes fails after two succeeded. Define recovery.
