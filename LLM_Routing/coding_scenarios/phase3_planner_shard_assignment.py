"""
Staff+ ML Infra Interview Exercise
==================================

Scenario 2: Model-Revision-Affine Planner Shard Assignment

Latest architecture alignment
-----------------------------
Phase 2 selects a model revision for quality.

Phase 3 may then compare several compatible serving deployments of that revision:

    Llama-3B-v7
      -> D1: H100 + vLLM
      -> D2: H100 + TensorRT-LLM
      -> D3: A100 + vLLM

Therefore Planner ownership should normally be affine to the MODEL REVISION, not
to one individual deployment. The owning Planner group keeps all relevant
deployments/ServingProfiles/replicas warm and can score them together.

Problem Statement
-----------------
Build:

    ModelRevisionKey -> PlannerShardGroup

using deterministic placement with warm ownership handoff.

Staff+ focus
------------
- stable model ownership vs per-request instance load balancing
- rendezvous hashing
- readiness/draining
- warm-before-cutover
- ownership epoch/fencing
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True, slots=True)
class ModelRevisionKey:
    model_id: str
    model_revision: str

    def stable_key(self) -> str:
        return f"{self.model_id}|{self.model_revision}"


@dataclass(frozen=True, slots=True)
class ModelDeployment:
    """
    One execution deployment belonging to the same quality-selected model revision.
    """

    model: ModelRevisionKey
    deployment_id: str
    serving_profile_id: str
    worker_role: str = "unified"


class PlannerInstanceState(str, Enum):
    WARMING = "WARMING"
    READY = "READY"
    DRAINING = "DRAINING"
    UNHEALTHY = "UNHEALTHY"


@dataclass(frozen=True, slots=True)
class PlannerInstance:
    instance_id: str
    endpoint: str
    state: PlannerInstanceState
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class PlannerShardGroup:
    """
    Logical owner group.

    Several READY processes may serve the same group. They materialize the same
    model-revision state, including all compatible deployments.
    """

    group_id: str
    epoch: int
    instances: Tuple[PlannerInstance, ...]
    weight: float = 1.0

    def ready_instances(self) -> Tuple[PlannerInstance, ...]:
        return tuple(
            instance
            for instance in self.instances
            if instance.state == PlannerInstanceState.READY
        )


@dataclass(frozen=True, slots=True)
class PlannerAssignment:
    model: ModelRevisionKey
    group_id: str
    group_epoch: int


class NoPlannerAvailable(RuntimeError):
    pass


class RendezvousHasher:
    @staticmethod
    def score(key: str, node_id: str) -> int:
        """
        Deterministic score; higher wins.

        Do not use Python hash(), which is randomized per process.

        Suggested implementation:
            hashlib.blake2b(..., digest_size=8)
        """
        raise NotImplementedError


class PlannerShardAssigner:
    """
    MAIN CLASS TO IMPLEMENT.

    Two separate decisions:

    A. Stable model ownership:
           ModelRevisionKey -> PlannerShardGroup

    B. Per-request process selection inside owner group:
           PlannerShardGroup -> READY PlannerInstance

    Do not conflate them.
    """

    def __init__(self) -> None:
        self._groups: Dict[str, PlannerShardGroup] = {}

    def replace_groups(
        self,
        groups: Iterable[PlannerShardGroup],
    ) -> None:
        """
        Atomically replace topology.

        Production discussion:
        immutable topology snapshot swapping can keep request reads lock-light.
        """
        raise NotImplementedError

    def assign_model(
        self,
        model: ModelRevisionKey,
    ) -> PlannerAssignment:
        """
        Choose logical owner using rendezvous hashing.

        Requirements:
        - deterministic with identical topology
        - ignore groups with no READY serving instance
        - include group epoch in assignment
        - explicit NoPlannerAvailable failure
        """
        raise NotImplementedError

    def choose_instance(
        self,
        assignment: PlannerAssignment,
        *,
        request_id: str,
    ) -> PlannerInstance:
        """
        Choose one READY process inside the selected group.

        Model ownership does not change here.

        Request-ID rendezvous hashing is a good deterministic V1 choice.
        """
        raise NotImplementedError

    def route_request(
        self,
        model: ModelRevisionKey,
        *,
        request_id: str,
    ) -> tuple[PlannerAssignment, PlannerInstance]:
        raise NotImplementedError

    def rebalance_preview(
        self,
        models: Iterable[ModelRevisionKey],
        new_groups: Iterable[PlannerShardGroup],
    ) -> Dict[ModelRevisionKey, tuple[Optional[str], Optional[str]]]:
        """
        Return only ownership changes:
            model -> (old_group, new_group)

        Used by control plane before actual cutover.
        """
        raise NotImplementedError


class OwnershipCoordinator:
    """
    Supporting Level-2 control-plane skeleton.
    """

    async def warm_new_owner(
        self,
        *,
        model: ModelRevisionKey,
        new_group_id: str,
        deployments: Iterable[ModelDeployment],
    ) -> None:
        """
        Expected flow:
        1. load model revision's deployment/ServingProfile metadata
        2. bootstrap replica/backend state
        3. subscribe to discovery/load/performance/queue state
        4. obtain KV-index client handles for relevant namespaces
        5. catch up to a known stream position
        6. mark model READY in new group
        """
        raise NotImplementedError

    async def cutover(
        self,
        *,
        model: ModelRevisionKey,
        old_group_id: Optional[str],
        new_group_id: str,
        new_group_epoch: int,
    ) -> None:
        """
        Switch ownership only after warm-up.

        Explain how epoch/fencing prevents an old owner from accepting new work.
        """
        raise NotImplementedError


# Interview scenarios:
#
# 1. D1/H100-vLLM and D2/H100-TRTLLM must be scored by the same model owner.
# 2. Add a Planner group and estimate key movement under rendezvous hashing.
# 3. Remove a READY instance from a group without changing model ownership.
# 4. Explain why hot models map to a group, not a single process.
