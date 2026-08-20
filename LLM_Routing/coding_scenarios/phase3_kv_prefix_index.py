"""
Staff+ ML Infra Interview Exercise
==================================

Scenario 4: Sharded Precise KV Prefix Index

Latest architecture alignment
-----------------------------
One quality-selected model revision may have heterogeneous ServingProfiles.

A canonical KV namespace should therefore represent CACHE COMPATIBILITY, not
blindly equal one deployment_id.

Examples of namespace-affecting properties:
- model revision
- tokenizer/template revision
- adapter
- KV block size
- canonical hash schema

Different deployments may share a namespace if these semantics are compatible.
If vLLM/TRT-LLM configurations use incompatible block semantics, they use
different kv_compatibility_id / namespaces.

Problem Statement
-----------------
Implement the single-shard domain model behind a distributed precise KV index.

Maintain:
1. request_block_key -> replica residency
2. (replica, epoch, engine_block_key) -> request_block_key
3. (replica, epoch) -> request keys
4. replica epoch/sequence cursor/confidence

Support:
- BlockStored / BlockRemoved / BlockMoved / ClearReplica
- gap detection
- whole-replica cleanup
- longest consecutive prefix lookup
- prefix-branch sharding
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple


class CacheTier(str, Enum):
    GPU = "GPU"
    CPU = "CPU"
    DISK = "DISK"
    SHARED = "SHARED"


class KVEventType(str, Enum):
    BLOCK_STORED = "BLOCK_STORED"
    BLOCK_REMOVED = "BLOCK_REMOVED"
    BLOCK_MOVED = "BLOCK_MOVED"
    CLEAR_REPLICA = "CLEAR_REPLICA"


class ReplicaKVConfidence(str, Enum):
    TRUSTED = "TRUSTED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class ModelKVNamespace:
    model_id: str
    model_revision: str

    # Explicit cache-compatibility generation. This can separate incompatible
    # serving-engine/block layouts without forcing deployment_id into the key.
    kv_compatibility_id: str

    block_size_tokens: int
    hash_schema_version: str

    adapter_id: Optional[str] = None
    tokenizer_template_version: Optional[str] = None

    def stable_bytes(self) -> bytes:
        return (
            f"{self.model_id}|{self.model_revision}|"
            f"{self.kv_compatibility_id}|{self.block_size_tokens}|"
            f"{self.hash_schema_version}|{self.adapter_id}|"
            f"{self.tokenizer_template_version}"
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class KVBlockKey:
    value: bytes


@dataclass(frozen=True, slots=True)
class EngineBlockRef:
    replica_id: str
    replica_epoch: int
    engine_block_key: str


@dataclass(frozen=True, slots=True)
class Residency:
    replica_id: str
    replica_epoch: int
    tier: CacheTier


@dataclass(frozen=True, slots=True)
class KVEvent:
    namespace: ModelKVNamespace

    replica_id: str
    replica_epoch: int
    sequence: int

    event_type: KVEventType

    engine_block_key: Optional[str] = None
    request_block_key: Optional[KVBlockKey] = None
    new_tier: Optional[CacheTier] = None


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    replica_id: str
    matched_blocks: int
    matched_tokens: int
    best_tier: Optional[CacheTier]
    confidence: ReplicaKVConfidence


class EventApplyResult(str, Enum):
    APPLIED = "APPLIED"
    DUPLICATE_OR_OLD = "DUPLICATE_OR_OLD"
    OLD_EPOCH = "OLD_EPOCH"
    GAP_DETECTED = "GAP_DETECTED"


@dataclass(slots=True)
class _ReplicaCursor:
    epoch: int
    last_sequence: int
    confidence: ReplicaKVConfidence = ReplicaKVConfidence.TRUSTED


class CanonicalBlockHasher:
    """
    Canonical request-side block hashing.

    Engine internal block IDs are intentionally NOT exposed to the Planner.
    """

    def __init__(self, namespace: ModelKVNamespace) -> None:
        self._namespace = namespace

    def block_keys(
        self,
        token_ids: Sequence[int],
    ) -> Tuple[KVBlockKey, ...]:
        """
        Split exact target-model token IDs into fixed-size blocks and compute
        chained prefix hashes.

        Suggested:
            prev = H(namespace)
            K_i = H(namespace || prev || encoded_block_i)

        Chaining makes block identity prefix-position-sensitive.
        """
        raise NotImplementedError

    @staticmethod
    def _encode_token_block(
        token_ids: Sequence[int],
    ) -> bytes:
        """
        Deterministically encode token IDs.

        Suggested interview implementation:
            b"".join(struct.pack(">I", token_id) ...)
        """
        raise NotImplementedError


class KVPrefixIndex:
    """
    MAIN CLASS TO IMPLEMENT.

    One logical shard for one ModelKVNamespace.
    """

    def __init__(self, namespace: ModelKVNamespace) -> None:
        self._namespace = namespace

        self._residency: Dict[
            KVBlockKey,
            Dict[str, Residency],
        ] = {}

        self._engine_to_request: Dict[
            EngineBlockRef,
            KVBlockKey,
        ] = {}

        self._replica_to_blocks: Dict[
            tuple[str, int],
            Set[KVBlockKey],
        ] = {}

        self._cursor: Dict[str, _ReplicaCursor] = {}

    def apply_event(
        self,
        event: KVEvent,
    ) -> EventApplyResult:
        """
        Required ordering:
        - namespace must match
        - older epoch -> reject
        - newer epoch -> clear old replica epoch and reset cursor
        - duplicate/old sequence -> ignore
        - sequence gap -> mark UNCERTAIN
        - otherwise apply mutation

        Explain whether you pause after a gap or continue applying while UNCERTAIN.
        """
        raise NotImplementedError

    def apply_events(
        self,
        events: Iterable[KVEvent],
    ) -> list[EventApplyResult]:
        """Batch mutation API."""
        raise NotImplementedError

    def _apply_block_stored(
        self,
        event: KVEvent,
    ) -> None:
        """
        Update:
          request key -> residency
          engine ref -> request key
          replica reverse membership
        """
        raise NotImplementedError

    def _apply_block_removed(
        self,
        event: KVEvent,
    ) -> None:
        """
        Resolve canonical key through engine reverse map, then remove all three
        relationships consistently.
        """
        raise NotImplementedError

    def _apply_block_moved(
        self,
        event: KVEvent,
    ) -> None:
        """Update tier while preserving locality."""
        raise NotImplementedError

    def clear_replica(
        self,
        replica_id: str,
        replica_epoch: int,
    ) -> int:
        """
        Use replica reverse map for O(number-of-blocks-on-replica) cleanup.
        """
        raise NotImplementedError

    def mark_replica_uncertain(
        self,
        replica_id: str,
    ) -> None:
        raise NotImplementedError

    def restore_replica_confidence(
        self,
        *,
        replica_id: str,
        replica_epoch: int,
        last_sequence: int,
    ) -> None:
        """
        Called only after replay/snapshot reconciliation rebuilds trusted state.
        """
        raise NotImplementedError

    def find_prefix_matches(
        self,
        request_block_keys: Sequence[KVBlockKey],
        candidate_replicas: Iterable[str],
    ) -> Dict[str, PrefixMatch]:
        """
        Longest consecutive prefix per candidate.

        Once a candidate misses a block, it never re-enters the active set.

        Do not give precise cache credit for UNCERTAIN replicas.
        """
        raise NotImplementedError

    def memory_stats(self) -> Mapping[str, int]:
        """
        Return cardinality counters for capacity reasoning.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PrefixShardAssignment:
    namespace: ModelKVNamespace
    prefix_anchor: Tuple[KVBlockKey, ...]
    shard_id: str


class PrefixShardRouter:
    """
    Route:
        namespace + shallow prefix anchor -> one KV shard

    V1 avoids request-path scatter/gather.
    """

    def __init__(self, prefix_depth: int = 2) -> None:
        self._prefix_depth = prefix_depth
        self._shards: Tuple[str, ...] = ()

    def replace_shards(
        self,
        shard_ids: Iterable[str],
    ) -> None:
        raise NotImplementedError

    def assign(
        self,
        namespace: ModelKVNamespace,
        block_keys: Sequence[KVBlockKey],
    ) -> PrefixShardAssignment:
        """
        Use deterministic hashing/rendezvous; never Python's randomized hash().
        """
        raise NotImplementedError


# Interview scenarios:
#
# 1. Same model revision, two deployments, same kv_compatibility_id.
#    They can participate in one canonical namespace.
# 2. Same model revision but different block sizes -> separate namespaces.
# 3. Missing sequence 103 makes R7 UNCERTAIN; R7 gets no precise cache credit.
# 4. Replica restarts with epoch 15; all epoch-14 memberships must be removed.
