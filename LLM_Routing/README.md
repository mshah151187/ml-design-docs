# LLM Routing Platform — V5 Canonical Design Documents

**Status:** V5 locked architecture  
**Theme:** Stateful/session-aware inference + SLO/cost-aware replica scheduling + queue-arbitrated execution + first-class cross-replica KV transfer.

V5 extends V4 with a correctness-level session model. A conversational request may contain only `session_id + new_turn`; the platform resolves durable conversation state, creates an immutable logical context snapshot, selects a model, materializes exact target-model tokens, chooses a destination replica and context-materialization strategy, and commits the resulting turn only after successful execution.

The central V5 invariant is:

```text
Durable Session State     = correctness state
Target-Model Token Artifact = reconstructible model-specific state
KV Cache                  = ephemeral execution optimization
```

## Canonical request path

```text
Incoming Request
    -> RequestNormalizer
    -> ContextInitializer
         -> SessionStore              (stateful requests)
    -> QualityModelSelector           (Phase 2)
    -> TargetModelContextMaterializer
    -> Planner
         -> ReplicaDiscovery + state/KV lookups
    -> SLOCostAwareReplicaScheduler
    -> ExecutionCoordinator
    -> RequestQueueSubsystem
    -> ReplicaExecutioner
         -> RequestExecutionRegistry
         -> ContextMaterializationCoordinator
              -> ModelServerContextAPI
              -> KVTransferService
    -> ModelServer execution
    -> SessionCommitter               (stateful successful requests)
    -> Feedback / Performance Pipeline
```

## Component documents

1. `00_v5_end_to_end_architecture.md`
2. `01_request_normalizer.md`
3. `02_session_store.md`
4. `03_context_initializer.md`
5. `04_phase_2_quality_model_selection.md`
6. `05_target_model_context_materializer.md`
7. `06_replica_discovery.md`
8. `06a_serving_metadata_registry.md`
9. `06b_precise_kv_index.md`
10. `07_planner.md`
11. `08_replica_scheduler_v5.md`
12. `09_context_materialization_coordinator.md`
13. `10_kv_transfer_service.md`
14. `11_execution_coordinator.md`
15. `12_request_queue_subsystem.md`
16. `13_replica_executioner.md`
17. `14_request_execution_registry.md`
18. `15_model_server_context_api.md`
19. `16_session_committer.md`
20. `17_feedback_performance_pipeline.md`
21. `18_subsystems_v5.md`
22. `llm_routing_phase_3_replica_scheduler_strategy_layers.md` — copied byte-for-byte unchanged from V4.

The component documents are intentionally contract-oriented so one component can be redesigned later without rewriting the entire architecture.
