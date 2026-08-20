# Component — ReplicaExecutioner

## Responsibility

Consume one replica's FIFO candidate queue, acquire atomic request ownership, orchestrate context materialization for the claimed destination, execute the request, and park/requeue sibling candidates safely.

Several Executioner workers may consume one logical replica queue.

## Input — `ReplicaQueueEntry`

All fields are defined in `12_request_queue_subsystem.md`.

The Executioner resolves `replica_execution_plan_ref` to the immutable `ReplicaExecutionPlan` selected by ReplicaScheduler.

## Processing state machine

```text
READY queue entry
    |
    v
TryClaim(request_id, phase, destination)
    |
    +-- STALE_PHASE / TERMINAL -> discard
    |
    +-- OWNED_BY_OTHER -> PARKED
    |
    +-- CLAIMED
          |
          v
      MATERIALIZING_CONTEXT
          |
          v
 ContextMaterializationCoordinator
          |
          +-- all plans failed -> release_for_retry -> PENDING
          |
          +-- success
                 |
                 v
         EXECUTING_PRE_STREAM
                 |
                 +-- retryable pre-stream failure -> release_for_retry
                 |
                 +-- first visible token -> STREAMING
                 |
                 v
          generation completed
                 |
        +--------+---------+
        |                  |
     stateless          stateful
        |                  |
        |          COMMITTING_SESSION
        |                  |
        |            SessionCommitter
        |                  |
        +--------+---------+
                 |
                 v
              SUCCESS
```

## Input dependencies

```text
RequestExecutionRegistry client
RequestQueueSubsystem client
ContextMaterializationCoordinator
ModelServerExecution client
SessionCommitter client
request/response payload store
```

## Output / side effects

ReplicaExecutioner has no routing-plan output. It causes authoritative transitions:

```text
RequestExecutionRegistry state changes
RequestQueue logical state changes
Model server execution
Context materialization observations
Execution observations
response publication
```

## Parked entry behavior

When another destination owns a live lease:

```text
READY -> PARKED
```

A shared poller/watch reconciles parked entries:

```text
request terminal          -> discard
phase changed              -> discard stale
request PENDING            -> READY/requeue
owner lease expired        -> READY/requeue and compete via TryClaim
owner active pre-stream    -> remain PARKED
STREAMING                  -> remain blocked until terminal in base V5
```

Do not create one timer/task per parked request at high cardinality; use a shared batched poll/watch mechanism.

## Context-materialization failure

The Executioner does **not** release ownership after the first transfer failure. The ContextMaterializationCoordinator first walks the destination's entire ordered fallback chain.

Only `ALL_PLANS_FAILED` causes release to `PENDING`.

## Model-server backpressure

A pre-stream `THROTTLE` / `OVER_CAPACITY` response is retryable:

```text
release lease -> PENDING
```

Parked sibling destination candidates may then compete.

## Streaming boundary

Once client-visible streaming begins, base V5 does not transparently move the request to another replica. Mid-stream recovery and hedged execution are separate advanced execution policies.

## Stateful completion rule

For a session-driven request, the Executioner must not mark platform `SUCCESS` merely because model generation completed. It first transitions the registry to `COMMITTING_SESSION`, invokes `SessionCommitter`, and marks `SUCCESS` only after the durable turn commits. A terminal commit failure becomes a request/session consistency failure.
