# Component — ReplicaDiscovery

## Responsibility

Provide a request-path-local materialized view of all currently discoverable serving deployments/replicas for a selected model revision. Kubernetes/control-plane calls are not performed synchronously per inference request.

## Control-plane inputs

### `DeploymentRegistration`

```text
DeploymentRegistration {
    model_id: string
    model_revision: string

    deployment_id: string
    serving_profile_id: string
    kv_compatibility_id: string

    worker_role: string
    region: string
    policy_tags: set<string>
}
```

### Runtime membership updates

```text
ReplicaMembershipUpdate {
    deployment_id: string
    replica_id: string

    endpoint: string
    node_id: string?
    zone: string?
    region: string

    replica_epoch: int

    ready: bool
    draining: bool
    health: HEALTHY | DEGRADED | UNHEALTHY

    membership_sequence: int
    observed_at: timestamp
}
```

## Request-path API

### Input

```text
ReplicaDiscoveryRequest {
    model_id: string
    model_revision: string
    required_worker_role: string?
    region: string?
    policy_tags: set<string>
}
```

### Output — `ReplicaDescriptor[]`

```text
ReplicaDescriptor {
    model_id: string
    model_revision: string

    deployment_id: string
    replica_id: string
    replica_epoch: int

    endpoint: string
    worker_role: string

    serving_profile_id: string
    kv_compatibility_id: string

    region: string
    zone: string?
    node_id: string?

    ready: bool
    draining: bool
    health: HEALTHY | DEGRADED | UNHEALTHY

    discovery_version: int
    observed_at: timestamp
}
```

## Invariants

- Discovery states who exists; it does not state how busy a replica is.
- `serving_profile_id` is static/slow execution metadata.
- `kv_compatibility_id` defines whether cross-replica KV representation is compatible at the high-level contract boundary.
- A model revision can have several compatible deployments with different hardware/engines.

## Next component

The `Planner` joins discovery descriptors with dynamic state, precise KV locality, and target-model context.
