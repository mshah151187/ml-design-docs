# Feature Registry and Metadata Store

Related: `ml_feature_pipeline.md`, `funnel_analytics_pipeline.md`, `realtime_business_events.md`

Source: Uber Engineering — Palette Meta Store (https://www.uber.com/blog/palette-meta-store-journey/)

---

## Problem

`ml_feature_pipeline.md` covers how features are computed and stored.
But it does not answer:

```
Which feature groups exist, and who owns them?
What is the schema of user:{id}:features in Redis — which fields, which types?
Which ML model depends on which feature group?
If I add a field to a FeatureGroup, who breaks?
If I delete a FeatureGroup, which models are affected?
How does the Flink job know what to write, and the model server know what to read?
```

These questions require a **Feature Registry**: a centralized metadata store that acts
as the source of truth for all feature definitions, schemas, storage mappings, and
ownership. Without it, each team maintains their own schema constants in code →
schema drift, silent breakage, no audit trail.

---

## What Went Wrong Without a Registry (Uber 2021 Incident)

```
Problem: A bad metadata change was pushed to OnlineServing config.
         Validation logic lived in client-side scripts.
         Not all clients picked up the latest validation rules.
         OnlineServing broke in production.

Root cause: no server-side enforcement. Validation was optional, client-controlled.
            Any team could push invalid metadata and it would reach production.

Fix: move all validation to the registry server.
     Config changes must pass server-side validation before registration.
     Invalid configs → rejected at write time, never reach serving.
```

---

## Data Model

```
FeatureGroup            Logical table. Unit of ownership and deployment.
  │                     Backed by: Hive snapshot (offline) or Cassandra/Redis (online).
  │
  ├── Feature[]         Individual columns within the group.
  │     name            "user_order_count_7d"
  │     dtype           INT64, FLOAT32, STRING, BOOL, LIST<FLOAT32>
  │     description     human-readable
  │     nullable        bool
  │     default_value   used when feature is missing at serve time
  │
  ├── Dataset           Storage location metadata.
  │     keyspace        Cassandra keyspace or Redis cluster name
  │     partition_keys  ["user_id"]         — must exist in Feature[]
  │     cluster_keys    ["date"]            — for time-partitioned groups
  │     ttl_hours       for online stores
  │
  ├── Storage           Underlying infrastructure.
  │     type            CASSANDRA | REDIS | HIVE | S3
  │     cluster_name    "cassandra-prod-us-east-1"
  │     tier            CRITICAL | STANDARD | BATCH
  │
  └── FeatureServingGroup   SLA-guaranteed serving unit.
        sla_tier            P99_10MS | P99_50MS | P99_500MS
        storage_clusters[]  list of Storage entries (primary + replicas)
        routing_map         { read_primary: ..., failover: ... }
```

**Config-as-code (3 files per FeatureGroup, stored in Git):**

```
feature_groups/
  user_behavior/
    description.json      ownership, alerting, offline serving config
    features.json         schema — Feature[] definitions (the CRD)
    OnlineServing.json    cluster, SLA tier, routing
    generate.hql          Hive query that produces the feature values (offline)
```

**description.json:**
```json
{
  "name":        "user_behavior_features",
  "owner":       "ml-personalization@uber.com",
  "team":        "Personalization",
  "alert_slack": "#ml-feature-alerts",
  "offline": {
    "hive_database":   "feature_store",
    "hive_table":      "user_behavior_features",
    "partition_col":   "ds",
    "schedule":        "0 2 * * *"
  }
}
```

**features.json (the CRD):**
```json
{
  "schema_ver": 3,
  "features": [
    { "name": "orders_last_7d",        "dtype": "INT64",   "nullable": false, "default": 0   },
    { "name": "orders_last_30d",       "dtype": "INT64",   "nullable": false, "default": 0   },
    { "name": "avg_order_value_30d",   "dtype": "FLOAT32", "nullable": true,  "default": 0.0 },
    { "name": "preferred_cuisine",     "dtype": "STRING",  "nullable": true,  "default": ""  },
    { "name": "last_search_embedding", "dtype": "LIST<FLOAT32>", "dim": 64, "nullable": true }
  ]
}
```

**OnlineServing.json:**
```json
{
  "serving_group": "user-features-critical",
  "sla_tier":      "P99_10MS",
  "primary":       "cassandra-prod-us-east-1",
  "replicas":      ["cassandra-prod-us-west-2"],
  "partition_key": "user_id",
  "ttl_hours":     168
}
```

---

## Registry Architecture

```
Git (config-as-code)
  │
  │  PR merged
  ▼
CI/CD Pipeline
  │  1. schema validation (field types, required fields)
  │  2. cross-CRD validation (dependencies exist)
  │  3. compute delta (git diff vs last registered state)
  ▼
Registry Service (gRPC API)
  │  stores canonical state
  ├── Metadata DB (MySQL/Postgres)   source of truth, all FeatureGroup records
  ├── Schema Cache (Redis)           fast reads for serving, TTL 5min
  └── Event Bus (Kafka)             publishes RegistryChangeEvent on every update
        │
        ├── Kubernetes Controller   syncs to ObjectConfig (HA config cache)
        │     → online serving reads from ObjectConfig at startup
        │
        ├── Flink Feature Jobs      reload schema on RegistryChangeEvent
        │     → validates write schema matches registry before HSET
        │
        └── Model Registry          cross-references feature deps on model deploy
```

---

## Incremental Delta Propagation

**Old approach (broken):**
```
Any config change → re-register ALL feature groups → serialize all configs →
push to all downstream consumers → 1+ hour end-to-end
```

**New approach (delta):**
```
Git diff on merge → list of changed files → only affected FeatureGroups
→ register only those → propagate only those changes → 15 minutes

Implementation:
  1. CI records last_registered_git_sha per FeatureGroup in Registry DB
  2. On merge: git diff last_registered_git_sha..HEAD
               → changed_files = [feature_groups/user_behavior/features.json, ...]
               → affected_groups = extract group name from path
  3. Registry Service processes only affected_groups, in dependency order:
       Storage → Dataset → FeatureGroup → Feature
     (must register Storage before FeatureGroup that references it)
  4. Emit RegistryChangeEvent per changed group → downstream consumers react
```

**Why dependency order matters:**
```
OnlineServing.json for group A references cluster "cassandra-prod-us-east-1".
If Storage record for that cluster doesn't exist yet → validation fails.
Order: register Storage first, then Dataset, then FeatureGroup.
This is topological sort on the dependency DAG.
```

---

## Server-Side Validation (Layers)

All validation runs in Registry Service before any config is stored:

### Layer 1 — Field-Level Validation
```
dtype is one of: INT64, FLOAT32, FLOAT64, STRING, BOOL, LIST<FLOAT32>
nullable is bool
default_value type matches dtype
feature name matches regex: [a-z][a-z0-9_]*
schema_ver is integer, monotonically increasing
```

### Layer 2 — Group-Level Validation
```
partition_keys in Dataset must all exist in Feature[] by name
cluster_keys in Dataset must all exist in Feature[] by name
no duplicate feature names within a group
if dtype=LIST<FLOAT32>, dim must be specified
ttl_hours must be > 0 for online stores, absent for offline
```

### Layer 3 — Cross-CRD Validation
```
cluster_name in OnlineServing.json must exist in Storage registry
serving_group in OnlineServing.json must exist in FeatureServingGroup registry
if FeatureGroup declares dependency on another FeatureGroup (for join keys):
  → dependent group must be registered and ACTIVE
  → partition_key of dependent must be in this group's Feature[]
```

### Layer 4 — Backward Compatibility Check
```
schema_ver must increment by exactly 1
removed fields → BLOCKED (breaking change — models may depend on them)
renamed fields → BLOCKED (same reason — use deprecation instead)
type change on existing field → BLOCKED
adding optional field → ALLOWED
adding field with default_value → ALLOWED
changing default_value → ALLOWED
```

**Deprecation pattern (never delete, mark deprecated):**
```json
{ "name": "old_order_count", "dtype": "INT64", "deprecated": true,
  "deprecated_since": "2026-06-01", "replacement": "orders_last_30d" }
```
Deprecated fields: still served (no breakage), flagged in registry UI,
removed after all consumers migrate (tracked by Model Registry).

---

## Kubernetes Controller Pattern

Online serving needs low-latency access to metadata (cluster routing, partition keys).
Fetching from Registry Service on every request = too slow.

```
Registry Service
  │  RegistryChangeEvent (Kafka)
  ▼
Kubernetes Controller (watches Kafka)
  │  on change: fetch updated FeatureGroup metadata from Registry Service
  │  serialize to protobuf
  ▼
ObjectConfig (distributed HA config cache, like etcd or ZooKeeper)
  │
  ▼
Online Serving Nodes (read from ObjectConfig at startup + watch for updates)
  → hot-reload on change (no restart), atomic reference swap
```

**Why not read from Registry Service directly at serve time:**
```
Registry Service: ~10ms read (DB + cache hop)
ObjectConfig:     ~1ms read (local cache, replicated)

Online serving SLO: p99 < 10ms total
Feature fetch from Cassandra: ~5ms
Model inference: ~3ms
→ only 2ms left for metadata lookup → must be ObjectConfig, not Registry RPC
```

---

## Migration Safety: Wrapper + Kill Switch

During the Palette re-architecture, Uber ran old and new metadata paths in parallel:

```
Metadata Wrapper
  ├── New path: Registry Service (new format)
  └── Old path: legacy fragmented metadata (old format)
  └── Kill switch: feature flag per FeatureGroup

  kill_switch = OFF  → serve from old path (default during migration)
  kill_switch = ON   → serve from new path

Shadow mode:
  Both paths run on every request.
  Compare outputs: if new_output == old_output → match logged.
  When match rate reaches 100% for a FeatureGroup → flip kill switch ON.
  If regression detected → kill switch flips back to OFF instantly (no deploy needed).
```

**Why this pattern is reusable:**
```
Any stateful migration where:
  - you can't afford downtime
  - correctness is hard to verify in staging
  → run both paths, compare, build confidence, flip atomically
This is the same principle as blue-green deployment but at the data layer.
```

---

## Connection to ml_feature_pipeline.md

The registry is the contract between producers (Flink jobs) and consumers (model servers):

```
Producer (Flink Feature Job):
  On startup → fetch FeatureGroup schema from Registry
  On RegistryChangeEvent → reload schema
  Before every HSET → validate field names + types match registered schema
  If mismatch → route to DLQ, alert (never write invalid data silently)

Consumer (Model Serving):
  On model deploy → Model Registry checks: does this model's feature list
                    match current FeatureGroup schema in Feature Registry?
  If field missing or type changed → deploy blocked
  At serve time → reads feature schema from ObjectConfig (< 1ms)
                  uses schema to parse Redis Hash into model input vector

Offline Training (Spark):
  On job start → fetch registered schema for all feature groups in training config
  Point-in-time join uses registered schema_ver at label_timestamp:
    "use schema as it was when the label was generated, not today's schema"
  Prevents training/serving skew on schema change mid-experiment
```

---

## Operational Metrics to Track

```
Metric                              Alert threshold
──────────────────────────────────────────────────────────────────
delta propagation latency           > 20 minutes (regression from 15min SLO)
validation rejection rate           > 5% of PRs (schema hygiene issue)
cross-CRD validation failures       any (dependency graph broken)
deprecated feature usage            > 0 consumers after sunset date
ObjectConfig sync lag               > 30s (serving using stale routing)
Registry DB replication lag         > 5s (reads from stale replica)
FeatureGroup with no active model   > 90 days (orphan cleanup candidate)
```

---

## Key Tradeoffs

```
Config-as-code (Git) vs UI-driven registry:
  UI: easy for non-engineers, but no audit trail, no PR review, no rollback.
  Git: full history, PR review enforces cross-team validation, rollback = git revert.
  → Git wins for production feature stores. UI can be read-only view layer.

Server-side validation vs client-side:
  Client-side: fast feedback, but optional — teams can skip or use old validator.
  Server-side: enforced at write time, no bypass — one correct version.
  → Always server-side for correctness guarantees. Client-side only for
    fast pre-commit feedback (run the same validator locally before push).

Incremental delta vs full re-registration:
  Full re-register: simple, always consistent, but O(N) on every change.
  Delta: O(changed) but requires accurate last-registered-sha tracking.
  At N=10K feature groups, full re-register = 1 hour. Delta = 15 minutes.
  → Delta is non-negotiable at scale.

Deprecation vs deletion:
  Deletion: clean, but breaks any consumer that didn't migrate.
  Deprecation: field stays in schema, marked deprecated, sunset date set.
  → Never delete. Deprecate + track consumer migration + delete after all clear.
  Same principle as API versioning: additions are safe, removals are breaking.
```
