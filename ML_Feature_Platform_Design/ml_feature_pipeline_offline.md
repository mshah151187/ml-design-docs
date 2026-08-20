# ML Feature Pipeline — Offline (Batch) Path

Related: `ml_feature_pipeline_nearline.md`, `feature_registry_and_metadata.md`

---

## Why Offline Exists at All

The nearline doc (Flink reading `app.sessions`) gives us minutes-fresh features for serving.
But two things it cannot do:

**1. Training requires exact history, not approximations.**
Flink's rolling counts use watermarks — they're correct for serving but approximate over
multi-month windows. For training, a model trained on approximate counts learns noise.
You need exact feature values computed over the full historical record.

**2. Flink only sees app events. ML features need more.**
`orders_last_30d` isn't just a count of events in Kafka. The authoritative order record lives
in the Orders OLTP database. Driver ratings live in another DB. Billing data in another.
None of this flows through `app.events`. We need a separate path that can ingest from
multiple source types and join them together.

That's the offline path: correctness over freshness, multi-source, full history.

---

## Step 1: What Sources Do We Need?

Start by asking: what data do ML features actually depend on?

```
Feature                       Source system
──────────────────────────────────────────────────────
orders_last_30d               Orders OLTP DB
avg_order_value_30d           Orders OLTP DB
driver_rating                 Ratings OLTP DB
ltv_90d                       Billing DB
restaurant_cancellation_rate  Orders OLTP DB
preferred_cuisine             app events (Kafka)
session_click_to_order_rate   app events (Kafka)
city_surge_multiplier         Surge Pricing DB
```

So we're pulling from: Kafka (streaming events) + multiple OLTP databases + possibly
batch warehouse exports. Three fundamentally different source types:

```
Streaming   App events flow continuously through Kafka at high volume.
            Already landing to S3 raw log (from nearline setup).

CDC         OLTP databases change continuously — orders are created, updated, cancelled.
            We need every change, not just a daily snapshot.

Batch       Some sources only support periodic exports (3rd party feeds, billing snapshots).
            Pull on a schedule.
```

Each of these has a different shape, cadence, and delivery mechanism. How do we get
them all into a form where we can join them?

---

## Step 2: Landing — How Do We Make Heterogeneous Sources Queryable?

The naive approach: read each source differently in every downstream job.

```
Spark job reads orders from MySQL directly.
Spark job reads events from S3 raw log in its own format.
Spark job reads billing from warehouse export.
```

Problem: every downstream consumer has to know the shape, auth, and quirks of every
upstream source. A schema change in the Orders DB breaks every Spark job that reads it.
Adding a new source means updating every consumer.

**The fix: a single landing layer with a uniform envelope.**

All sources land into one storage layer in the same format:

```json
{
  "source":       "orders_db",
  "event_time":   1720951500000,
  "ingest_time":  1720951502000,
  "op":           "u",
  "payload":      { "order_id": "ord_88", "status": "cancelled", "amount": 24.50 }
}
```

Fields:
```
source       which producer system (for lineage, DQ monitoring)
event_time   when it actually happened on the source side
             (use this for joins and window computation, not ingest_time)
ingest_time  when we received it (for SLA monitoring only)
op           c/u/d/r — create/update/delete/read-snapshot
             app events are always op=c; OLTP changes give us all four
payload      field-projected to only what downstream needs (5 of 50 fields)
```

This layer is called **Bronze**: raw, append-only, ground truth.

Why append-only? Because Bronze is the source of truth for reprocessing. If a downstream
job has a bug, we rerun it from Bronze — we must never corrupt what's in Bronze.
Every write is INSERT, never UPDATE or DELETE.

---

## Step 3: How Do We Ingest Each Source Type?

Each source has a completely different native format. The first job of each ingestion
path is to translate that native format into the common Bronze envelope before anything
lands in storage. Downstream Silver never sees the native format — it only sees Bronze rows.

### The Native-to-Bronze Translation

**Source 1: Kafka app event (native)**

```json
{
  "event_id":    "evt_uuid_001",
  "user_id":     "u_123456",
  "session_id":  "s_abc789",
  "event_name":  "order_placed",
  "entity_ids":  { "restaurant_id": "r_99", "order_id": "ord_88", "order_value": "24.50" },
  "timestamp":   1720951500000,
  "client_meta": { "platform": "ios", "app_ver": "5.142.0", "lat": 37.774, "lng": -122.419 },
  "schema_ver":  3,
  "debug_info":  { "trace_id": "...", "sampled": true, "internal_flags": "..." }
}
```

Problems: 50+ fields, nested blobs, debug_info is large and useless downstream.

**→ Bronze row (after projection):**

```json
{
  "source":      "kafka.app_events",
  "event_time":  1720951500000,
  "ingest_time": 1720951502341,
  "op":          "c",
  "payload": {
    "event_id":      "evt_uuid_001",
    "user_id":       "u_123456",
    "session_id":    "s_abc789",
    "event_name":    "order_placed",
    "restaurant_id": "r_99",
    "order_id":      "ord_88",
    "order_value":   24.50,
    "platform":      "ios"
  }
}
```

What changed: `debug_info` dropped entirely, nested `entity_ids` flattened to top-level
fields in payload, `timestamp` mapped to `event_time`, `op` hardcoded to `c` (Kafka events
are always creates — no updates or deletes flow through app events).

---

**Source 2: Debezium CDC event from Orders Postgres (native)**

```json
{
  "before": {
    "order_id":           "ord_88",
    "customer_id":        "cust_9988",
    "status":             "confirmed",
    "amount":             24.50,
    "payment_ref":        "stripe_ch_abc",
    "internal_notes":     "routed via gateway-v2",
    "updated_at":         "2026-07-15T10:05:00Z"
  },
  "after": {
    "order_id":           "ord_88",
    "customer_id":        "cust_9988",
    "status":             "cancelled",
    "amount":             24.50,
    "payment_ref":        "stripe_ch_abc",
    "internal_notes":     "routed via gateway-v2",
    "updated_at":         "2026-07-15T10:07:30Z"
  },
  "op":    "u",
  "source": {
    "connector": "postgresql",
    "db":        "orders_prod",
    "table":     "orders",
    "lsn":       8849201,
    "ts_ms":     1720951650000
  }
}
```

Problems: carries `before` + `after` blocks (large), internal fields (`payment_ref`,
`internal_notes`) not needed, `ts_ms` buried inside `source`.

**→ Bronze row (after projection):**

```json
{
  "source":      "orders_db.orders",
  "event_time":  1720951650000,
  "ingest_time": 1720951651820,
  "op":          "u",
  "payload": {
    "order_id":      "ord_88",
    "customer_id":   "cust_9988",
    "status_before": "confirmed",
    "status_after":  "cancelled",
    "amount":        24.50,
    "lsn":           8849201
  }
}
```

What changed: `before`/`after` collapsed into `*_before`/`*_after` fields for the changed
columns only; `ts_ms` from the `source` block promoted to `event_time`; `payment_ref` and
`internal_notes` dropped; `lsn` kept (useful for ordering events within the same
transaction). `op=u` preserved as-is.

Note: `event_time` here is the Postgres commit timestamp — not when Debezium emitted it,
not when we received it. This is the time the DB row actually changed, which is what PIT
joins need.

---

**Source 3: S3 restaurant dimension export (native Parquet file)**

```
Daily dump: s3://data-exports/restaurant_dim/2026-07-14/part-00000.parquet

restaurant_id | city_id | name         | cuisine_type | avg_rating | partner_id | internal_rank | pricing_tier | ... 40 more columns
r_99          | c_sf    | "Taco Place" | mexican      | 4.3        | p_westcoast| 7             | premium      | ...
r_100         | c_sf    | "Ramen Bar"  | japanese     | 4.7        | p_westcoast| 2             | standard     | ...
```

Problems: 50 columns, most unused downstream; no natural `event_time` field (it's a
snapshot, not a change event); no `op` field.

**→ Bronze rows (one per restaurant, after projection):**

```json
{
  "source":      "s3.restaurant_dim_export",
  "event_time":  1720828800000,
  "ingest_time": 1720952100000,
  "op":          "r",
  "payload": {
    "restaurant_id": "r_99",
    "city_id":       "c_sf",
    "cuisine_type":  "mexican",
    "avg_rating":    4.3,
    "partner_id":    "p_westcoast"
  }
}
```

What changed: `event_time` = export_date midnight (2026-07-14 00:00:00 UTC) — the date
the snapshot was valid, not when the Spark pull job ran (ingest_time). This matters:
if a training label is from 2026-07-13, PIT join needs the restaurant's attributes
*as of* that date, not today's. `op=r` (read-snapshot) signals this is not a real change
event — it's a full state read. `internal_rank`, `pricing_tier`, and 38 other columns dropped.

---

### Summary: What the Ingestion Consumer Does Per Source

```
For every source, the ingestion consumer does exactly three things:

1. Identify event_time
     Kafka:   event.timestamp  (client-side, when user acted)
     CDC:     source.ts_ms     (Postgres/MySQL commit timestamp)
     Batch:   export_date      (midnight of the snapshot date)

2. Map op
     Kafka:   always "c"       (creates only — no updates in app events)
     CDC:     from Debezium op field (c / u / d)
     Batch:   always "r"       (read-snapshot)

3. Project payload
     Keep only fields downstream Silver needs (typically 5-10 of 30-50 source fields)
     Flatten nested structures
     Drop debug/internal fields
     Collapse before/after into *_before/*_after for CDC

Everything else is uniform — source, event_time, ingest_time, op, payload.
Silver never needs to know if a row came from Kafka, Debezium, or a batch export.
```
---

## Step 4: We Have Raw Data — Why Can't We Just Query It?

A clarification first: "same format" from Step 2 means same envelope schema
(`source, event_time, ingest_time, op, payload`), not one physical table.
Bronze has one table per source — each with typed payload fields specific to that source —
but all follow the same envelope so Silver can process them uniformly.

```
bronze.app_events        { source, event_time, ingest_time, op, payload: {event_id, user_id, event_name, ...} }
bronze.orders_cdc        { source, event_time, ingest_time, op, payload: {order_id, customer_id, status_before, ...} }
bronze.restaurant_export { source, event_time, ingest_time, op, payload: {restaurant_id, city_id, cuisine_type, ...} }
```

Why per-source tables, not one giant table?
- Each source has different typed payload fields — a single table forces untyped JSON blobs
- Per-source retention: app events 90 days, billing records 7 years
- Per-source DQ: "CDC consumer lag" vs "Kafka consumer lag" need separate monitoring

Now try to answer: "how many orders did user X place in the last 30 days?"

```sql
SELECT COUNT(*) FROM bronze.orders_cdc
WHERE customer_id = 'cust_9988'
  AND event_time >= now() - interval 30 days
```

**Problem 1: The same user has different IDs in different systems.**

```
bronze.app_events:    user_id      = "u_123"
bronze.orders_cdc:    customer_id  = "cust_9988"
bronze.billing:       billing_id   = "bill_XY77"
```

These are the same person. There's no shared key. The join is impossible.

**Problem 2: Multiple op types in CDC create duplicate counting.**

Bronze has every insert + update + delete. If an order was created then cancelled,
Bronze has two rows for it. A naive COUNT gives 2, not 1.

**Problem 3: Late data creates dirty windows.**

A CDC event with `commit_ts` from yesterday might land in Bronze today (network delay,
Debezium lag). A query over yesterday's Bronze partition misses it.

These three problems mean Bronze is not directly queryable for business logic.
We need a **conform layer** that solves them.

---

## Step 5: Silver — Conform, Deduplicate, Resolve Identities

Silver reads from Bronze and produces a clean, conformed view. Two jobs run:

```
Flink (continuous): Bronze → Silver for streaming freshness (~15 min lag)
Spark (hourly):     Bronze → Silver for full correctness (catches late CDC events)
Spark overwrites Flink's Silver partition each hour — correctness wins.
```

### Solving Problem 1: Identity Resolution

Build an **identity graph** that maps every source ID to a stable `unified_user_id`:

```
Deterministic matching (exact key):
  billing_id "bill_XY77" → same verified email as user_id "u_123" → unified_id "uid_555"
  account_id matches → link

Probabilistic matching (fuzzy):
  fuzzy name + device fingerprint + address → scored, thresholded
  gray zone → human review queue

Output: mapping table { source, source_id → unified_user_id }
        versioned + time-aware (entities can merge and split over time)
```

Every Silver fact is enriched with `unified_user_id` before writing. All downstream
joins use `unified_user_id` — no source-specific key ever leaks into Gold.

Why this is non-negotiable: business questions like "churn rate for users with ≥1 order"
require joining `app_events` (user_id) with `orders_cdc` (customer_id). Without
`unified_user_id` this join is impossible and the metric cannot be computed.

### Solving Problem 2: Canonical Model (Conformed Dimensions)

Define shared dimensions every source maps onto:

```
dim_user(unified_user_id, account_id, region, tier, created_ts)
dim_restaurant(restaurant_id, city_id, cuisine_type, partner_id)
dim_driver(unified_driver_id, rating, vehicle_type)

Per-source facts keyed by conformed dimensions:
  fact_app_event(unified_user_id, restaurant_id, event_time, action, session_id)
  fact_order(unified_user_id, restaurant_id, order_ts, amount, status)
  fact_rating(unified_user_id, unified_driver_id, rating_ts, score)
```

Once every fact carries `unified_user_id + restaurant_id + event_time`, every business
question is a join over the star schema using the same keys.

### Solving Problem 3: Watermarks for Late Data

Flink holds a window open until watermark passes `window_end + allowed_lateness`.
Spark covers anything past that lateness with its full Bronze re-read.

```
Silver record: one row per event, deduped on event_id, enriched with unified_user_id,
               conforming to canonical dimension keys.
```

---

## Step 6: Gold — Per-Consumer Serving Marts

Silver has clean, joined, conformed data. Why not let ML teams query Silver directly?

**Problem: different consumers need different slices, different security, different joins.**

```
Internal ML team:        all users, all features, full history
Partner dashboard:       only their own restaurants, no other partner's data
Finance:                 all billing, but PII columns masked
ML training job:         point-in-time correct slice (see Step 7)
```

If every consumer queries Silver directly, they re-implement the join logic, the row
filtering, the PII masking — and each team does it differently, so metrics drift across teams.

**The fix: Gold marts.**

Gold runs a Spark job (Airflow-scheduled) that does the PIT join and materializes
per-persona tables with security enforced at write time:

```
gold.ml_user_features         (for ML training + serving snapshot)
  unified_user_id
  feature_date                T-1 daily partition
  orders_last_7d              exact from Silver
  orders_last_30d             exact from Silver
  avg_order_value_30d
  ltv_90d
  preferred_cuisine
  churn_risk_score

gold.partner_restaurant_view  (for partner dashboards)
  row filter: WHERE partner_id = :requesting_partner_id
  column mask: no PII fields
```

Consumers read Gold and never re-join Silver. The platform owns correctness.

---

## Step 7: Point-in-Time Correctness — the Training Trap

We have Gold with daily feature snapshots. Can we just join training labels to today's Gold?

**No. This is the single most common silent ML bug.**

```
Label: user u_123 churned on 2026-01-15.
Feature: orders_last_30d

WRONG: read today's (2026-07-15) orders_last_30d for u_123 → includes 6 months of future orders
       Model sees "this user placed 40 orders" and labels them as churned → nonsense
       Model trains on future data → inflated offline AUC → underperforms in production

RIGHT: read orders_last_30d AS OF 2026-01-15 → only orders before the churn event
       Model sees "this user placed 2 orders recently" and labels them as churned → correct
```

This is called **future leakage**. The model looks brilliant offline and collapses in production.

The fix: Silver stores history as `(entity_id, feature_ts, value)` tuples — INSERT only,
never UPDATE. Every feature value is preserved with its timestamp.

Spark PIT join in Gold:

```sql
SELECT
  t.unified_user_id,
  t.label,
  t.label_ts,
  f.orders_last_30d,
  f.ltv_90d,
  f.preferred_cuisine
FROM training_labels t
LEFT JOIN silver_user_features f
  ON t.unified_user_id = f.unified_user_id
 AND f.feature_ts = (
   SELECT MAX(feature_ts)
   FROM silver_user_features
   WHERE unified_user_id = t.unified_user_id
     AND feature_ts <= t.label_ts         -- AS-OF: only data before the label
 )
```

Why can't Flink do this? Flink keeps recent state in RocksDB (days to weeks).
A training label from 6 months ago needs feature values from 6 months ago — that's only
in the full historical Silver on disk. Spark reads it all; Flink can't.

---

## Step 8: Why Raw Parquet Is Not Enough — Enter Apache Iceberg

So far we've described Bronze/Silver/Gold as storage layers. What format do we use?

The naive answer: Parquet files on S3. Let's see where that breaks.

**Backfill problem:**
A bug in the Silver identity resolution job misidentifies 5% of users. We fix the bug and
need to reprocess yesterday's Silver partition.

```
With raw Parquet:
  Step 1: delete yesterday's Parquet files
  Step 2: run Spark, write new files

  If Spark fails mid-write: partition is half old, half new → corrupted
  If a training job reads during Step 1-2: reads empty or partial data
  No way to roll back to good state
```

**Apache Iceberg solves this** with ACID table semantics:

```
With Iceberg:
  Spark writes new files to a staging location
  On success: atomically commits a new snapshot (old snapshot still fully readable)
  On failure: new files are abandoned, old snapshot untouched
  A concurrent training job reading during backfill sees the old snapshot — consistent read

Backfill = new Iceberg snapshot. Old snapshot survives until GC.
```

Other Iceberg capabilities that matter here:

```
Schema evolution:    add/rename columns without rewriting data
                     sources add fields constantly; we shouldn't need full table rewrites

Partition evolution: change hourly→daily partitioning without history rewrite
                     as data grows we change partition granularity; raw Parquet can't do this

Time travel:         query Silver "as of snapshot_id 8849a" (6 months ago)
                     this is what makes PIT joins feasible at scale
                     Iceberg stores snapshot history → Spark reads historical state

Hidden partitioning: query by event_date without knowing physical layout
                     partition pruning is automatic → cheap daily training pulls
```

---

## Step 9: Idempotent Backfill — Correctness Under Failure

What triggers a need to reprocess?

```
1. Late CDC event arrives past Flink watermark → reprocess that window
2. Bug fix in Silver logic → reprocess all affected windows
3. Identity graph update (two users merged) → downstream Silver + Gold reprocess
4. New feature definition added to Gold → backfill full history
```

**The rule: every pipeline window must be idempotently reprocessable.**

Idempotent means: run the backfill once or ten times → same result. No double-counting.

```
POST /backfill { table: "gold.ml_user_features", window: [2026-06-01, 2026-06-30] }
  → Spark reads Bronze/Silver for that date range
  → Applies current logic
  → Iceberg ACID overwrite of those partitions (replaces, not appends)
  → DQ checks run on overwritten partitions
  → Orchestrator marks windows as re-verified

Common mistake: append-only backfill
  If you INSERT new rows without deleting old ones → double-counts in Gold
  Training job sees each order twice → all counts are wrong → model is wrong
  Iceberg overwrite = safe atomic replace; this mistake is impossible with Iceberg
```

Backfill parallelism: 2 weeks of hourly windows = 336 windows → Spark runs them in
parallel bounded by cluster size. Not sequential — backfilling a month takes hours, not weeks.

---

## Step 10: Training-Serving Skew — The Final Correctness Problem

We now have:
- Flink computing features for serving (nearline, from `app.sessions`)
- Spark computing features for training (offline, from Gold Iceberg)

**Problem: if they use different logic, the model trains on data it will never see in production.**

```
Training (Spark Gold):  orders_last_30d = COUNT of order_placed events where amount > 0
Serving  (Flink Redis): orders_last_30d = COUNT of all order_placed events

A $0 test order counted in serving but not in training.
Model saw 5 orders in training → sees 6 orders in serving → feature mismatch → degraded ranking.
```

This is **training-serving skew** — silent in CI, shows up as offline-online metric gap.

**Fix: single feature definition compiled to both paths.**

The feature registry (→ `feature_registry_and_metadata.md`) stores one definition:

```yaml
feature: orders_last_30d
entity:  user_id
window:  30d
logic:   COUNT of order_placed events WHERE amount > 0
```

The platform materializes it twice using the same logic:
- Offline (Spark → Gold Iceberg → training datasets)
- Online (Flink → Redis → serving)

Individual teams don't implement this logic themselves — the platform owns it.
Teams can't drift because there's only one definition.

---

## Step 11: Schema Safety at Producer Boundary

One more failure mode: a source producer renames a field.

```
Orders DB team renames: order_value → order_amount
  → CDC event now has order_amount, not order_value
  → Silver job reads order_value → null
  → Gold features: avg_order_value_30d = null for all users
  → Training data silently corrupted for 24h before anyone notices
```

Fix: enforce contracts at the producer boundary, not in the consumer.

```
Producer registers a schema contract:
  PUT /contracts/orders_db {
    schema:         Avro schema (order_id, order_amount, status, commit_ts)
    event_time:     commit_ts
    pii_tags:       [customer_name, address]
    compat_policy:  backward
  }

CI gate runs on every schema change:
  Breaking change (rename order_value → order_amount): FAIL THE BUILD
  Additive change (add new field coupon_id): PASS

Rules:
  Add optional field      → backward compatible, allowed
  Rename / drop field     → breaking, blocked; must deprecate + dual-write first
  Tighten validation      → only on new schema version
```

Producer owns schema stability. Downstream Silver never has to defensively parse
for fields that may or may not exist.

---

## Full Architecture (Derived)

```
SOURCES (with data contracts enforced at CI boundary)
  Kafka: app.events      CDC: Orders/Ratings/Billing DBs     Batch: warehouse exports
         │                       │  (Debezium WAL tailing)          │  (Spark pull)
         └───────────────────────┴──────────────────────────────────┘
                                  │
                            INGESTION
                  per-source consumer + field projection
                  uniform landing envelope: {source, event_time, op, payload}
                                  │
                                  ▼
                   BRONZE (Iceberg, append-only, partitioned by event_date)
                   ground truth — never modified
                          │              │
                   Flink (stream)   Spark (batch)
                   conform + dedup  full history, catches late CDC
                          │              │
                          └──────┬───────┘
                                 ▼
                  SILVER (Iceberg, INSERT-only history, conformed)
                  - identity resolution → unified_user_id on every fact
                  - canonical dimensions (dim_user, dim_restaurant, ...)
                  - dedup + watermark-bounded late-data handling
                  - all values time-stamped: (entity_id, feature_ts, value)
                                 │
                          Spark PIT join
                          (point-in-time correct)
                                 ▼
                    GOLD (Iceberg, per-persona marts)
                    - row/column security enforced here
                    - daily feature snapshots for training
                    - nightly snapshot → Redis (corrects nearline approximations)
                          │               │               │
                  ML training jobs   Partner dashboards  Internal analytics
```

Orchestration: Airflow/Temporal DAG runs ingest → Silver → Gold with explicit dependencies.
Per-pipeline SLO: hourly Gold completes within the hour.
End-to-end SLO: joined Gold view fresh within 1h for operational marts.

---

## Data Quality Signals

How do you know the pipeline is producing correct data?

```
Signal                  What a drop/spike means
────────────────────────────────────────────────────────────────────────
join-hit-rate           fraction of Silver facts that found a dimension
                        (fact_order unified_user_id found in dim_user)
                        drop → identity resolution broke or new source has unmapped IDs

completeness            row-count vs expected from source
                        drop → CDC lag, ingest consumer fell behind

schema-drift count      fields in payload that changed type or disappeared
                        any → contract violation, investigate producer

late-data rate          events arriving past Flink watermark
                        spike → Kafka consumer lag or source clock skew

freshness lag           time since last Gold partition committed
                        breach → Spark job failed or was preempted

feature null rate       null values in Gold features per column
                        > 0.1% → upstream Silver null propagation, bad identity join
```

`join-hit-rate` is the most important. A 10% drop means 10% of training examples have
null features. The model silently trains on corrupted data — no error is thrown.

---

## Key Tradeoffs Derived

```
Why three layers, not one?
  Bronze exists because raw data must be preserved for reprocessing — append-only ground truth.
  Silver exists because raw data can't be joined (different IDs, different shapes, CDC noise).
  Gold exists because different consumers need different projections + security.
  Merging them creates a layer that can't serve all three purposes cleanly.

Why Iceberg and not raw Parquet?
  Backfill safety: ACID overwrite vs corrupt-prone delete+rewrite.
  Schema evolution: add columns without full table rewrite.
  Time travel: PIT joins against historical snapshots.
  These three requirements together rule out raw Parquet.

Why Spark for Silver/Gold, Flink for Bronze→Silver streaming?
  Flink: low latency, native watermarks, continuous — good for ~15 min Silver freshness.
  Spark: full history, massive parallelism, shuffle joins across TBs — required for PIT joins
         and multi-week backfills. Flink state (RocksDB) doesn't hold 6 months of history.

Why identity resolution in Silver, not Gold?
  Gold is per-persona and potentially row-filtered.
  Identity resolution needs to see all source IDs across all personas to build the graph.
  If done in Gold, a partner mart might miss cross-source links that span tenant boundaries.
```
