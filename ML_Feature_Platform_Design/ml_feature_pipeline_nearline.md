# ML Feature Pipeline — Nearline (Streaming) Path

Related: `ml_feature_pipeline_offline.md`, `funnel_analytics_pipeline.md`, `realtime_business_events.md`, `feature_registry_and_metadata.md`, `streaming_feature_flink_processors.md`

---

## Problem

Raw app events (clicks, searches, orders, ratings) are not directly usable by ML models.
Models need:
- Sessionized behavior sequences ("last 5 searches before order")
- Rolling aggregates ("order count last 30 days")
- Point-in-time correct features (training must not leak future data)
- Low-latency online features at serve time (<10ms)

---

## Canonical Event Schema

Every app event is a single JSON blob written to Kafka.

```json
{
  "event_id":    "uuid-v4",
  "user_id":     "u_123456",
  "session_id":  "s_abc789",
  "event_name":  "restaurant_clicked",
  "entity_ids":  { "restaurant_id": "r_99", "search_id": "srch_42" },
  "timestamp":   1720951500000,
  "client_meta": {
    "platform":  "ios",
    "app_ver":   "5.142.0",
    "lat":       37.774,
    "lng":       -122.419
  },
  "schema_ver":  3
}
```

**Key fields:**
```
event_id        Dedup key — idempotent writes
user_id         Join key across all pipelines
session_id      Set by the client SDK; gaps > 30min = new session
event_name      Controlled vocabulary: search, click, add_to_cart, checkout, order_placed, rated
entity_ids      Variable payload per event type (restaurant, order, driver)
timestamp       Client-side epoch ms; server_timestamp added at ingest
schema_ver      Incremented on any field change; consumers check this
```

---

## Ingestion Path

```
Mobile App
  │
  ├── Client SDK
  │     batches events every 10-30s, retries on failure
  │     attaches session_id (new UUID after 30min inactivity detected on-device)
  │     sends explicit session_end event on app background/close
  │
  └── Event Ingestion Service (HTTP/2 endpoint)
        │  validates schema_ver, enriches with server_timestamp, geo-lookup
        │
        └── Kafka: app.events (partitioned by user_id)
              │
              ├── [Shared] Flink Sessionization Job
              │     dedup (bloom filter on event_id, 24h TTL)
              │     state machine: group by session_id, emit on session_end or 30min timer
              │     → Kafka: app.sessions (partitioned by user_id)
              │           ├── ML Feature Pipeline   (this doc)
              │           └── Funnel Analytics      (→ funnel_analytics_pipeline.md)
              │
              ├── S3 Raw Event Log (Kafka → S3 sink, ground truth)
              │
              └── Real-time Business (→ realtime_business_events.md)
                    reads app.events directly — CEP must react to individual events
                    immediately, cannot wait for session to close
```

**Why app.sessions as the shared intermediate:**
Sessionization is stateful and expensive (30min gap window, bloom filter dedup).
Running it once and publishing to `app.sessions` lets both the ML pipeline and
Funnel Analytics consume pre-sessionized data without duplicating that work.

**Why Real-time Business reads app.events (not app.sessions):**
Fraud detection needs to react to individual events immediately — e.g., block on
the 3rd failed payment before the session closes. Waiting for a 30min session
window to close is too late.

---

## Dedup and Late Arrival

Both are handled in the **shared Flink Sessionization Job** before events reach
`app.sessions`. By the time the ML Feature Flink Job reads from `app.sessions`,
events are already deduped and assigned to correct sessions.

```
Dedup (in shared Sessionization Job):
  Bloom filter on event_id, 24h TTL.
  Duplicate event_id → dropped before session assignment.

Late arrival (in shared Sessionization Job):
  Flink event-time watermark, 2h allowed lateness.
  Events within 2h of watermark → buffered, assigned to correct session window.
  Events past 2h → Kafka: late.events.dlq
    → Spark reprocessing job re-computes affected features in offline store.

ML Feature Flink Job does not need its own dedup or late-arrival logic —
it trusts that app.sessions contains clean, ordered SessionRecords.

Server time vs client time:
  Use client timestamp for session ordering (user behavior is client-side).
  Use server_timestamp only for SLA monitoring and lag detection.
```

---

## Sessionization

### Who owns what

```
Client owns:   session boundary detection (30min inactivity gap tracked on-device)
               session_id generation (new UUID after each 30min gap)
               explicit session_end signal sent via app lifecycle events
                 iOS:     applicationDidEnterBackground
                 Android: onStop

Server owns:   event buffering per session_id
               30min allowed lateness window (retries + delayed session_end)
               emit trigger (session_end OR 30min timer, whichever fires first)
               DLQ routing for events arriving after window closes
```

### Why client owns session detection (not server time window)

Client knows exactly when the user stopped interacting. A server-side Flink session
time window re-derives this from event timing gaps — duplicating work the client
already did and adding 30min of emit latency in the common case.

If client attaches session_id → server just groups by session_id. No time window
needed as the primary mechanism.

### Server Flink state machine (replaces session time window)

```
State per user_id:
  current_session_id
  buffers:  HashMap<session_id, List<events>>
  timers:   HashMap<session_id, 30min_deadline>

On event (user_id=X, session_id=S, event_name, ts):
  → append to buffers[S]

  if S != current_session_id[X]:
    → set timer(old_session_id, now + 30min)   // allowed lateness window starts
    → current_session_id[X] = S

  if event_name == "session_end" for session S:
    → emit SessionRecord(buffers[S]) immediately  // fast path
    → cancel timer[S], clear buffers[S]

On timer fire for session_id S:
  → emit SessionRecord(buffers[S])               // fallback path
  → clear buffers[S]

On event for already-emitted session_id S:
  → DLQ (arrived after 30min window closed)
```

### Why 30min timer serves double duty

```
Same 30min value, two purposes:
  1. Fallback for missing session_end:
       App crash or force-kill → client never sends session_end
       Timer fires after 30min → session emitted anyway

  2. Allowed lateness for late S1 events (retry / network reorder):
       Client has S1 retry buffered + starts S2 simultaneously
       Server sees S2 first → starts 30min timer for S1
       S1 retry arrives (seconds later, well within 30min) → added to S1 buffer
       session_end(S1) or timer → emit complete S1 record

Events arriving after the 30min timer fires → DLQ (beyond allowed lateness)
In practice retries arrive in seconds, not minutes, so DLQ rate is near zero.
```

### Session features extracted per emit

The ML Feature Flink Job reads each SessionRecord from `app.sessions` and extracts:

```
session_duration_sec       end_ts - start_ts
event_count                len(events)
search_to_click_count      count(search) followed by count(click) in events list
click_to_order_flag        any(event_name == 'order_placed') in events
searched_cuisines          set of cuisine filters seen in entity_ids
last_event_name            events[-1].event_name
```

Rolling features (orders_last_7d, avg_order_value_30d) are accumulated across
sessions using Flink ValueState per user_id — each session emit updates the
rolling state.

---

## Feature Computation

Two layers: streaming (Flink) for freshness, batch (Spark) for correctness.

### Layer 1 — Streaming Features (Flink, minutes-fresh)

Full processor code and Flink patterns: `streaming_feature_flink_processors.md`

```
Input:  app.sessions Kafka topic (pre-sessionized, deduped)
Output: online feature store (Redis)

Computed per user_id, updated on every session close:

  Feature                   Processor                     Flink state
  ─────────────────────────────────────────────────────────────────────
  last_session_id           P1 SessionFeatureExtractor    none (stateless)
  last_search_ts            P1 SessionFeatureExtractor    none
  click_to_order_flag       P1 SessionFeatureExtractor    none
  searched_cuisines         P1 SessionFeatureExtractor    none
  last_event_name           P1 SessionFeatureExtractor    none

  orders_last_7d            P2 SlidingWindowOrders        ListState<(ts,value)>
  orders_last_30d           P2 SlidingWindowOrders        ListState<(ts,value)>
  avg_order_value_30d       P2 SlidingWindowOrders        ListState<(ts,value)>
  is_new_user               P2 SlidingWindowOrders        ValueState<Integer>

  cuisines_last_10_clicks   P3 CuisineRingBuffer          ValueState<ArrayDeque>

  engagement_score          P4 EngagementScoreDecay       ValueState<(score,ts)>
```

Key implementation notes:
  P1 is stateless — derives all features from current SessionRecord alone.
  P2 uses manual ListState eviction, not Flink SlidingWindow (which would create
     720 copies of data for a 30d window with 1h slide).
  P3 stores a bounded deque — oldest cuisine dropped when size > 10.
  P4 decays score lazily on next session arrival (no timer per user).
  All processors write to Redis via pipelined HSET (one round trip per session).

### Layer 2 — Batch Features (Spark, daily)

```
Input:  S3 event log (partitioned by date)
Output: offline feature store (Hive/Parquet on S3) + snapshot to Redis

Computed daily, point-in-time correct:

  user_features_daily partition=2026-07-15
    user_id
    orders_last_30d_correct   exact count from full history (not approx)
    ltv_90d                   lifetime value last 90 days
    churn_risk_score          % sessions with no order, last 4 weeks
    preferred_cuisine         mode of cuisines ordered, last 60 days
    wilson_rating_given       avg rating given by user (Wilson-adjusted)

Why batch for these:
  Sliding window in Flink approximates counts using watermarks.
  Spark reads full history → exact. For model training, exact features matter.
  Daily cadence acceptable for slow-moving features (LTV, churn risk).
```

---

## Feature Store: Online vs Offline

```
                Online Store (Redis)        Offline Store (S3 + Hive)
────────────────────────────────────────────────────────────────────────
Purpose         Model serving (<10ms)       Model training, backfills
Written by      Flink (streaming)           Spark (batch daily)
Read by         Ranking model at request    Training jobs, Spark pipelines
Freshness       Minutes                     Daily (T-1)
Data retained   7 days (TTL)                Unlimited (partitioned by date)
Key format      user:{id}:features          partition=2026-07-15/user_features
Schema          Redis Hash                  Parquet columns
```

**Point-in-time correctness for training:**
```
Problem:  Model trained on features as of today → but today's features include
          data from after the label event → future leakage → inflated offline AUC.

Solution: Point-in-time join in Spark.
          For each training example (user_id, label_timestamp):
            → join to offline feature store partition where date <= label_timestamp
            → pick the feature snapshot closest to but before label_timestamp

Spark query:
  SELECT t.user_id, t.label, f.orders_last_30d, f.ltv_90d
  FROM training_labels t
  LEFT JOIN user_features f
    ON t.user_id = f.user_id
   AND f.date = (
     SELECT MAX(date) FROM user_features
     WHERE user_id = t.user_id AND date <= t.label_timestamp
   )
```

---

## Schema Evolution

```
Problem:  Client SDK v5.142 adds a new field "dark_mode: bool".
          Consumers reading schema_ver=2 don't know this field.

Strategy: Backwards-compatible evolution only (never remove/rename fields).

  schema_ver 1 → 2: ADD optional field entity_ids.coupon_id
  schema_ver 2 → 3: ADD optional field client_meta.dark_mode

Consumer behavior:
  schema_ver >= expected  → process normally (ignore unknown fields)
  schema_ver <  expected  → route to dead-letter-queue (DLQ) for reprocessing
                            after schema backfill

Schema registry:
  All schemas stored in Confluent Schema Registry.
  Producer validates before publishing.
  Consumer validates on read.
  Breaking changes (remove field, rename) blocked at registry level.
```

---

## Data Quality Monitoring

```
Check                           Threshold       Action
────────────────────────────────────────────────────────────────────────
events_per_minute drop > 50%    pagerduty       SDK instrumentation bug
event_id null rate > 0.1%       alert           client SDK version check
session_id null rate > 1%       alert           session tracking regression
schema_ver mismatch > 5%        alert           force-upgrade old SDK
late events > 5% of volume      alert           clock skew investigation
user_id null rate > 0%          block + DLQ     never allow anonymous events
feature freshness > 30min       alert           Flink lag or Redis write failure
```

---

## Write Path Summary

```
Client SDK
  │  attaches session_id per event
  │  sends session_end on app background/close
  │
  └── Kafka: app.events
        │
        ├── [Shared] Flink Sessionization Job
        │     ├── dedup: bloom filter on event_id (24h TTL)
        │     ├── state machine: buffer by session_id
        │     │     emit trigger: session_end event  → emit immediately (fast path)
        │     │                   30min timer        → emit on timeout  (fallback)
        │     │     late events after timer          → DLQ
        │     └── Kafka: app.sessions
        │               │
        │               └── ML Feature Flink Job          (this pipeline)
        │                     ├── extract session features from SessionRecord
        │                     ├── update rolling ValueState (7d, 30d windows)
        │                     └── HSET → Redis (online feature store)
        │
        └── S3 Raw Event Log (Kafka → S3 sink, partitioned by date)
              │
              └── Spark Daily Job                         (this pipeline)
                    ├── reads S3 T-1 partition (full history, exact)
                    ├── point-in-time correct feature computation
                    ├── writes Hive/Parquet (offline feature store)
                    └── snapshot hot features → Redis (corrects Flink approximations)
```

---

## Serving at Request Time

```
User opens Uber Eats → Feed Ranking Model scores restaurants:

  1. Read user:{id}:features from Redis         (< 5ms, online store)
  2. Read restaurant features from Redis         (< 5ms)
  3. Pass to ranking model (XGBoost / DLRM)
  4. Return ranked list

SLO: p99 feature fetch < 10ms
     Redis cluster: 3 shards, 3 replicas each, read from replica
```

---

## Key Tradeoffs

```
Streaming (Flink) vs Batch (Spark):
  Flink: fresh (minutes), approximate rolling counts due to watermarks
  Spark: stale (daily), exact from full history
  → Use both: Flink for serving, Spark for training. Snapshot Spark results
    to Redis nightly to correct Flink approximations.

Session window vs tumbling window:
  Tumbling breaks sessions that span window boundaries.
  Session window respects natural user activity gaps.
  → Always use session window for user behavior sessionization.

Point-in-time join:
  Skip it → training/serving skew, inflated offline AUC, model underperforms.
  → Non-negotiable for any time-sensitive label.
```
