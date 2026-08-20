# Funnel Analytics Pipeline

Related: `ml_feature_pipeline.md`, `realtime_business_events.md`

---

## Problem

Analysts and product managers need to answer questions like:
- "What % of users who searched also clicked, then ordered — and where did they drop off?"
- "How does conversion rate compare for new vs returning users in the last 7 days?"
- "Which restaurants show the highest add-to-cart to no-order drop-off?"

These are not simple click counters — they require:
- **Funnel linkage**: A then B then C, in order, by the same user, within a time window
- **Multi-dimensional slicing**: by platform, city, cuisine, cohort
- **Both fresh and historical data**: dashboard shows last 1h; analyst queries 90-day history

---

## Architecture Diagram

```
┌─────────────────────┐
│   Mobile / Web App  │
│     (Client SDK)    │  batches events, attaches session_id, retries
└──────────┬──────────┘
           │ HTTP/2
           ▼
┌──────────────────────────┐
│  Event Ingestion Service │  validate schema_ver, enrich server_ts + city,
└──────────┬───────────────┘  rate limit per user_id
           │
           ▼
┌──────────────────────────────────────────────┐
│            Kafka: app.events                 │  partitioned by user_id
└──────┬───────────────────────────┬───────────┘
       │                           │
       ▼                           ▼
┌──────────────────┐      ┌─────────────────────┐
│  Flink           │      │  S3 Raw Event Log   │  Kafka→S3 sink
│  Sessionization  │      │  (ground truth,     │  partitioned by date
│  Job             │      │   immutable)        │  never modified
│  (state machine: │      └──────────┬──────────┘
│   group by       │                 │ daily T-1
│   session_id,    │                 ▼
│   bloom dedup,   │
│   emit on        │
│   session_end or │
│   30min timer)   │
└──────┬───────────┘      ┌─────────────────────┐
       │ SessionRecord     │  Spark Batch Job    │  runs 02:00 UTC
       ▼                  │  (exact history,    │  reads full T-1 partition
┌──────────────────┐      │   handles late      │  including late arrivals
│  Kafka:          │      │   arrivals)         │
│  app.sessions    │      └──────────┬──────────┘
└──────┬───────────┘                 │
       │                             │
       ▼                             │
┌──────────────────────┐             │
│  Flink Session       │             │
│  Flattener Job       │             │
│  (explode session →  │             │
│   1 row per event,   │             │
│   batch insert 1min) │             │
└──────┬───────────────┘             │
       │                             │
       ▼                             ▼
┌──────────────────────────────────────────────────────────┐
│                    ClickHouse Cluster                    │
│                                                          │
│  funnel_events_streaming   MergeTree, Flink writes, 7d  │
│  funnel_events_historical  MergeTree, Spark writes       │
│         │                                                │
│         │  ClickHouse Materialized Views (auto rollup)   │
│         ▼                                                │
│  funnel_agg_1m   AggregatingMergeTree, 30d              │
│  funnel_agg_1h   AggregatingMergeTree, 6 months         │
│  funnel_agg_1d   AggregatingMergeTree, 2 years          │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    Query Service     │  route by time range,
                │                     │  result cache (Redis),
                │                     │  access control,
                │                     │  query cost limits
                └──────┬───────────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
      Grafana     Superset /      Scheduled
      (live        Tableau         Reports
      dashboard)  (ad-hoc)        (Spark→PDF)

Late Event Path:
  Flink (past 2h watermark) ──► Kafka: late.events.dlq
                                  │
                                  ▼
                            Reprocessing Job (Spark, hourly)
                                  │
                                  ▼
                            ClickHouse (backfill correct partition)

Data Quality Monitor (daily Spark job):
  S3 partition (ground truth) ◄──► ClickHouse aggregates
  discrepancy > 0.5% → alert + trigger Reprocessing Job
```

---

## Architecture Components

**Client SDK**
Runs inside mobile/web app. Batches events locally before sending (prevents request storms on
flaky networks). Owns session lifecycle entirely:
- Generates new session_id (UUID) after 30min inactivity detected on-device
- Attaches session_id and session_start_ts to every event in a session
- Sends explicit session_end event on app background/close (iOS: applicationDidEnterBackground,
  Android: onStop) — this is the primary trigger for the server to emit the SessionRecord
Retries with exponential backoff on network failure. Attaches app_ver and platform.

**Event Ingestion Service**
Stateless HTTP/2 service, horizontally scalable. Responsibilities: validate schema_ver (reject
unknown versions to DLQ), enrich with server_timestamp (authoritative clock for SLA monitoring),
city from IP geo-lookup, rate limit per user_id. Writes valid events to Kafka. Invalid events go
to a separate ingest.dlq topic with a rejection reason tag.

**Kafka: app.events**
Central event bus. Partitioned by user_id so all events for one user land on the same partition.
This is critical: the Flink Sessionization Job is stateful keyed by user_id — if events for the
same user were spread across partitions, the session state would require cross-partition joins
(expensive). Partitioning by user_id eliminates this.

**S3 Raw Event Log**
Written by a Kafka → S3 sink (Kafka Connect S3 Sink Connector). Stores raw events partitioned
by `year/month/day`. Immutable — never modified after write. This is the ground truth for all
correctness checks, backfills, reprocessing, ML training data, and audit. ClickHouse is a
derived view; S3 is the authoritative record.

**Flink Sessionization Job**
Stateful Flink job keyed by user_id. Does NOT use a Flink session time window — client already
defines session boundaries. Runs as a state machine:

```
State per user_id:
  current_session_id
  buffers:  HashMap<session_id, List<events>>
  timers:   HashMap<session_id, 30min_deadline>

On event (session_id=S):
  → bloom filter dedup on event_id (24h TTL) — drop duplicates
  → append to buffers[S]
  → if S != current_session_id: set 30min timer on old session_id, update current
  → if event_name == "session_end": emit SessionRecord immediately, cancel timer

On 30min timer fire for session_id S:
  → emit SessionRecord(buffers[S])    ← fallback for crash/force-kill/delayed session_end
  → clear buffers[S]

Late events arriving after timer fires → Kafka: late.events.dlq
```

Emits SessionRecord { session_id, user_id, start_ts (from client), end_ts (from session_end
event or last event in buffer), events: List } to Kafka: app.sessions.

**Kafka: app.sessions**
Intermediate topic carrying SessionRecords. Partitioned by user_id. Decouples sessionization
from downstream consumers — if the Session Flattener Job falls behind, Sessionization Job is
unaffected.

**Flink Session Flattener Job**
Not an aggregation job — a misnomer to avoid. Reads SessionRecords from app.sessions and
explodes each one into individual event rows for ClickHouse:

```
SessionRecord { session_id, user_id, events: [E1, E2, E3] }
  →  row (user_id, session_id, E1.event_name, E1.timestamp, platform, city)
  →  row (user_id, session_id, E2.event_name, E2.timestamp, platform, city)
  →  row (user_id, session_id, E3.event_name, E3.timestamp, platform, city)
```

The 1-minute tumbling window here is only for **batching inserts** — ClickHouse performs
poorly with thousands of tiny single-row inserts; buffering 1 minute of rows and doing one
bulk INSERT is far more efficient. No aggregation happens at this stage. The actual aggregation
(counts, HLL) is handled by ClickHouse Materialized Views on those raw rows.

**Spark Batch Job**
Runs daily at 02:00 UTC for the T-1 date partition. Reads raw events directly from S3 (not from
Kafka). Groups events by session_id (already attached by client — no re-sessionization needed).
Key advantage over Flink: reads the full T-1 partition including late-arriving events (events
with client_timestamp in T-1 that physically arrived at S3 hours later). These late events
carry the correct session_id from the client → GROUP BY session_id reconstructs complete
sessions even with late arrivals. Writes to funnel_events_historical. Authoritative path for
analyst queries beyond 24h.

**ClickHouse Cluster**
Columnar OLAP store. Four table types:
```
funnel_events_streaming    MergeTree, Flink writes, 7-day TTL
                           ORDER BY (user_id, session_id, timestamp)
                           → events per session physically co-located

funnel_events_historical   MergeTree, Spark writes, no TTL
                           Same schema, same ORDER BY
                           → analysts query this for > 24h ranges

funnel_agg_1m/1h/1d        AggregatingMergeTree
                           Populated by Materialized Views (see below)
                           → dashboards query these, never the raw tables
```

**ClickHouse Materialized Views (Rollup — not a Flink job)**
Defined once in ClickHouse, run forever, triggered automatically on every INSERT into source
tables. No external job needed. Uses AggregatingMergeTree with AggregateFunction state so
partial aggregates merge correctly across inserts:
```sql
CREATE MATERIALIZED VIEW funnel_agg_1m_mv
TO funnel_agg_1m AS
SELECT
  toStartOfMinute(timestamp) AS window_start,
  platform, city, event_name,
  countState()               AS event_count,   -- AggregateFunction(count)
  uniqState(user_id)         AS unique_users   -- AggregateFunction(uniq) = HLL
FROM funnel_events_streaming
GROUP BY window_start, platform, city, event_name;
```
At query time: `SELECT countMerge(event_count), uniqMerge(unique_users) FROM funnel_agg_1m`
This is why "Flink rollup job reading from ClickHouse" is wrong — ClickHouse already handles
this natively, and doing it in Flink would mean reading from ClickHouse (slow, coupling) just
to write back to ClickHouse.

**Query Service**
The most commonly missed component. Analysts and dashboards do not hit ClickHouse directly.
The Query Service sits in front and handles:
```
Time range routing:
  < 24h         → query funnel_events_streaming or funnel_agg_1m
  24h – 30d     → query funnel_events_historical + funnel_agg_1h
  > 30d         → query funnel_events_historical + funnel_agg_1d

Result cache (Redis, TTL 30s):
  Live dashboard refreshes every 30s → cache hit on second refresh
  Prevents ClickHouse from absorbing dashboard polling as query load

Access control:
  Team A can only see events for their own feature flags / experiments
  Row-level security enforced at Query Service, not ClickHouse

Query cost limits:
  Max row scan: 10B rows per query (analyst ad-hoc)
  Timeout: 30s
  Heavy queries → queued, not dropped — analyst gets result eventually
```

**Late Event DLQ (Kafka: late.events.dlq)**
Kafka topic receiving events that Flink's Sessionization Job could not assign to a window
(arrived more than 2h after their client_timestamp). These events are not lost — they wait
here for the Reprocessing Job to pick them up hourly.

**Reprocessing Job**
Spark job, runs hourly. Reads late.events.dlq → sessionizes → re-inserts into the correct
date partition in ClickHouse. ClickHouse MergeTree deduplicates on background merge (idempotent
inserts). Also triggered on demand by the Data Quality Monitor when reconciliation fails.

**Data Quality Monitor**
Daily Spark job. Reads S3 T-1 partition (ground truth) → counts events by type and date.
Reads ClickHouse funnel_events_historical for same partition → counts. If discrepancy > 0.5%
→ alert on-call + trigger Reprocessing Job for that partition. Also checks: conversion rate
anomaly (> 30% drop vs 7-day average), session_id null rate, schema_ver distribution.

**Dashboard Layer**
```
Grafana       → live dashboards, queries funnel_agg_1m via Query Service, auto-refresh 30s
Superset      → analyst ad-hoc SQL, queries funnel_events_historical, results cached
Tableau       → BI / exec dashboards, scheduled extracts from funnel_agg_1d
Scheduled     → Spark reads ClickHouse → generates PDF/email reports (weekly exec summary)
```

---

## Canonical Event Schema

Same as `ml_feature_pipeline.md` — all events flow from the same Kafka topic.

```
Kafka: app.events (partitioned by user_id)
  event_id, user_id, session_id, event_name, entity_ids,
  timestamp, client_meta, schema_ver
```

Funnel-relevant event names:
```
search_submitted      user submitted a search query
restaurant_clicked    user tapped a restaurant card
menu_opened           user opened restaurant menu
item_added_to_cart    user added an item
checkout_started      user tapped checkout
order_placed          order confirmed
order_cancelled       user cancelled before prep
```

---

## Sessionization: Why It's Required for Funnels

A funnel query without sessions is broken:

```
Without sessions:
  User A searched at 9:00am, then ordered at 11:30pm same day.
  Did the search lead to the order? Probably not — 2.5h gap.

With sessions (30min gap rule):
  9:00am search → 9:28am click → 9:31am order    → same session → funnel hit
  9:00am search → [no activity for 2h] → 11:00pm order → different session → not linked
```

Sessionization runs in the shared Flink Sessionization Job (see `ml_feature_pipeline.md`
for full state machine design). Session boundaries are detected by the client SDK
(30min inactivity gap on-device). Server groups by session_id and emits on
session_end event or 30min fallback timer — no server-side time window.

```
output: SessionRecord {
  session_id, user_id, start_ts, end_ts,
  events: List<{ event_name, entity_ids, timestamp }>
}
→ written to: Kafka topic: app.sessions
```

---

## Dual Pipeline Architecture

```
Kafka: app.sessions
  │
  ├── Flink Session Flattener Job  → ClickHouse funnel_events_streaming (fresh, < 5min)
  │                                   ↓ ClickHouse Materialized Views
  │                                   funnel_agg_1m / 1h / 1d (rollup, for dashboards)
  │
  └── Spark Batch Job              → ClickHouse funnel_events_historical (T-1, exact)

Query Service routes by time range to the right table.
Analysts never hit ClickHouse directly.
```

---

## Flink Session Flattener (Fresh Data, <5min latency)

```
Input:  app.sessions Kafka topic
Output: ClickHouse table: funnel_events_streaming

For each SessionRecord, explode into one row per event:

  INSERT INTO funnel_events_streaming
  SELECT
    user_id,
    session_id,
    event_name,
    entity_ids['restaurant_id'] AS restaurant_id,
    timestamp,
    toDate(timestamp)           AS date,
    client_meta['platform']     AS platform,
    client_meta['city']         AS city

Late events: watermark with 2h allowed lateness.
             Events past watermark → DLQ → Spark reprocessing.
```

**ClickHouse table engine:**
```sql
CREATE TABLE funnel_events_streaming (
  user_id       String,
  session_id    String,
  event_name    String,
  restaurant_id String,
  timestamp     DateTime64(3),
  date          Date,
  platform      String,
  city          String
) ENGINE = MergeTree()
PARTITION BY date
ORDER BY (user_id, session_id, timestamp);
```

Why `ORDER BY (user_id, session_id, timestamp)`:
Events for one session are physically co-located → funnel query reads
one contiguous block per session. No shuffle needed.

---

## Spark Batch Job (Historical, Daily Backfill)

```
Input:  S3 raw event log (partitioned by date)
Output: ClickHouse table: funnel_events_historical (same schema)

Runs daily at 02:00 UTC for T-1 partition.
Handles late events correctly: reads full T-1 partition including
events that arrived late but have client timestamp in T-1.

Why Spark for historical, not Flink:
  Flink sliding window over 90 days of data = massive state.
  Spark reads S3 partitions directly, no state management.
  For backfill and correctness, Spark wins.
```

---

## Funnel Query in ClickHouse

ClickHouse has a native `windowFunnel` function built for this:

```sql
-- Conversion funnel: search → click → order, within 1 hour
SELECT
  platform,
  city,
  countIf(funnel_step >= 1) AS reached_search,
  countIf(funnel_step >= 2) AS reached_click,
  countIf(funnel_step >= 3) AS reached_order,
  round(100 * countIf(funnel_step >= 2) / countIf(funnel_step >= 1), 2) AS search_to_click_pct,
  round(100 * countIf(funnel_step >= 3) / countIf(funnel_step >= 2), 2) AS click_to_order_pct
FROM (
  SELECT
    user_id,
    platform,
    city,
    windowFunnel(3600)(   -- 1 hour window
      timestamp,
      event_name = 'search_submitted',
      event_name = 'restaurant_clicked',
      event_name = 'order_placed'
    ) AS funnel_step
  FROM funnel_events_historical
  WHERE date BETWEEN '2026-07-01' AND '2026-07-14'
  GROUP BY user_id, platform, city
)
GROUP BY platform, city
ORDER BY reached_search DESC;
```

```
windowFunnel(3600)(timestamp, cond1, cond2, cond3):
  For each user, finds the longest prefix of conditions
  that occur in order within 3600 seconds.
  Returns: 0 (none), 1 (only search), 2 (search+click), 3 (all three)
```

---

## Pre-Aggregation Pyramid

Raw event queries over 90 days at high QPS are expensive.
Pre-aggregate at multiple granularities:

```
Raw events (1-row-per-event) → keep 7 days → TTL delete
1-minute rollup              → keep 30 days
1-hour rollup                → keep 6 months
1-day rollup                 → keep 2 years
```

Rollups are driven by **ClickHouse Materialized Views** (not a Flink job — see Architecture
Components). MVs trigger on every INSERT into raw tables. AggregatingMergeTree stores partial
aggregate state so concurrent inserts from Flink and Spark merge correctly:

```
funnel_events_streaming  ──MV──►  funnel_agg_1m  ──MV──►  funnel_agg_1h  ──MV──►  funnel_agg_1d
funnel_events_historical ──MV──►  (same targets)
```

Pre-aggregated schema (AggregatingMergeTree, per platform, city, event_name):
```
window_start    DateTime
platform        String
city            String
event_name      String
event_count     AggregateFunction(count)         -- merge with countMerge()
unique_users    AggregateFunction(uniq, String)  -- HLL, merge with uniqMerge(), ~1% error
```

**Dashboard query on pre-aggregated table (fast):**
```sql
SELECT
  city,
  SUM(event_count) FILTER (WHERE event_name = 'search_submitted') AS searches,
  SUM(event_count) FILTER (WHERE event_name = 'order_placed')     AS orders,
  round(100 * orders / searches, 2)                               AS conversion_pct
FROM funnel_agg_1h
WHERE window_start >= now() - INTERVAL 24 HOUR
GROUP BY city
ORDER BY searches DESC;
```

---

## Cohort Analysis

Cohort = group of users who did their first order in the same week.

```sql
-- Week-over-week retention: did cohort users come back?
WITH cohort AS (
  SELECT user_id, toMonday(MIN(timestamp)) AS cohort_week
  FROM funnel_events_historical
  WHERE event_name = 'order_placed'
  GROUP BY user_id
)
SELECT
  c.cohort_week,
  dateDiff('week', c.cohort_week, toMonday(f.timestamp)) AS weeks_since_join,
  COUNT(DISTINCT f.user_id) AS retained_users
FROM cohort c
JOIN funnel_events_historical f ON c.user_id = f.user_id
WHERE f.event_name = 'order_placed'
GROUP BY c.cohort_week, weeks_since_join
ORDER BY c.cohort_week, weeks_since_join;
```

---

## Late Arrival and Reprocessing

```
Problem:   User places order offline (no network). App reconnects 4 hours later.
           Events arrive with client timestamp 4h in the past.

Streaming: Flink 2h watermark tolerance. Beyond 2h → routed to late DLQ.

Reprocessing job (Spark, runs hourly):
  Reads late DLQ → re-inserts into correct date partition in ClickHouse.
  ClickHouse MergeTree: new insert merges with existing data → correct counts.

Reconciliation check (daily):
  Spark re-counts events from S3 raw log (ground truth).
  Compares to ClickHouse aggregates.
  Discrepancy > 0.5% → alert + trigger reprocessing for that partition.
```

---

## Data Quality Checks

```
Check                                   Action
──────────────────────────────────────────────────────────────────
session_id null in funnel event         drop + alert (funnel broken)
event sequence impossible               flag (order_placed before search)
  (e.g., order without prior search)
funnel conversion rate drops > 30%      alert (tracking regression)
unique_users count mismatch > 1%        trigger S3 → ClickHouse reconcile
rollup sum != raw event count           reprocess affected partition
schema_ver mismatch in session events   route to DLQ, hold for schema fix
```

---

## Serving Surfaces

```
Surface                 Source                      Latency SLO
──────────────────────────────────────────────────────────────────────
Live dashboard          funnel_agg_1m (ClickHouse)  < 500ms
Hourly trend chart      funnel_agg_1h               < 1s
Ad-hoc analyst query    funnel_events_historical     < 10s (acceptable)
Cohort analysis         funnel_events_historical     minutes (batch)
Executive weekly report Spark → PDF/Tableau          hours (scheduled)
```

---

## Key Tradeoffs

```
windowFunnel vs self-join for funnel:
  Self-join approach: JOIN event table to itself per step → O(N²) per user.
  windowFunnel: single pass per user, O(N) → orders of magnitude faster.
  → Always use windowFunnel in ClickHouse for funnel analysis.

HyperLogLog for unique users:
  Exact COUNT(DISTINCT user_id) = expensive, requires full set in memory.
  HLL: ~1% error, fixed 12KB per bucket regardless of cardinality.
  → Acceptable for dashboards. Use exact count only for audit/reconciliation.

Flink for fresh + Spark for historical:
  Single Flink job over 90 days = unbounded state, memory pressure.
  Dual path: Flink handles last 24h; Spark handles history.
  Dashboard query routes to fresh table for < 24h, historical for > 24h.
```
