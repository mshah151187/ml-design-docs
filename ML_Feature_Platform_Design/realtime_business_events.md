# Real-Time Business Events Pipeline

Related: `ml_feature_pipeline.md`, `funnel_analytics_pipeline.md`

---

## Problem

Certain business decisions cannot wait for a batch job or even a 5-minute streaming window.
They must act within milliseconds to seconds of the triggering event:

```
Fraud detection:      block a suspicious payment before it completes (< 200ms)
Refund automation:    auto-approve refund if driver_not_arrived pattern confirmed (< 5s)
Surge pricing:        detect demand spike in a zone → update surge multiplier (< 30s)
Account suspension:   N failed logins within M minutes → lock account (< 10s)
```

These require **Complex Event Processing (CEP)**: detecting patterns across multiple events
from the same user/entity in a short time window — not just reacting to a single event.

---

## Canonical Event Schema

Same Kafka topic as other pipelines:

```json
{
  "event_id":        "uuid-v4",
  "user_id":         "u_123456",
  "session_id":      "s_abc789",
  "session_start_ts": 1720951200000,
  "event_name":      "payment_failed",
  "entity_ids":      { "order_id": "ord_88", "payment_method_id": "pm_7" },
  "timestamp":       1720951500000,
  "client_meta":     { "platform": "ios", "ip": "203.0.113.42", "app_ver": "5.142.0" },
  "schema_ver":      3
}
```

`session_id` and `session_start_ts` are attached by the client SDK (client owns session
boundary detection). The CEP job does not use them for grouping — CEP patterns are keyed
by user_id, order_id, or zone_id, not session_id.

Business-critical event names (CEP reacts to these):
```
payment_attempted        user submitted payment
payment_failed           payment declined by processor
payment_succeeded        payment confirmed
order_placed             order confirmed end-to-end
driver_not_arrived       user reported driver no-show
order_cancelled          cancellation before pickup
refund_requested         user requested refund
login_failed             authentication failure
account_flagged          manual fraud flag by ops team
```

Session lifecycle events (CEP filters these out — not business events):
```
session_end              app backgrounded/closed — consumed by Flink Sessionization Job only
```

---

## Architecture Overview

```
Kafka: app.events (partitioned by user_id)
  │
  │  ← reads app.events directly, NOT app.sessions
  │    Reason: CEP must react to individual events immediately.
  │    Session close (session_end or 30min timer) can be 30min away.
  │    Blocking a payment after 30min is useless — the transaction already went through.
  │
  └── Flink CEP Job
        │
        ├── Pre-filter: drop event_name == "session_end" (not a business event)
        │
        ├── Pattern Engine (detects multi-event sequences)
        │
        ├── ML Scoring Service (for ambiguous cases)
        │
        └── Decision Router
              ├── Fraud Block   → Payment Gateway API (< 200ms)
              ├── Refund Auto   → Refund Service API  (< 5s)
              ├── Surge Update  → Surge Pricing Redis  (< 30s)
              └── Account Lock  → Auth Service API     (< 10s)
```

**Why app.events and not app.sessions:**
```
app.sessions emits only when a session CLOSES (session_end event or 30min timer).
For fraud: need to block on the 3rd failed payment NOW, not when the session closes.
For brute force: need to lock account after 5 failed logins within 5 minutes.
For surge: need to detect 500 searches in a zone within 5 minutes.

All of these require reacting to individual events in real time.
Waiting for a session boundary would break every CEP pattern in this pipeline.
```

---

## Flink CEP: Complex Event Processing

Flink's CEP library lets you define event patterns declaratively.
The pattern engine runs statefully per key (user_id, driver_id, zone_id).

### Pattern 1: Fraud — Multiple Failed Payments

```java
Pattern<AppEvent, ?> fraudPattern = Pattern
  .<AppEvent>begin("first_failure")
    .where(e -> e.eventName.equals("payment_failed"))
  .followedByAny("more_failures")
    .where(e -> e.eventName.equals("payment_failed"))
    .times(2)                           // 2 more = 3 total
  .within(Time.minutes(10));            // all within 10 minutes

// On pattern match:
//   → call ML scoring model with user features
//   → if risk_score > 0.85 → block payment + flag account
//   → else → pass through (false positive avoidance)
```

### Pattern 2: Refund Automation — Driver No-Show

```java
Pattern<AppEvent, ?> noShowPattern = Pattern
  .<AppEvent>begin("order_confirmed")
    .where(e -> e.eventName.equals("order_placed"))
  .followedBy("driver_report")
    .where(e -> e.eventName.equals("driver_not_arrived"))
  .followedBy("cancellation")
    .where(e -> e.eventName.equals("order_cancelled"))
  .within(Time.minutes(30));

// On match:
//   → cross-check with GPS: was driver actually near pickup location?
//   → if driver_distance_at_pickup > 500m → auto-approve full refund
//   → else → route to manual review queue
```

### Pattern 3: Account Security — Brute Force Login

```java
Pattern<AppEvent, ?> bruteForcePattern = Pattern
  .<AppEvent>begin("first_fail")
    .where(e -> e.eventName.equals("login_failed"))
  .followedByAny("repeat_fails")
    .where(e -> e.eventName.equals("login_failed"))
    .times(4)                           // 5 total
  .within(Time.minutes(5));

// On match:
//   → lock account (Auth Service API)
//   → send 2FA re-verification email
//   → emit security_alert event to ops dashboard
```

### Pattern 4: Surge Pricing — Demand Spike

```java
// Keyed by zone_id (H3 level-7 cell), not user_id
Pattern<AppEvent, ?> surgePattern = Pattern
  .<AppEvent>begin("requests")
    .where(e -> e.eventName.equals("search_submitted"))
    .times(500).greedy()               // 500+ searches in window
  .within(Time.minutes(5));

// On match:
//   → compute demand/supply ratio: search_count / active_drivers_in_zone
//   → if ratio > 3.0 → HSET surge:{zone_id} multiplier 1.8
//   → Feed Service reads surge multiplier at serve time
```

---

## ML Scoring for Ambiguous Cases

Not all patterns are deterministic. For fraud, a rule like "3 failed payments = block" has
high false-positive rate (user with bad card legitimately retrying).

```
Pattern match → Feature fetch → ML Model → Decision

Feature fetch (< 5ms from Redis):
  user:{id}:features
    orders_last_30d         (established user vs new account)
    avg_order_value_30d     (order amount consistent with history?)
    is_new_user             (higher risk)
    device_fingerprint_seen (known device vs new device)
    ip_country_match        (IP country vs billing country)

ML Model (XGBoost, loaded in-process in Flink task):
  input:  pattern_type + user_features + event_context
  output: risk_score ∈ [0.0, 1.0]

Decision thresholds:
  risk_score > 0.90  → auto-block (high confidence fraud)
  risk_score 0.60–0.90 → soft challenge (3DS re-auth, CAPTCHA)
  risk_score < 0.60  → pass through (low risk)
```

**Why ML model in-process (not RPC):**
```
RPC to separate model server adds ~10-50ms.
For fraud decision SLO of < 200ms, every ms counts.
XGBoost is small (< 50MB), loads into Flink task JVM.
Model updated via Flink operator reload (no restart needed).
```

---

## Exactly-Once Semantics

Fraud block that fires twice = double account suspension. Must not happen.

```
Flink exactly-once:
  checkpoint.mode = EXACTLY_ONCE
  Kafka → Flink: transactional consumer (offset committed only on checkpoint)
  Flink → Kafka output: two-phase commit (KafkaProducer in transaction)

For external API calls (Payment Gateway, Auth Service):
  Flink cannot do XA transactions across Kafka + HTTP.
  → Use idempotency keys on all API calls.
  → Payment Gateway: idempotency_key = event_id + pattern_type
  → Auth Service lock: idempotency_key = user_id + window_start_ts
  → Duplicate API call with same key → no-op on receiver side.
```

---

## Late Events and Out-of-Order Delivery

```
Problem:  login_failed events arrive out of order (Kafka partition rebalance
          or network jitter). Pattern expects A then B then C.
          If C arrives before B, does pattern still match?

Flink CEP handles this:
  Event-time processing: events sorted by timestamp before pattern matching.
  Watermark: 30s allowed lateness for business event patterns.
  Events within 30s of watermark → buffered, sorted, then matched.
  Events beyond 30s → routed to late-event Kafka topic.

Late event handling for fraud:
  If 3rd failed payment arrives 35s late (past watermark):
    → pattern already processed without it → fraud missed
    → late DLQ: separate Spark job reprocesses for audit/reporting only
    → real-time action window already closed (user got through or didn't)
  Acceptable: 30s window covers 99.9% of network jitter; rare late events
  accepted as miss rather than degrading real-time path.
```

---

## Dedup

```
Client SDK may retry on timeout → same payment_failed event sent twice.

Flink dedup operator (same as other pipelines):
  Bloom filter on event_id, 24h TTL.
  Duplicate → dropped before reaching CEP pattern engine.
  Why bloom filter not HashMap: 50M events/day × 24h = memory blowup with HashMap.
  Bloom filter false positive rate 0.01% → acceptable (rare legitimate event dropped
  vs constant duplicate noise without dedup).
```

---

## Decision Router and External Action

```
On pattern + decision:
  Flink emits DecisionEvent to Kafka: app.decisions
    { user_id, pattern_type, decision, risk_score, evidence_event_ids, ts }

Decision Consumer (separate microservice, reads app.decisions):
  FRAUD_BLOCK      → POST /payments/block     (Payment Gateway, idempotent)
  REFUND_AUTO      → POST /refunds/approve    (Refund Service, idempotent)
  ACCOUNT_LOCK     → POST /auth/lock          (Auth Service, idempotent)
  SURGE_UPDATE     → HSET surge:{zone}        (Redis, directly in Flink)

Why separate Decision Consumer (not direct from Flink):
  Flink task failure during API call = retry storm.
  Kafka decouples: Flink writes decision once → consumer handles retries,
  circuit breakers, and DLQ for failed API calls independently.
```

---

## Monitoring and Alerting

```
Metric                              Threshold       Alert
──────────────────────────────────────────────────────────────────────
fraud_block_rate spikes > 5x        immediate       attack in progress
fraud_false_positive_rate > 5%      alert           model drift, tune threshold
pattern_match_latency p99 > 150ms   alert           Flink lag or state bloat
late_events_rate > 1%               alert           Kafka consumer lag
decision_api_failure_rate > 0.1%    alert           downstream service down
refund_auto_approval_rate drop      alert           GPS data unavailable
account_lock_rate spikes            immediate       credential stuffing attack
```

---

## Key Tradeoffs

```
CEP vs rules engine:
  Rules engine (Drools, custom): fast, interpretable, easy to update by ops.
  Flink CEP: handles complex sequences, event-time ordering, stateful windows.
  → Use Flink CEP for sequence patterns; use rules engine for single-event checks
    (e.g., "order value > $500 on new account → review").

ML model in-process vs RPC:
  In-process: no network hop, < 1ms inference, but model updates require
              Flink operator reload.
  RPC:        independent model deployment, but adds 10-50ms per call.
  → For p99 < 200ms SLO: in-process. Coordinate model updates with Flink
    via versioned model store (S3 + model_version metadata).

Exactly-once vs at-least-once:
  Exactly-once: higher checkpoint overhead (2-phase commit).
  At-least-once + idempotent APIs: simpler, nearly as safe.
  → Fraud and account lock: at-least-once + idempotency keys on API calls.
    Duplicate API call = safe no-op. Simpler than distributed XA.

Bloom filter vs HashMap for dedup:
  HashMap: exact, but O(N) memory for N events in window.
  Bloom filter: O(1) space per element, 0.01% false positives.
  → Bloom filter wins at 50M events/day scale.
```
