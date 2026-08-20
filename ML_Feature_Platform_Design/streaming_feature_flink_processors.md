# Streaming Feature Flink Processors

Related: `ml_feature_pipeline.md`

---

## Input: SessionRecord

All processors below read from `Kafka: app.sessions`. Each message is a `SessionRecord`
emitted by the shared Flink Sessionization Job when a session closes.

```java
public class SessionRecord {
    public String sessionId;
    public String userId;
    public long   startTs;          // client-side, attached to first event
    public long   endTs;            // from session_end event or last buffered event
    public List<EventRecord> events;
}

public class EventRecord {
    public String              eventId;
    public String              eventName;
    public Map<String, String> entityIds;   // e.g. { "restaurant_id": "r_99", "cuisine_type": "indian" }
    public long                timestamp;
    public Map<String, String> clientMeta;  // platform, city, app_ver
}
```

All processors are **keyed by userId** — Flink guarantees all SessionRecords for
the same user are processed by the same task node, so per-user state is local.

---

## Processor 1 — Session Feature Extractor (Stateless)

**Computes:** last_session_id, last_session_ts, session_duration_sec, click_to_order_flag,
searched_cuisines, last_event_name, last_search_ts

**Flink pattern:** `RichFlatMapFunction` — no state needed. Each SessionRecord is
self-contained. Every field is derived from the events list within one session.

```java
public class SessionFeatureExtractor
        extends RichFlatMapFunction<SessionRecord, Void> {

    private transient Pipeline redisPipeline;
    private transient Jedis    jedis;

    @Override
    public void open(Configuration params) {
        jedis = new Jedis("redis-host", 6379);
    }

    @Override
    public void flatMap(SessionRecord session, Collector<Void> out) {
        long   durationSec     = (session.endTs - session.startTs) / 1000;
        int    eventCount      = session.events.size();
        String lastEventName   = session.events.isEmpty() ? "" :
                                 session.events.get(session.events.size() - 1).eventName;

        boolean clickToOrderFlag = session.events.stream()
            .anyMatch(e -> e.eventName.equals("order_placed"));

        // Cuisines seen in search events this session
        String searchedCuisines = session.events.stream()
            .filter(e -> e.eventName.equals("search_submitted"))
            .map(e -> e.entityIds.getOrDefault("cuisine_filter", ""))
            .filter(c -> !c.isEmpty())
            .distinct()
            .collect(Collectors.joining(","));

        // Timestamp of last search in this session (empty string if no search)
        String lastSearchTs = session.events.stream()
            .filter(e -> e.eventName.equals("search_submitted"))
            .mapToLong(e -> e.timestamp)
            .max()
            .stream()
            .mapToObj(String::valueOf)
            .findFirst()
            .orElse("");

        // Batch all field updates in one Redis pipeline round trip
        Pipeline pipe = jedis.pipelined();
        pipe.hset("user:" + session.userId + ":features", Map.of(
            "last_session_id",      session.sessionId,
            "last_session_ts",      String.valueOf(session.endTs),
            "session_duration_sec", String.valueOf(durationSec),
            "event_count",          String.valueOf(eventCount),
            "click_to_order_flag",  String.valueOf(clickToOrderFlag),
            "searched_cuisines",    searchedCuisines,
            "last_event_name",      lastEventName,
            "last_search_ts",       lastSearchTs
        ));
        pipe.sync();
    }

    @Override
    public void close() { if (jedis != null) jedis.close(); }
}
```

**Why stateless:** Every feature is derived from the current SessionRecord alone.
No history across sessions is needed. No Flink state → no checkpoint overhead →
this processor restarts instantly on failure (just replay the Kafka offset).

---

## Processor 2 — Sliding Window Order Metrics (ListState + Eviction)

**Computes:** orders_last_7d, orders_last_30d, avg_order_value_30d, is_new_user

**Flink pattern:** `KeyedProcessFunction` with `ListState<Tuple2<Long, Double>>`.
Each list entry is `(order_timestamp, order_value)`. On every session close:
1. Add new order events from this session to state
2. Evict entries older than 30 days
3. Compute counts and average across remaining entries
4. Derive is_new_user from a separate lifetime counter (ValueState)

```java
public class SlidingWindowOrderMetricsProcessor
        extends KeyedProcessFunction<String, SessionRecord, Void> {

    private static final long WINDOW_7D  = 7L  * 24 * 3600 * 1000;
    private static final long WINDOW_30D = 30L * 24 * 3600 * 1000;

    // Sliding window: (timestamp, order_value) per order event
    private ListState<Tuple2<Long, Double>> orderHistory;

    // Lifetime total — never evicted, used for is_new_user
    private ValueState<Integer> totalOrderCount;

    private transient Jedis jedis;

    @Override
    public void open(Configuration params) {
        orderHistory = getRuntimeContext().getListState(
            new ListStateDescriptor<>("orderHistory",
                Types.TUPLE(Types.LONG, Types.DOUBLE)));

        totalOrderCount = getRuntimeContext().getState(
            new ValueStateDescriptor<>("totalOrderCount", Types.INT));

        jedis = new Jedis("redis-host", 6379);
    }

    @Override
    public void processElement(SessionRecord session, Context ctx, Collector<Void> out)
            throws Exception {

        // 1. Extract order events from this session
        List<Tuple2<Long, Double>> newOrders = session.events.stream()
            .filter(e -> e.eventName.equals("order_placed"))
            .map(e -> Tuple2.of(
                e.timestamp,
                Double.parseDouble(e.entityIds.getOrDefault("order_value", "0"))))
            .collect(Collectors.toList());

        // 2. Append to ListState
        for (Tuple2<Long, Double> order : newOrders) {
            orderHistory.add(order);
        }

        // 3. Evict entries older than 30d (the longer window)
        long cutoff30d = session.endTs - WINDOW_30D;
        List<Tuple2<Long, Double>> history = new ArrayList<>();
        orderHistory.get().forEach(history::add);
        history.removeIf(t -> t.f0 < cutoff30d);   // remove stale entries
        orderHistory.update(history);               // write back compacted list

        // 4. Compute sliding window features
        long   cutoff7d    = session.endTs - WINDOW_7D;
        long   orders7d    = history.stream().filter(t -> t.f0 >= cutoff7d).count();
        long   orders30d   = history.size();
        double avgValue30d = history.stream().mapToDouble(t -> t.f1).average().orElse(0.0);

        // 5. Update lifetime counter and derive is_new_user
        int prevTotal    = totalOrderCount.value() == null ? 0 : totalOrderCount.value();
        int newTotal     = prevTotal + newOrders.size();
        totalOrderCount.update(newTotal);
        boolean isNewUser = newTotal < 3;

        // 6. Write all features atomically via Redis pipeline
        Pipeline pipe = jedis.pipelined();
        pipe.hset("user:" + session.userId + ":features", Map.of(
            "orders_last_7d",      String.valueOf(orders7d),
            "orders_last_30d",     String.valueOf(orders30d),
            "avg_order_value_30d", String.format("%.2f", avgValue30d),
            "orders_total",        String.valueOf(newTotal),
            "is_new_user",         String.valueOf(isNewUser)
        ));
        pipe.sync();
    }

    @Override
    public void close() { if (jedis != null) jedis.close(); }
}
```

**Why ListState and not Flink's built-in SlidingWindow:**
Flink's `SlidingWindow(30d, 1h)` would create 720 overlapping window copies
per user (30d / 1h = 720 slides). Each window holds a full copy of events
within its range → memory explosion. Manual eviction with ListState keeps
exactly one copy of the data, evicting only what's expired.

**State size estimate:**
1 order event = ~16 bytes (long + double). User with 1 order/day over 30d = 30 entries = 480 bytes.
At 10M users → 4.8GB total state → fits comfortably in RocksDB state backend.

---

## Processor 3 — Cuisine Ring Buffer (ValueState with Bounded Deque)

**Computes:** cuisines_last_10_clicks

**Flink pattern:** `KeyedProcessFunction` with `ValueState<ArrayDeque<String>>`.
A ring buffer of the last 10 cuisines the user clicked. Older entries fall off
the front as new ones are added to the back.

```java
public class CuisineRingBufferProcessor
        extends KeyedProcessFunction<String, SessionRecord, Void> {

    private static final int MAX_SIZE = 10;

    private ValueState<ArrayDeque<String>> cuisineBuffer;
    private transient Jedis jedis;

    @Override
    public void open(Configuration params) {
        cuisineBuffer = getRuntimeContext().getState(
            new ValueStateDescriptor<>("cuisineBuffer",
                TypeInformation.of(new TypeHint<ArrayDeque<String>>() {})));
        jedis = new Jedis("redis-host", 6379);
    }

    @Override
    public void processElement(SessionRecord session, Context ctx, Collector<Void> out)
            throws Exception {

        // 1. Extract cuisines from restaurant_clicked events (preserving order)
        List<String> sessionCuisines = session.events.stream()
            .filter(e -> e.eventName.equals("restaurant_clicked"))
            .map(e -> e.entityIds.getOrDefault("cuisine_type", ""))
            .filter(c -> !c.isEmpty())
            .collect(Collectors.toList());

        if (sessionCuisines.isEmpty()) return;  // nothing to update

        // 2. Load ring buffer from state (initialize if first session)
        ArrayDeque<String> buffer = cuisineBuffer.value();
        if (buffer == null) buffer = new ArrayDeque<>(MAX_SIZE);

        // 3. Add new cuisines, evict oldest when over capacity
        for (String cuisine : sessionCuisines) {
            if (buffer.size() >= MAX_SIZE) {
                buffer.pollFirst();  // evict oldest
            }
            buffer.addLast(cuisine);
        }
        cuisineBuffer.update(buffer);

        // 4. Write to Redis as JSON array
        // e.g. ["indian","pizza","sushi","burgers","indian"]
        String json = "[\"" + String.join("\",\"", buffer) + "\"]";
        jedis.hset("user:" + session.userId + ":features",
            "cuisines_last_10_clicks", json);
    }

    @Override
    public void close() { if (jedis != null) jedis.close(); }
}
```

**Why ValueState<Deque> and not ListState:**
ListState would require iterating all entries on every read to reconstruct
the ordered deque. ValueState stores the deque directly — one `state.value()`
call returns the ordered buffer ready to use. Tradeoff: ValueState serializes
the entire object on every `state.update()`, but at 10 strings × ~10 bytes = ~100
bytes per user, this is negligible.

**Flink serialization note:**
`ArrayDeque<String>` requires a Kryo serializer. For production, register a
custom `TypeSerializer<ArrayDeque<String>>` to avoid Kryo's reflection overhead
and to make the type compatible with Flink's schema migration.

---

## Processor 4 — Engagement Score with Exponential Time Decay (ValueState with Decay)

**Computes:** engagement_score — a single float summarizing how actively a user has
been engaging recently, where older activity decays over time.

**Flink pattern:** `KeyedProcessFunction` with `ValueState<Tuple2<Double, Long>>`
storing `(score, last_update_ts)`. Each new session adds a contribution based on
event mix. The old score is decayed proportionally to how much time has elapsed
since the last session — no timer needed, decay is computed lazily on next event.

```java
public class UserEngagementScoreProcessor
        extends KeyedProcessFunction<String, SessionRecord, Void> {

    // Score halves every 7 days: λ = ln(2) / (7 days in ms)
    private static final double LAMBDA =
        Math.log(2) / (7.0 * 24 * 3600 * 1000);

    // Points contributed per event type within a session
    private static final Map<String, Integer> EVENT_WEIGHTS = Map.of(
        "search_submitted",   1,
        "restaurant_clicked", 3,
        "item_added_to_cart", 5,
        "checkout_started",   7,
        "order_placed",      10
    );

    // (decayed_score, last_session_end_ts)
    private ValueState<Tuple2<Double, Long>> scoreState;
    private transient Jedis jedis;

    @Override
    public void open(Configuration params) {
        scoreState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("scoreState",
                Types.TUPLE(Types.DOUBLE, Types.LONG)));
        jedis = new Jedis("redis-host", 6379);
    }

    @Override
    public void processElement(SessionRecord session, Context ctx, Collector<Void> out)
            throws Exception {

        // 1. Compute this session's contribution
        int sessionContribution = session.events.stream()
            .mapToInt(e -> EVENT_WEIGHTS.getOrDefault(e.eventName, 0))
            .sum();

        // 2. Load existing score and apply time decay since last session
        Tuple2<Double, Long> state  = scoreState.value();
        double decayedScore = 0.0;
        if (state != null) {
            long   deltaMs = session.endTs - state.f1;
            // Continuous decay: score × e^(-λ × Δt)
            // e.g. 7 days gap → e^(-ln2) = 0.5 → score halves
            decayedScore = state.f0 * Math.exp(-LAMBDA * deltaMs);
        }

        // 3. Add new session contribution
        double newScore = decayedScore + sessionContribution;
        scoreState.update(Tuple2.of(newScore, session.endTs));

        // 4. Write to Redis
        jedis.hset("user:" + session.userId + ":features",
            "engagement_score", String.format("%.4f", newScore));
    }

    @Override
    public void close() { if (jedis != null) jedis.close(); }
}
```

**Example walk-through:**
```
Session 1 (T=0):        search + click + order → contribution = 1 + 3 + 10 = 14
                        score = 0 + 14 = 14.00   state=(14.00, T=0)

Session 2 (T=7 days):   click + order → contribution = 3 + 10 = 13
                        decay: 14.00 × e^(-λ×7d) = 14 × 0.5 = 7.00
                        score = 7.00 + 13 = 20.00  state=(20.00, T=7d)

Session 3 (T=21 days):  search only → contribution = 1
                        decay: 20.00 × e^(-λ×14d) = 20 × 0.25 = 5.00
                        score = 5.00 + 1 = 6.00   state=(6.00, T=21d)

Inactive user (T=35 days, no session):  score in state = 6.00
  Next session at T=42d:  decay = 6.00 × e^(-λ×21d) = 6 × 0.125 = 0.75
```

**Why lazy decay (not a timer):**
A Flink timer per user would fire every N minutes even for inactive users → millions
of empty timer callbacks → wasted CPU. Lazy decay: multiply on the next real event.
Result is mathematically identical to timer-based decay (continuous exponential function).

---

## Wiring All 4 Processors in One Flink Job

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
env.enableCheckpointing(30_000, CheckpointingMode.EXACTLY_ONCE);

// Use RocksDB for large state (order history can be GBs across all users)
env.setStateBackend(new EmbeddedRocksDBStateBackend());

Properties kafkaProps = new Properties();
kafkaProps.setProperty("bootstrap.servers", "kafka:9092");
kafkaProps.setProperty("group.id", "ml-feature-pipeline");

// Source: Kafka app.sessions, keyed by userId
KeyedStream<SessionRecord, String> sessions = env
    .addSource(new FlinkKafkaConsumer<>(
        "app.sessions",
        new SessionRecordDeserializationSchema(),
        kafkaProps))
    .keyBy(s -> s.userId);

// Each processor is an independent operator on the same keyed stream.
// Flink shares the key grouping — no extra shuffle per processor.
sessions.flatMap(new SessionFeatureExtractor());        // P1: stateless
sessions.process(new SlidingWindowOrderMetricsProcessor()); // P2: ListState
sessions.process(new CuisineRingBufferProcessor());        // P3: ValueState<Deque>
sessions.process(new UserEngagementScoreProcessor());      // P4: ValueState<Tuple>

env.execute("ML Feature Pipeline");
```

**State isolation:** Each processor has its own named state descriptors (`"orderHistory"`,
`"cuisineBuffer"`, `"scoreState"`). Even though they run on the same keyed stream,
their state is completely independent. A checkpoint failure in one does not affect others.

---

## State Backend and Checkpointing

```
HashMapStateBackend  (default, heap)
  Good for:   small state (< 1GB total), fast reads
  Bad for:    large ListState like orderHistory at 10M users

EmbeddedRocksDBStateBackend  (disk-backed)
  Good for:   large state (orderHistory at 10M users × 30 orders × 16 bytes = 4.8GB)
  Tradeoff:   reads/writes are ~10× slower than heap (JNI + disk)
  Use when:   state doesn't fit in heap

Checkpoint interval: 30s
  On Flink task failure → restore from last checkpoint → replay 30s of Kafka lag
  EXACTLY_ONCE + Kafka transactions → no duplicate writes to Redis on recovery
```

---

## Feature → Processor Map

```
Feature                    Processor                   Flink state type
──────────────────────────────────────────────────────────────────────────────
last_session_id            P1 SessionFeatureExtractor  none (stateless)
last_session_ts            P1 SessionFeatureExtractor  none
session_duration_sec       P1 SessionFeatureExtractor  none
click_to_order_flag        P1 SessionFeatureExtractor  none
searched_cuisines          P1 SessionFeatureExtractor  none
last_search_ts             P1 SessionFeatureExtractor  none
last_event_name            P1 SessionFeatureExtractor  none

orders_last_7d             P2 SlidingWindowOrders      ListState<(ts, value)>
orders_last_30d            P2 SlidingWindowOrders      ListState<(ts, value)>
avg_order_value_30d        P2 SlidingWindowOrders      ListState<(ts, value)>
orders_total               P2 SlidingWindowOrders      ValueState<Integer>
is_new_user                P2 SlidingWindowOrders      derived from orders_total

cuisines_last_10_clicks    P3 CuisineRingBuffer        ValueState<ArrayDeque>

engagement_score           P4 EngagementScore          ValueState<(score, ts)>
```
