# Need of Disaggregated Inference

**Interview question:** Why do Prefill/Decode disaggregation at all?

**One-line answer:** At high QPS, Prefill and Decode have different execution characteristics and different latency objectives. When they share one engine, large Prefills disturb active Decode iterations. Disaggregation isolates that interference and lets us tune and scale the phases independently.

## 1. Start with the two user-visible SLOs

```text
request arrives
      |
      | PREFILL
      v
first output token          <- TTFT
      |
      | DECODE
      v
token -> token -> token     <- TPOT / ITL / TBT
```

A request can have good TTFT but poor TPOT, or poor TTFT but smooth streaming. At scale we normally care about both.

## 2. The interference problem

An aggregated engine serves Prefill and Decode on the same workers.

```text
iteration 1:
    decode A
    decode B
    decode C

iteration 2:
    decode A
    decode B
    decode C
    + large 32K-token Prefill D

iteration 3:
    decode A
    decode B
    decode C
```

The second iteration is much heavier. Existing requests observe:

```text
token ---- token ---------------- token
                     ^
                     |
             Prefill interference
```

The new request wants aggressive Prefill to minimize TTFT. Existing requests want short, predictable iterations to minimize TPOT.

That is the central conflict.

DistServe describes this as strong Prefill/Decode interference and notes that colocating both phases forces the serving system to trade one latency objective against the other.

## 3. Why not just prioritize one phase?

### Prefill priority

```text
new long prompt arrives
      |
      v
run Prefill aggressively
```

Result:

```text
new request TTFT      -> good
existing Decode TPOT  -> worse
```

### Decode priority

```text
keep running Decode iterations
Prefill waits
```

Result:

```text
existing Decode TPOT -> good
new request TTFT     -> worse
```

So priority policy mostly chooses **which SLO absorbs the interference**.

## 4. Chunked Prefill helps, but does not eliminate the conflict

Instead of:

```text
32K Prefill
```

run:

```text
4K Prefill
Decode
4K Prefill
Decode
...
```

Large chunks:

```text
fewer interruptions
but longer interruption per chunk
```

Small chunks:

```text
shorter interruptions
but more frequent interruptions
and more scheduling overhead
```

Chunking is an excellent aggregated-serving technique, but it manages interference rather than removing it.

Current NVIDIA Dynamo documentation explicitly describes this mixed-iteration tradeoff.

## 5. Why the phases naturally want different workers

At first order:

```text
PREFILL
-------
many prompt tokens processed together
large GEMMs / high arithmetic intensity
compute-heavy
TTFT-sensitive


DECODE
------
roughly one new token per active sequence per step
repeated model-weight and KV-cache reads
lower arithmetic intensity
memory-bandwidth / KV-capacity sensitive
TPOT-sensitive
```

Splitwise's characterization is built around exactly this difference: prompt processing is compute intensive, while token generation is much more memory intensive and often underutilizes available compute.

This means one aggregated worker configuration is inherently a compromise.

## 6. The second problem: configuration coupling

Suppose profiling tells us:

```text
best Prefill profile:
    TP2
    small request batch
    large token work unit
    context-optimized kernels

best Decode profile:
    TP4
    high sequence concurrency
    large persistent KV arena
    decode-specialized kernels
```

An aggregated engine has to use one:

```text
TP setting
PP setting
hardware shape
scheduler
memory-allocation policy
token budget
kernel/runtime configuration
```

for both phases.

Disaggregation removes this coupling.

DistServe explicitly identifies independent resource allocation and phase-specific parallelism as a major benefit in addition to interference isolation.

## 7. The third problem: capacity coupling

Traffic can change radically.

```text
Workload A:
    average ISL = 32K
    average OSL = 200
    -> Prefill-heavy

Workload B:
    average ISL = 500
    average OSL = 4K
    -> Decode-heavy
```

Aggregated serving scales both forms of capacity together.

Disaggregation creates:

```text
Prefill Pool:
    P1 P2 P3 ...

Decode Pool:
    D1 D2 D3 D4 ...
```

and makes independent capacity planning possible.

The exact number of P and D workers is a later topic; the key point is that disaggregation creates the control knob.

## 8. The disaggregated execution path

```text
Request
   |
   v
Prefill worker P3
   |
   | computes prompt
   | creates KV cache
   |
   +----------- KV -----------> Decode worker D7
                                  |
                                  v
                               token
                                  |
                                  v
                               token
```

Now:

```text
large Prefill
    primarily competes with Prefill work

long Decode
    primarily consumes Decode/KV capacity
```

## 9. What we gain

### Interference isolation
Long Prefills no longer directly stretch unrelated Decode iterations.

### Independent parallelism
For example, Prefill and Decode may use different TP/PP configurations.

### Independent hardware
Prefill can emphasize dense-math throughput; Decode can emphasize memory bandwidth/capacity and cost efficiency.

### Independent scheduler tuning
Prefill can optimize prompt work for TTFT; Decode can optimize continuous sequence batching for TPOT.

### Independent scaling
Eventually, P and D capacity can be resized according to input-length and output/concurrency pressure.

## 10. What disaggregation costs

We now have to move KV state:

```text
P ---> KV transfer ---> D
```

That introduces:

```text
transfer latency
network bandwidth
orchestration
failure modes
topology constraints
pool fragmentation
```

So disaggregation is **not automatically better**.

For:

```text
short prompt
low concurrency
small model
slow transfer network
```

an aggregated engine may be faster and simpler.

Current Dynamo guidance explicitly makes this point.

## 11. Interview reasoning chain

```text
1. Requests expose two latency objectives:
       TTFT and TPOT.

2. Prefill and Decode are different GPU workloads:
       P = compute-heavy
       D = bandwidth/KV-heavy.

3. On one engine, long Prefills disturb active Decode iterations.
       -> TTFT/TPOT interference.

4. Prioritizing one phase hurts the other.
   Chunking reduces, but does not remove, the conflict.

5. Colocation also forces both phases to share parallelism,
   batching, memory and hardware choices.

6. Separate pools isolate the phases and permit independent
   tuning and scaling.

7. The price is KV transfer and extra orchestration.

8. Therefore disaggregation wins only when isolation and
   specialization benefits exceed that transfer cost.
```

## 12. Short interview answer

> **Prefill/Decode disaggregation exists because the two phases have different execution characteristics and different user-facing SLOs. Prefill processes many prompt tokens in parallel, is typically compute-heavy, and determines TTFT. Decode produces tokens autoregressively, repeatedly streams model weights and KV state, is much more memory-bandwidth/capacity sensitive, and determines TPOT. On an aggregated engine, a large Prefill makes iterations longer and interferes with requests already decoding, so improving the new request's TTFT can hurt existing requests' TPOT. Chunked Prefill reduces this interference but doesn't eliminate it. Separating the phases isolates those workloads and lets us independently tune parallelism, batching, memory, hardware and eventually pool size. The tradeoff is KV transfer plus orchestration, so disaggregation is valuable when those benefits exceed the transfer overhead.**

## 13. Memory note

```text
WHY DISAGGREGATION?

Goal:
    meet TTFT + TPOT at high QPS

Main problem:
    Prefill / Decode interference

Why:
    Prefill = compute-heavy
    Decode  = memory/KV-heavy
    but aggregated workers share one execution schedule

Secondary benefit:
    independent:
        TP / PP
        batching
        hardware
        KV policy
        kernels
        capacity

Cost:
    KV transfer + orchestration

Conclusion:
    powerful, but not universally optimal
```

## References

Primary references behind this note:

- Splitwise: Efficient Generative LLM Inference Using Phase Splitting, ISCA 2024.
- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.
- NVIDIA Dynamo, Disaggregated Serving.
- NVIDIA Dynamo, Disaggregation and Performance Tuning.
