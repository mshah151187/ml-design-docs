# Grammar Guided Decoding

## What Problem Does It Solve?

LLMs generate tokens one at a time. At each step, the model outputs a probability
distribution over its entire vocabulary (~32K to 128K tokens) and samples one token.

Nothing stops the model from generating invalid output:
```json
{ "title": "data engineer", "loc:  ← missing closing quote, breaks JSON parser
```
or hallucinating fields:
```json
{ "title": "engineer", "salary": ">$150K" }  ← "salary" not in our schema
```

In production, you cannot afford JSON parse errors or hallucinated fields. Every QU
output must be machine-readable. Grammar guided decoding enforces this at the token
level — invalid tokens are blocked before they can be sampled.

---

## What is a Grammar?

A grammar is a formal set of rules that define what strings are valid.

Example: a simple JSON grammar rule says:
- An object starts with `{`
- Followed by zero or more `"key": value` pairs separated by `,`
- Ends with `}`

A **context-free grammar (CFG)** expresses these rules recursively:
```
object  → '{' members '}'
members → pair | pair ',' members | ε
pair    → string ':' value
value   → string | number | object | array | 'true' | 'false' | 'null'
```

Given a schema like:
```json
{ "router": "criteria" | "rewrite" | "unsupported" | "blocked",
  "tags": { "title": string, "location": string, ... } }
```

You can express exactly which JSON structures are valid as a grammar.

---

## How Grammar Guided Decoding Works

At each decoding step, instead of sampling freely from all ~100K vocab tokens, the
system computes which tokens are LEGAL given the grammar state and the tokens generated
so far. It masks all illegal tokens to probability 0 before sampling.

```
Step-by-step example, generating: { "router": "criteria" }

Token 1: only '{' is valid to start an object    → mask everything except '{'
Token 2: only '"' is valid after '{'             → mask everything except '"'
Token 3: only valid field names: "router", "tags", "rewrite", "safety"
         → mask everything except those prefixes
...
Token N: after "router": , only valid values: "criteria", "rewrite", "unsupported", "blocked"
         → mask all other strings
```

The model still uses its learned probability distribution to CHOOSE between valid tokens —
grammar just eliminates impossible choices.

```
Without grammar:  P(token) over all vocab       → may sample invalid tokens
With grammar:     P(token | token is valid) = P(token) / Σ P(valid tokens)
                                                → renormalize over valid tokens only
```

---

## The Constraint Mechanism: Token Mask

Concretely, before softmax sampling:

```python
logits = model.forward(input_ids)          # raw logits over full vocab
mask = grammar.get_valid_token_mask()      # True for valid next tokens
logits[~mask] = float('-inf')              # block invalid tokens
token = sample(softmax(logits))            # sample from valid tokens only
grammar.advance(token)                     # update grammar state
```

The grammar tracks state: after seeing `{ "router": `, the grammar knows we're inside
the value position of the "router" key, and only the 4 allowed enum values are valid.

---

## xgrammar (Used in LinkedIn's QU Paper)

xgrammar is an open-source library (from MLC team) that implements grammar guided decoding
efficiently for LLM inference.

Key features:
- Compiles grammar into a pushdown automaton for fast state transitions
- Precomputes token masks for common grammar states (avoids recomputing each step)
- Integrates with vLLM and other serving frameworks
- Supports JSON Schema, EBNF grammars, regex-constrained generation

The LinkedIn QU paper uses xgrammar to enforce the QU output schema at inference time.
Result: JSON parse success rate 99.9%–100% across prompt/model iterations.

---

## What Grammar Guided Decoding is NOT

### Not Speculative Decoding

These are completely different techniques:

| | Grammar Guided Decoding | Speculative Decoding |
|---|---|---|
| **Goal** | Output correctness / structure | Inference speed |
| **Mechanism** | Mask invalid tokens at each step | Draft model generates tokens, large model verifies in batch |
| **Changes output?** | Yes — prevents invalid tokens | No — same output distribution as large model alone |
| **Requires extra model?** | No | Yes (small draft model) |

They are orthogonal — you can use both at the same time: speculative decoding for speed,
grammar guided decoding for output validity.

### Not Beam Search

Beam search keeps top-K candidate sequences and picks the best one at the end. It explores
multiple paths but doesn't enforce structural validity. Grammar guided decoding enforces
validity at every single step.

---

## Speculative Decoding (Separate Concept)

Since the two are often confused, here's speculative decoding briefly:

### The Problem It Solves

LLM inference is memory-bandwidth bound, not compute bound. Reading 70B model weights
from GPU HBM for every single token is slow. The GPU is underutilized.

### How It Works

```
1. Small draft model (e.g., 1B) generates K tokens quickly (low memory bandwidth cost)
2. Large target model (e.g., 70B) verifies all K tokens in ONE forward pass (parallel)
3. Accept tokens up to the first mismatch; reject the rest
4. Generate 1 correction token from the large model
5. Repeat
```

Key insight: the large model can process K tokens in parallel in one forward pass almost
as fast as processing 1 token. So if the draft model is right most of the time (high
acceptance rate), you get K tokens for roughly the cost of 1 large model step.

### Acceptance Rate

If draft model agrees with large model 80% of the time per token:
- Expected tokens per large model call ≈ 1/(1-0.8) = 5 tokens
- Effective throughput: ~5× speedup vs naive large model decoding

The output distribution is mathematically identical to large model alone (it's a rejection
sampling scheme — no quality loss).

### Where It's Used

- vLLM supports speculative decoding natively
- Effective when: draft model is fast + high acceptance rate (same domain, similar training)
- Less effective when: queries are very diverse (low acceptance rate)

---

## Summary

| Technique | Problem | Mechanism | Output Change |
|---|---|---|---|
| Grammar Guided Decoding | Invalid/hallucinated structure | Mask illegal tokens at each step | Valid structure guaranteed |
| Speculative Decoding | Inference too slow | Draft + verify in parallel | None (same distribution) |
| Beam Search | Greedy decoding suboptimal | Keep top-K sequences | Better sequence score |
