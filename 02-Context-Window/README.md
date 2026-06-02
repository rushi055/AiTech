# Context Window

## What it is

The context window is the maximum number of tokens a transformer can process in a single forward pass. Input tokens + output tokens — everything combined must fit inside this limit.

It is the model's **entire working memory**. There is no external storage, no "remembering" from previous requests. If information is not inside the context window, it does not exist to the model.

```
┌────────────────────────── 128k token window ──────────────────────────┐
│ System prompt │ Conversation history │ Your message │ Model response │
│ (~500 tokens) │ (~50,000 tokens)     │ (~200 tokens)│ (~4,000 tokens)│
└───────────────────────────────────────────────────────────────────────┘
         Total: ~54,700 tokens — fits within 128k limit
```

If total exceeds the window, one of three things happens:
- Oldest messages get truncated (most chat UIs do this silently)
- API returns an error
- Request is rejected

---

## Why the limit exists — the architectural constraint

Self-attention computes a relationship score between **every pair of tokens**. For a sequence of length `n`:

```
Attention matrix shape: n × n
Compute cost: O(n²)
Memory cost: O(n²)
```

This means:

```
4k tokens   → 4,096 × 4,096   = 16 million attention entries per head per layer
128k tokens → 128,000 × 128,000 = 16 billion attention entries per head per layer
```

That's 1000x more memory and compute from a 32x increase in context length. At some point, you physically run out of GPU memory. The context window is where that hard ceiling sits.

---

## KV Cache — why long context is expensive

During generation, the model produces one token at a time (autoregressive). Without optimization, generating token #500 would require reprocessing all 499 previous tokens from scratch.

The **KV cache** solves this by storing the Key and Value projections from every previous position:

```
Generating token at position t:
  Query:  computed fresh (only for position t)
  Keys:   retrieved from cache (positions 0 to t-1) + computed for t
  Values: retrieved from cache (positions 0 to t-1) + computed for t
```

The Query is cheap — it's just one vector. But you need all previous Keys and Values to compute attention against the full history.

### Memory calculation for KV cache

Formula:
```
KV cache memory = 2 × num_layers × context_length × num_kv_heads × head_dim × bytes_per_value
```

Where each term comes from:

| Term | Value (GPT-4 scale estimate) | Why |
|------|-----|-----|
| `2` | 2 | You store both Keys AND Values. Queries are not cached — only computed for the current token. |
| `num_layers` | 96 | Each transformer layer has independent attention with its own K, V projections. Layer 12's keys are different from Layer 13's keys. Must cache each layer separately. |
| `context_length` | 128,000 | One KV entry per token position that's been processed. Full context = 128k entries per layer. |
| `num_kv_heads` | 96 (MHA) or 8 (GQA) | Multi-head attention runs 96 parallel attention computations. Each head has its own K and V. (With GQA, multiple query heads share fewer KV heads — see below.) |
| `head_dim` | 128 | Model dimension split across heads. If model_dim=12,288 and 96 heads: 12,288 / 96 = 128 floats per K or V vector per head per position. |
| `bytes_per_value` | 2 | float16 storage. Each scalar = 16 bits = 2 bytes. |

### Worked example

Assumption: 96 layers, 128k context, 96 KV heads, 128 head_dim, float16:

```
2 × 96 × 128,000 × 96 × 128 × 2 bytes

Step by step:
  Single KV vector (one head, one position): 128 × 2 bytes = 256 bytes
  K and V together:                          256 × 2 = 512 bytes
  All 96 heads at one position, one layer:   512 × 96 = 49,152 bytes ≈ 48 KB
  All 96 layers at one position:             49,152 × 96 = 4,718,592 bytes ≈ 4.5 MB
  All 128,000 positions:                     4,718,592 × 128,000 ≈ 604 GB
```

~600 GB of KV cache **per request** at full context. This is why:
- Long-context inference requires many GPUs
- Providers charge more for long context
- Most deployments don't actually fill the full window

### GQA reduces this dramatically

Grouped Query Attention (used in LLaMA 3, likely GPT-4):

```
Standard MHA: 96 query heads, 96 key heads, 96 value heads
GQA (8 KV heads): 96 query heads, 8 key heads, 8 value heads

KV cache shrinks by 12× (96 → 8 KV heads)
~600 GB → ~50 GB  (manageable across a few GPUs)
```

Multiple query heads share the same K and V. Empirically, quality barely drops while memory savings are massive.

---

## Positional encoding — how the model knows token order

Self-attention is permutation-invariant. `["cat", "sat", "the"]` and `["the", "cat", "sat"]` would produce identical attention scores without position information.

Positional encoding injects order:

**Learned absolute positions (GPT-2, BERT era):**
```
Position 0 → learned_vector_0
Position 1 → learned_vector_1
...
Position 1023 → learned_vector_1023
```
Hard cap. Cannot generalize beyond training length. This is why GPT-2 was stuck at 1024 tokens.

**RoPE — Rotary Position Embedding (modern models):**
- Encodes *relative* distance between tokens using rotation matrices
- Position 5 attending to position 3 has the same geometric relationship as position 100 attending to position 98
- Can extrapolate beyond training length (with quality degradation)
- Enabled the jump from 4k → 128k contexts via fine-tuning with extended RoPE frequencies

RoPE is what makes context extension possible without retraining from scratch.

---

## Attention quality degrades over distance

Even within the context window, the model doesn't attend uniformly. Empirically observed patterns:

```
┌─────────────────────────────────────────────────────────────┐
│ Attention strength across a 100k-token context:             │
│                                                             │
│ ████                                              ████████  │
│ ████                                              ████████  │
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████  │
│ Position 0        ...middle...              Position 100k   │
│                                                             │
│ ▲ Primacy bias     ▲ "Lost in the middle"   ▲ Recency bias  │
└─────────────────────────────────────────────────────────────┘
```

**Primacy bias:** First few tokens get strong attention regardless of content. System prompts benefit from this.

**Recency bias:** Most recent tokens get strong attention. The model "remembers" what it just read.

**Lost in the middle:** Information placed in the middle of a long context is retrieved less reliably. Demonstrated in Liu et al., 2023 — a fact placed at position 1k out of 100k is significantly harder to recall than one at position 0 or position 99k.

Engineering implication: if you're stuffing context with retrieved documents, put the most important ones at the **beginning** or **end**, not the middle.

---

## What happens inside transformer layers

Every layer does two operations, in sequence:

```
Input → Self-Attention → Feed-Forward Network → Output
         (tokens talk     (each token processed
          to each other)   independently)
```

### Self-Attention: information routing

Each token produces:
- **Query**: "What am I looking for?"
- **Key**: "What do I contain?"
- **Value**: "What can I contribute?"

```
Sentence: "The cat sat on the mat because it was tired"

Processing token "it":
  Query("it"): looking for its referent
  Key("cat"): high match — subject noun, earlier in sentence
  Key("mat"): medium match — also a noun
  Key("sat"): low match — verb, less relevant for coreference

  Result: "it" representation updated with information from "cat"
          The model learned coreference resolution.
```

With 96 heads running in parallel, different heads specialize:
- Head A: tracks subject-verb agreement
- Head B: tracks pronoun → noun reference
- Head C: tracks negation scope
- Head D: tracks numerical relationships

These specializations emerge from training, not from manual design.

### Feed-Forward Network: knowledge storage and transformation

```python
FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂
# W₁: model_dim → 4× model_dim (expand)
# W₂: 4× model_dim → model_dim (compress back)
```

Research finding:
- **Attention** = routing. Decides what information flows where.
- **FFN** = memory. Stores factual knowledge and applies transformations.

FFN neurons encode specific concepts:
```
Neuron 3847 in layer 22: fires for "capital cities"
Neuron 9102 in layer 45: fires when context implies "past tense needed"
Neuron 1533 in layer 71: fires for "Python syntax patterns"
```

FFN contains ~2/3 of all model parameters. It's the heavy storage layer.

### Residual connections: layers add, not replace

```
output = x + Attention(LayerNorm(x))       # layer adds to input
output = output + FFN(LayerNorm(output))   # layer adds to input
```

Each layer contributes a *delta* to the representation. If a layer has nothing useful to add, it can output near-zero and the input passes through unchanged. This is why deep networks (96 layers) train stably — no vanishing gradient, no information destruction.

---

## What different layers learn — progressive abstraction

```
Layers 1–10:    Surface-level features
                Token identity, POS tags, adjacent relationships
                "This is a noun." "This follows a comma."

Layers 10–35:   Syntactic structure
                Phrase boundaries, clause nesting
                Subject-verb-object parsing
                "it" refers to "cat" (coreference)

Layers 35–65:   Semantic understanding
                Meaning composition, sentiment
                Factual knowledge retrieval
                "Paris is the capital of France"

Layers 65–90:   Reasoning and planning
                Multi-step inference chains
                Output structure planning
                "To answer this I need to first resolve X, then Y"

Layers 90–96:   Output preparation
                Convert abstract representation → next-token distribution
                Final confidence calibration
```

This is empirically validated. Probing layer 3 reveals POS information but not sentiment. Probing layer 40 reveals sentiment. Probing layer 80+ reveals the final answer.

---

## Why depth matters — layers as computational steps

Each layer is one sequential step of computation. Complex tasks require multiple steps:

```
Query: "Is it true that the capital of the country where the inventor
        of the telephone was born is Ottawa?"

Step 1 (early layers):   Parse nested noun phrase structure
Step 2 (layers ~20-35):  Resolve "inventor of telephone" → Alexander Graham Bell
Step 3 (layers ~35-50):  Resolve "country where he was born" → UK/Canada
Step 4 (layers ~50-70):  Resolve "capital of Canada" → Ottawa
Step 5 (layers ~70-90):  Compare: Ottawa == Ottawa → True
```

A 10-layer model physically cannot perform 5 sequential reasoning steps. It runs out of depth. This is not a matter of training longer — it's a hard architectural limitation.

### Why not just add more layers indefinitely?

| More layers gives you | More layers costs you |
|---|---|
| More reasoning steps | Linear increase in latency (layers are sequential, cannot parallelize) |
| More specialization per layer | Training instability in very deep networks |
| Better performance on hard tasks | Diminishing returns: 48→96 helps a lot, 96→192 helps marginally |

```
Latency math:
96 layers × ~0.5ms/layer = ~48ms per token
192 layers = ~96ms per token
→ 25 tok/s vs 50 tok/s — users notice this
```

The alternative: instead of going deeper (more sequential steps), go **wider** using Mixture-of-Experts:

```
Dense model:  1 FFN per layer, always used
MoE model:    8 FFN experts per layer, each token routed to only 2

Result: 4× total parameters, same compute per token, same latency
More knowledge storage without more sequential cost.
```

---

## Context window sizes — current landscape

| Model | Context Window | ~Pages of text |
|-------|---------------|----------------|
| GPT-2 | 1,024 | ~1.5 |
| GPT-3 | 4,096 | ~6 |
| GPT-3.5 Turbo | 16,384 | ~24 |
| GPT-4 | 8,192 / 32,768 | ~12 / ~48 |
| GPT-4o | 128,000 | ~190 |
| Claude 3.5 Sonnet | 200,000 | ~300 |
| Gemini 1.5 Pro | 1,000,000 | ~1,500 |
| LLaMA 3 | 8,192 / 128,000 | ~12 / ~190 |

(~1 page ≈ 500 words ≈ 670 tokens)

---

## Techniques for extending context

### RoPE scaling (NTK-aware, YaRN)

Modify rotary position encoding frequencies to handle positions beyond training. The model trained on 4k context learns to interpolate to 128k with additional fine-tuning on long sequences.

### Sliding window attention (Mistral)

Each token only attends to the nearest `w` tokens instead of all previous:
```
Full attention: O(n²) — every token sees everything
Sliding window (w=4096): O(n × w) — each token sees last 4096 only
```
Information from distant tokens propagates through intermediate layers (like message passing). Cheaper, but indirect.

### Grouped Query Attention (GQA)

Fewer KV heads than query heads. Reduces KV cache size without reducing model's ability to ask diverse queries.

### Ring attention / sequence parallelism

Distribute the sequence across GPUs. Each GPU owns a chunk of positions and they communicate KV data in a ring. Used for Gemini's 1M+ context.

---

## Cost implications

```
GPT-4o pricing (approximate):
  Input:  $2.50 / 1M tokens
  Output: $10.00 / 1M tokens

Sending 100k tokens of context per request:
  100,000 × $2.50/1M = $0.25 just for input
  1,000 requests/day = $250/day in context costs alone
```

This is why:
- You don't send your entire codebase as context for every question
- RAG exists: retrieve only relevant chunks, keep context small
- Summarization strategies compress old conversation
- System prompts should be concise — they're prepended to every request

---

## When you need RAG vs when context is enough

```
Knowledge base: 10M tokens
Context window: 128k tokens

Option A: Stuff everything → impossible, won't fit
Option B: Stuff first 128k → misses 98.7% of knowledge
Option C: Retrieve relevant chunks → only send what matters
```

Context window size determines your RAG architecture:
- **Chunk size**: small context → smaller chunks needed
- **Number of chunks**: how many retrieved passages fit alongside the question
- **Whether you need RAG at all**: if your data fits in context, skip retrieval entirely

---

## The mental model

Context window = **RAM, not disk.**

- Volatile: nothing persists between requests
- Fixed size: hard limit, cannot exceed
- Everything active: all tokens are "visible" simultaneously via attention
- No addressing: you can't "seek" to a position — the model attends to everything at once

What's outside the window is gone. The model has no concept of "earlier in the conversation" beyond what tokens remain in the current window. There is no hidden memory, no background storage. The window is all there is.

---

## Quick reference

| Question | Answer |
|----------|--------|
| What's the context window? | Max tokens (input + output) the model can handle in one pass |
| Why is there a limit? | Self-attention is O(n²) in memory and compute |
| What's the KV cache? | Stored Key/Value vectors from previous positions, avoids recomputation |
| Why does long context cost more? | KV cache scales linearly with context length × model size |
| Does the model use the whole window equally? | No — primacy and recency bias, "lost in the middle" effect |
| Why so many layers? | Each layer = one reasoning step; hard tasks need many sequential steps |
| More layers = always better? | Diminishing returns + latency cost; width (MoE) scales better |
| What do layers do? | Attention routes information between tokens; FFN stores/transforms knowledge |

---

[← Back to Index](../README.md)
