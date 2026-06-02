# Tokenization

## Why tokenization exists

Transformers operate on sequences of discrete integer IDs, not raw text. Tokenization is the mapping layer that converts a byte stream into a sequence of vocabulary indices the model can process.

The design constraint is straightforward: self-attention has O(n²) complexity in sequence length. You want the shortest possible sequence that still preserves semantic information. This rules out character-level encoding (too long) and word-level encoding (unbounded vocabulary, can't handle unseen words).

The practical solution is **subword tokenization** — a learned compression scheme that maps frequent byte sequences to single tokens while decomposing rare sequences into smaller known pieces.

---

## The two dominant algorithms

### BPE (Byte-Pair Encoding)

Used by: GPT-2, GPT-3, GPT-4, Claude

**Training procedure:**

1. Initialize vocabulary with 256 byte-level tokens
2. Scan corpus for the most frequent adjacent token pair
3. Merge that pair into a new token, add to vocabulary
4. Repeat until target vocabulary size is reached

Worked example on a toy corpus:

```
Corpus: "low low low lower lower newest"

Start:     l, o, w, _, e, r, n, s, t  (individual bytes)
Merge 1:   (l, o) → lo               freq=6 across corpus
Merge 2:   (lo, w) → low              freq=6
Merge 3:   (e, s) → es                freq=3
Merge 4:   (es, t) → est              freq=3
Merge 5:   (low, e) → lowe            freq=2
Merge 6:   (lowe, r) → lower          freq=2
...
```

After 50k–100k merges, you get a vocabulary where:
- High-frequency words are single tokens: `the`, `is`, `function`
- Morphemes become tokens: `un`, `ing`, `tion`, `ness`
- Rare words decompose into known subwords: `defenestration` → `def` + `en` + `est` + `ration`

**Encoding at inference time:**

Apply merges in priority order (same order they were learned):

```
Input: "lowest"

[l][o][w][e][s][t]
→ [lo][w][e][s][t]       merge #1
→ [low][e][s][t]         merge #2
→ [low][es][t]           merge #3
→ [low][est]             merge #4

Result: ["low", "est"] → IDs [4521, 478]
```

This is deterministic. Same input always produces same output.

### Unigram (SentencePiece)

Used by: Gemini, LLaMA 2, T5

Opposite direction from BPE — starts large, prunes down.

**Training procedure:**

1. Initialize with a massive candidate vocabulary (~1M substrings from corpus)
2. Assign log-probability to each token based on corpus frequency
3. For each token, compute: how much would corpus likelihood decrease if this token were removed?
4. Remove the 20–30% of tokens with smallest impact
5. Repeat until target vocabulary size

**Encoding at inference time:**

Given input text, find the segmentation that maximizes total log-probability:

```
Input: "tokenization"

Candidate segmentations:
  ["tokenization"]           → log P = -11.2  (single token, if in vocab)
  ["token", "ization"]      → log P = -7.8 + -7.1 = -14.9
  ["to", "ken", "ization"]  → log P = -4.2 + -9.3 + -7.1 = -20.6
  ["t","o","k","e"...]     → log P = -58.4

Winner: highest probability segmentation
```

Solved via Viterbi algorithm (dynamic programming, O(n²) in token length). Not brute force.

**Key difference from BPE:** Unigram encoding is probabilistic. In theory, multiple segmentations could be sampled during training for regularization. BPE is always deterministic.

---

## Pre-tokenization: what happens before subword splitting

Both GPT and Gemini apply a **pre-tokenization** step that splits raw text into chunks. Subword algorithms then run independently within each chunk.

GPT-4 uses this regex pattern:

```python
pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

What this does in practice:

```
"Hello, world! I have 12345 cats."
→ ["Hello", ",", " world", "!", " I", " have", " 123", "45", " cats", "."]
```

Design decisions visible here:
- Leading spaces attach to the following word (" world", not "world")
- Numbers split into groups of at most 3 digits
- Punctuation isolated into separate chunks
- BPE merges cannot cross chunk boundaries

This prevents pathological merges like `". T"` becoming a single token.

---

## Full pipeline: text to model input

```
Raw bytes
  → Unicode normalization (NFC)
  → Pre-tokenization (regex split into chunks)
  → Subword encoding (BPE merges or Unigram Viterbi per chunk)
  → Token ID lookup (string → integer via vocab table)
  → Special token insertion (<bos>, <eos>, role markers for chat)
  → Embedding lookup (ID → dense vector, e.g. 4096-dim)
  → Positional encoding (RoPE or learned)
  → Into transformer layers
```

---

## Vocabulary size trade-offs

| Model | Algorithm | Vocab Size | Notes |
|-------|-----------|-----------|-------|
| GPT-2 | BPE | 50,257 | Original baseline |
| GPT-3.5 | BPE (cl100k_base) | 100,256 | 2x vocab, better multilingual |
| GPT-4o | BPE (o200k_base) | 200,019 | Further expansion |
| Gemini | SentencePiece Unigram | ~256,000 | Largest production vocab |
| LLaMA 2 | SentencePiece BPE | 32,000 | Small, English-centric |
| LLaMA 3 | BPE | 128,256 | 4x LLaMA 2, much better multilingual |

**Engineering trade-off:**

- Larger vocab → shorter sequences → more content in context window → less compute per token
- Larger vocab → bigger embedding matrix → more memory

Back-of-envelope: GPT-4's embedding table at float16:
`200,019 × 4,096 × 2 bytes ≈ 1.6 GB`

That's just the lookup table, before any transformer weights.

---

## Practical consequences of tokenization

### Arithmetic is hard because numbers split at token boundaries

```
"3847" → ["384", "7"]     two separate tokens
"1234" → ["123", "4"]     two separate tokens
```

The model never sees `3847` as an atomic unit. It must reconstruct numerical value from separate token representations. This is a known failure mode and the reason many systems use tool-calling for math.

### Character-level tasks fail for the same reason

```
"strawberry" → ["straw", "berry"]
```

Asking "how many r's in strawberry" requires the model to mentally decompose tokens it treats as atomic units back into characters. The representation doesn't naturally support this.

### Non-English text is less efficient

```
"Hello world"          → 2 tokens
"こんにちは世界"         → 4-5 tokens  (same semantic content)
"नमस्ते दुनिया"          → 8-10 tokens (same semantic content)
```

This happens because English dominates training corpora, so English subwords get aggressively merged. Under-represented scripts remain fragmented. Consequences:
- Effective context window is smaller for non-English
- Per-token API pricing creates cost asymmetry across languages
- Inference is slower (more forward passes for same content)

### Emojis consume disproportionate tokens

```
"🎉" → 2-3 token IDs (multi-byte UTF-8, rarely merged)
"🎉🎊🎈" → 6+ tokens
```

---

## Experimenting with tokenization

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

def inspect(text: str) -> None:
    tokens = enc.encode(text)
    print(f"{text!r} → {len(tokens)} tokens")
    for tid in tokens:
        print(f"  {tid:>6} │ {enc.decode([tid])!r}")

inspect("Tokenization is the first stage of any LLM pipeline.")
inspect("3847 + 1234")
inspect("strawberry")
inspect("こんにちは世界")
```

Output gives direct visibility into how the model will "see" your input.

---

## Summary

| Property | Detail |
|----------|--------|
| What determines splits | Statistical frequency in training corpus + algorithm (BPE/Unigram) |
| Approximate ratio | ~1 token ≈ 4 characters ≈ 0.75 English words |
| Fixed after training | Yes — same text always yields same tokens |
| Context limits | Measured in tokens, not words or characters |
| Behavioral impact | Explains failures in spelling, counting, arithmetic, multilingual parity |

Tokenization is invisible to end users but dictates model capabilities and limitations at a fundamental level. Many "reasoning failures" attributed to transformers are actually tokenization artifacts.

---

[← Back to Index](../README.md)
