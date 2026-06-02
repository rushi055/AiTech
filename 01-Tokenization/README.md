# Chapter 1: Tokenization — How LLMs Read Text

---

## The Problem: Computers Don't Understand Words

Let's start with a simple question — when you type "Hello, how are you?" into ChatGPT, what does the model actually *see*?

The answer might surprise you: **it doesn't see words at all.**

Neural networks are mathematical machines. They work with numbers — vectors, matrices, dot products. So before any "thinking" happens, your text must be converted into numbers. This conversion process is called **tokenization**.

But here's the interesting part — we don't just assign one number per word. We don't assign one number per character either. We do something smarter.

---

## Why Not Just Use Words?

Your first instinct might be: "Why not just give every word a number?"

```
"the"  → 1
"cat"  → 2
"sat"  → 3
"on"   → 4
...
```

This seems reasonable until you think about it:

1. **How many words exist in English?** Hundreds of thousands. Add scientific terms, names, slang, other languages — millions. Your vocabulary table becomes impossibly large.

2. **What about new words?** Every day, new words appear — "ChatGPT", "defederate", "rizz". A word-level tokenizer would have no way to handle them.

3. **What about typos?** "helllo" isn't in any dictionary, but a model should still process it.

4. **What about other languages?** Chinese has tens of thousands of characters. Arabic morphology creates enormous word variations.

So word-level tokenization is out.

---

## Why Not Just Use Characters?

Okay, what about going the other direction — one token per character?

```
"Hello" → ['H', 'e', 'l', 'l', 'o'] → [72, 101, 108, 108, 111]
```

This solves the vocabulary problem — you only need ~256 entries (one per byte). But now you have a different problem:

**Sequences become extremely long.**

The sentence "The quick brown fox jumps over the lazy dog" is 44 characters. A typical paragraph might be 500+ characters. An article? 10,000+ characters.

Why does length matter? Because the Transformer architecture (the engine inside every modern LLM) uses something called **self-attention**, and its computational cost grows with the **square** of the sequence length. Double the sequence length → 4x the compute.

So character-level tokenization is too expensive.

---

## The Sweet Spot: Subword Tokenization

The solution is **subword tokenization** — a middle ground between words and characters.

The idea:
- Common words like "the", "is", "hello" become single tokens
- Rare words get broken into meaningful pieces: "unhappiness" → "un" + "happi" + "ness"
- Even completely unknown words can be represented by falling back to individual characters or bytes

This gives us:
- ✅ Small, fixed vocabulary (50k–200k tokens)
- ✅ Can represent ANY text (no "unknown word" problem)
- ✅ Reasonable sequence lengths
- ✅ Meaningful pieces ("un" carries meaning, "ness" carries meaning)

Now let's look at how these subword vocabularies are actually built.

---

## How BPE Works (Byte-Pair Encoding)

BPE is the algorithm used by **GPT-2, GPT-3, GPT-4, and Claude**. It's elegant in its simplicity.

### The Training Process (Building the Vocabulary)

Imagine you have a large corpus of text. Here's a tiny example:

```
Training text: "low low low low lower lower newest newest widest"
```

**Step 1:** Split everything into individual characters. Add a special end-of-word marker (let's use `_`):

```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, _}

Word frequencies:
  l o w _       → appears 4 times
  l o w e r _   → appears 2 times
  n e w e s t _ → appears 2 times
  w i d e s t _ → appears 1 time
```

**Step 2:** Count every pair of adjacent symbols across the entire corpus:

```
(l, o) → appears 6 times (4 from "low" + 2 from "lower")
(o, w) → appears 6 times
(w, _) → appears 4 times
(w, e) → appears 3 times (2 from "newer" + 1 from "widest"... wait)
...
```

Find the MOST frequent pair. Let's say it's `(l, o)` with 6 occurrences.

**Step 3:** Merge that pair into a new symbol:

```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, _, lo}  ← new token!

Now our words look like:
  lo w _         → 4 times
  lo w e r _     → 2 times
  n e w e s t _  → 2 times
  w i d e s t _  → 1 time
```

**Step 4:** Repeat. Find the next most frequent pair.

Now `(lo, w)` appears 6 times. Merge it:

```
Vocabulary: {l, o, w, e, r, n, s, t, i, d, _, lo, low}

  low _         → 4 times
  low e r _     → 2 times
  n e w e s t _ → 2 times
  w i d e s t _ → 1 time
```

**Step 5:** Continue merging. Maybe next is `(e, s)` → `es`, then `(es, t)` → `est`, then `(low, _)` → `low_`, and so on.

After thousands of iterations, you end up with a vocabulary like:

```
Final vocabulary (simplified):
{l, o, w, e, r, n, s, t, i, d, _, lo, low, low_, er, er_,
 new, est, est_, newest_, lower_, widest_, ...}
```

**The key insight:** BPE naturally learns that:
- Very common words become single tokens ("the", "is", "and")
- Common prefixes/suffixes become tokens ("un", "ing", "tion", "ness")
- Rare words remain split into these common pieces

### Using BPE at Inference Time (Encoding Your Text)

When you type something into ChatGPT, the tokenizer applies the learned merges **in the exact order they were learned during training**:

```
Input: "lowest"

Start:  [l] [o] [w] [e] [s] [t]
Merge 1 (l+o→lo):    [lo] [w] [e] [s] [t]
Merge 2 (lo+w→low):  [low] [e] [s] [t]
Merge 3 (e+s→es):    [low] [es] [t]
Merge 4 (es+t→est):  [low] [est]

Final tokens: ["low", "est"]
Token IDs: [4521, 478]
```

Notice something beautiful here — the model has never seen "lowest" as a whole word in its vocabulary, but it still represents it meaningfully as "low" + "est" (a base word + a superlative suffix).

---

## How Unigram/SentencePiece Works (Used by Gemini, LLaMA)

Unigram takes the **opposite approach** to BPE.

### BPE = Bottom-Up (start small, merge up)
### Unigram = Top-Down (start big, prune down)

Here's how:

**Step 1:** Start with a MASSIVE vocabulary — every possible substring that appears in your corpus, up to some length. This might be 1,000,000+ candidates.

```
Initial vocabulary:
["a", "b", ..., "th", "he", "the", "tok", "oken", "token",
 "ize", "ization", "tokenization", ...]
```

**Step 2:** Assign a probability to each token using corpus statistics. Frequent substrings get higher probability.

```
P("the") = 0.031
P("th") = 0.012
P("e") = 0.025
P("token") = 0.0004
P("tokenization") = 0.00001
```

**Step 3:** For the entire corpus, compute the total loss (negative log-likelihood). This measures: "How well can we represent all text using this vocabulary?"

**Step 4:** Ask: "If I remove each token, how much would the loss increase?" Tokens that barely affect the loss are redundant — the text can be represented just as well without them.

**Step 5:** Remove the bottom 20-30% of tokens (least useful ones).

**Step 6:** Repeat steps 2-5 until you reach your target vocabulary size.

### Encoding with Unigram (The Viterbi Algorithm)

Here's where Unigram gets interesting. Given a word, there are MANY ways to segment it:

```
"tokenization" could be:
  ["tokenization"]                    → 1 token  (if in vocab)
  ["token", "ization"]               → 2 tokens
  ["token", "iza", "tion"]           → 3 tokens
  ["to", "ken", "i", "za", "tion"]  → 5 tokens
  ["t","o","k","e","n","i"...]     → 12 tokens
```

Unigram picks the segmentation with the **highest total probability**. It calculates:

```
Score(segmentation) = P(token₁) × P(token₂) × ... × P(tokenₙ)

Score(["token", "ization"]) = P("token") × P("ization") = 0.0004 × 0.0008 = 3.2e-7
Score(["to", "ken", "ization"]) = P("to") × P("ken") × P("ization") = much smaller
```

Winner: `["token", "ization"]` ✓

This is solved efficiently using the **Viterbi algorithm** (dynamic programming — same algorithm used in speech recognition). It runs in O(n²) time, not exponential.

---

## The Complete Tokenization Pipeline

When you send a message to GPT-4 or Gemini, here's exactly what happens:

### Stage 1: Normalization

Clean up the text — handle Unicode edge cases.

```
"café"  → normalize to consistent Unicode form (NFC)
"ﬁ" (ligature) → might become "fi" (two characters)
```

Most modern LLMs do minimal normalization to preserve the original text.

### Stage 2: Pre-tokenization

Split text into **chunks** that the subword algorithm processes independently. This prevents weird merges across word boundaries.

GPT-4 uses this regex to pre-split:

```python
# Simplified explanation of what the regex does:
"Hello, world! I have 1234 cats." 
→ ["Hello", ",", " world", "!", " I", " have", " 1234", " cats", "."]
```

Notice:
- Spaces are attached to the NEXT word (" world", not "world ")
- Punctuation is separated
- Numbers are kept together (up to 3 digits per group)

BPE then runs **within each chunk** — so "Hello" and "," can never merge together.

### Stage 3: Subword Tokenization

Apply BPE merges (or Unigram Viterbi) to each chunk:

```
" world" → [" world"]  (common enough to be one token)
" 1234"  → [" 123", "4"]  (numbers split at 3 digits)
"Hello"  → ["Hello"]  (common word = one token)
```

### Stage 4: Convert to IDs

Look up each token in the vocabulary table:

```
["Hello"] → [9906]
[","]     → [11]
[" world"] → [1917]
["!"]     → [0]
...
```

### Stage 5: Add Special Tokens

The model needs markers to understand structure:

```
<|begin_of_text|> You are a helpful assistant <|end_of_turn|>
<|start_of_turn|> user
Hello, world!
<|end_of_turn|>
<|start_of_turn|> assistant
```

These special tokens are added by the **chat template**, not by you.

---

## Real-World Examples That Build Intuition

### Example 1: Why LLMs are bad at counting letters

```
"How many 'r's are in strawberry?"

Tokenized: ["How", " many", " '", "r", "'", "s", " are", " in", " straw", "berry", "?"]
```

The model sees "straw" and "berry" as separate tokens. It never sees the raw characters inside — it has to *reason* about what characters make up each token. This is like asking you to count letters in a word while looking at it through frosted glass.

### Example 2: Why LLMs struggle with arithmetic

```
"What is 3847 + 1234?"

Tokenized: ["What", " is", " 384", "7", " +", " 123", "4", "?"]
```

The model sees `384` and `7` as **completely separate tokens**. To add 3847 + 1234, it needs to:
1. Understand that tokens "384" and "7" together mean 3847
2. Understand that tokens "123" and "4" together mean 1234
3. Do the addition

This is like asking someone to add two numbers where each number is written on separate sticky notes that got shuffled.

### Example 3: Why the same text costs differently in different languages

```
English: "Hello, how are you?"     → 6 tokens
Japanese: "こんにちは、お元気ですか？"  → 11 tokens
Hindi: "नमस्ते, आप कैसे हैं?"       → 15 tokens
```

English dominates the training data, so English words are efficiently compressed into fewer tokens. Other languages, especially those with different scripts, get fragmented more heavily. This means:
- Less content fits in the context window
- API calls cost more
- Generation is slower

### Example 4: Emojis are expensive

```
"I love pizza 🍕"  →  ["I", " love", " pizza", " ", "🍕"]
                       But "🍕" internally = 2-3 token IDs!

"🎉🎊🎈" = 6 tokens! (each emoji takes ~2 tokens)
```

Emojis are represented as multi-byte UTF-8 sequences, and since they're rare in training data, they rarely get merged into single tokens.

---

## Vocabulary Sizes Across Models

| Model | Vocab Size | What this means |
|-------|-----------|------------------|
| GPT-2 | 50,257 | Smaller vocab → more tokens per text |
| GPT-3.5 | 100,256 | Good balance for English |
| GPT-4 / 4o | 200,019 | Larger → better multilingual efficiency |
| Gemini | ~256,000 | Largest → fewest tokens per text |
| LLaMA 2 | 32,000 | Small vocab → sequences are longer |
| LLaMA 3 | 128,256 | Big upgrade from LLaMA 2 |

**The tradeoff:**
- Bigger vocabulary = fewer tokens per sentence = more content fits in context window
- But bigger vocabulary = larger embedding matrix = more GPU memory

GPT-4's embedding table alone: 200,019 tokens × 4,096 dimensions × 2 bytes (float16) = **~1.6 GB** just for the token embeddings!

---

## How Tokens Become "Understanding"

Once we have token IDs, here's how they flow through the model:

```
"The cat sat" → [464, 3797, 3290]
                      ↓
         ┌────────────────────────┐
         │   EMBEDDING LOOKUP     │
         │   464  → [0.23, -0.11, 0.87, ...]  (4096-dim vector)  │
         │   3797 → [0.45, 0.33, -0.21, ...]  (4096-dim vector)  │
         │   3290 → [-0.12, 0.56, 0.09, ...]  (4096-dim vector)  │
         └────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │   + POSITIONAL INFO    │
         │   (so model knows ORDER │
         │    of tokens matters)   │
         └────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │   TRANSFORMER LAYERS   │
         │   (96 layers in GPT-4) │
         │   Self-attention +      │
         │   Feed-forward networks │
         └────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │   OUTPUT: Probability  │
         │   over ALL 200k tokens │
         │                        │
         │   "on"  → 0.15         │
         │   "down" → 0.08        │
         │   "there" → 0.03       │
         │   ...                   │
         └────────────────────────┘
                      ↓
         Pick next token ("on"), append, repeat.
```

This is **autoregressive generation** — predict one token at a time, feed it back in, predict the next.

---

## Try It Yourself

```python
# Install: pip install tiktoken
import tiktoken

# Load GPT-4's tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

text = "Tokenization is the bridge between human language and neural networks."
tokens = enc.encode(text)

print(f"Text: {text}")
print(f"Number of tokens: {len(tokens)}")
print(f"\nToken breakdown:")
for token_id in tokens:
    token_text = enc.decode([token_id])
    print(f"  ID {token_id:>6} → '{token_text}'")

# Fun experiment: compare languages
for text in ["Hello world", "こんにちは世界", "Привет мир", "مرحبا بالعالم"]:
    n = len(enc.encode(text))
    print(f"{text:<20} → {n} tokens")
```

---

## Summary — The Mental Model

Think of tokenization like this:

> Imagine you're writing a telegram where you pay per word. You'd naturally develop abbreviations and shorthands for common phrases. "See you later" becomes "CUL8R". Rare phrases stay spelled out.
>
> BPE does exactly this — but automatically, optimally, and at massive scale. It finds the best "abbreviations" (merges) for a given text corpus to minimize total message length.

Key facts to remember:

1. **Tokens ≠ words.** A token might be a whole word, part of a word, a single character, or even punctuation.
2. **~1 token ≈ 4 characters ≈ 0.75 words** in English (rough rule of thumb).
3. **Tokenization is fixed** — once the tokenizer is trained, it never changes. The same text always produces the same tokens.
4. **Context windows are measured in tokens** — "128k context" means 128,000 tokens, not words or characters.
5. **Token boundaries affect model behavior** — this explains many LLM quirks (bad at spelling, arithmetic, character counting).

---

[← Back to Index](../README.md)
