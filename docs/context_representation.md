# Context representation — diagnostic findings & pivot

**Status:** design finding, 2026-07. Measured on Alice (27k tokens) and Pride
& Prejudice (128k), using the shipped pieces (3F signatures, WGT1 surprisal,
`info_jaccard`). Throwaway probes under the session scratchpad; numbers below.

## The question

Pass-1 reconstructs the C (identity) band poorly (BER_C recall ≈ 0.56 even after
C-emphasis). Chasing it through the encoder — containment vs. Jaccard match,
argmax vs. graded firing, negative codewords, an exact-C index — never closed
it. So we measured the thing underneath: **how much does the layer-0 context
(the OR-pooled `L`,`R` bands) actually determine the token identity?**

## Findings

1. **Exact whole-sentence context → identity is a near-bijection, but it's
   memorization.** A token's exact `(L,R)` bag maps to ~1.09 identities
   (H ≈ 0.05 bits) — but **96% of contexts are seen exactly once**. So "context
   determines identity" holds only on data already seen.

2. **It's not data starvation — it's the pooling scope.** Going 5× larger
   (Alice → Pride) made singletons *worse* (95.6% → 97.9%). `L`,`R` are
   OR-pooled over the **whole sentence**, so each context is a one-off sentence
   *fingerprint* that essentially never recurs, at any corpus size.

3. **Fuzzy whole-context match is weak and spiky.** Nearest-context vote
   resolves only ~0.15 bits of the 8.6-bit identity (top-1 6.7% ≈ unigram
   floor). Loosening the match to a Hamming ball of just 8 bits already mixes in
   ~37 identities — no identity-preserving neighborhood.

4. **A *local* window generalizes and hits the n-gram floor.** Context = union
   of the ±1 neighbor words:

   | window | reuse% (Pride) | H(identity \| context) |
   |---|---|---|
   | ±1 | 61.5% (rises with data) | ~2–2.8 bits (per reused token) |
   | ±2 | 3.7% | memorization |
   | whole sentence | ~2% | memorization |

   `±1` cuts identity entropy 8.6 → ~2–2.8 bits — **the trigram entropy of
   English** — and reuse *improves* with data. `±2+` collapses back to unique
   fingerprints.

## Conclusion

The layer-0 context representation is the root limitation, not the encoder,
metric, or firing rule — all of which were downstream of a context that could
not generalize. **The whole-sentence OR-pool destroys the local recurrence that
carries prediction.**

## Pivot: context-parts (discover, don't memorize)

Whole contexts don't recur, but **parts of them do** (a coarse projection or a
local window recurs). The fix is the same trick used for tokens, one level up:

| Level | Raw units | Discovered parts | Generalizes because |
|---|---|---|---|
| Tokens (done) | 1M+ words | ~30k subword parts | parts recur across words |
| Contexts (next) | 27k+ one-off contexts | K **context-parts** | parts recur across contexts |

- A context should be **decomposed** into recurring context-parts, not matched
  as a whole (matching whole contexts is the "1M-word vocabulary" mistake at the
  context level).
- The **codeword *is* a context-part** — a recurring local pattern / phrase
  template (`"the ___ said"`), discovered by **recurrence** (extractor-style
  peel over the token stream), not centroid clustering, and kept **local**
  (whole-sentence pooling was the bug).
- This is exactly the spec's self-similar stacking (§6, "codewords become the
  new pieces"): chars→subwords→context-parts→…, with the discovery method made
  concrete and the pooling scope fixed. The pass-2 follower table is then the
  template → slot-filler (identity) distribution, whose residual is the entropy
  floor (§9).

## Implications for the build

- Rework the 3F encoder so `L`/`R` pool over a **local window**, or (better)
  represent context by its **discovered context-part set**.
- Reframe pass-1: codewords are context-parts found by recurrence over the
  token stream, not ODL clusters of whole-sentence signatures.
- The C identity stays an exact channel (near-unique decode key); context-parts
  carry the distributional prediction.
