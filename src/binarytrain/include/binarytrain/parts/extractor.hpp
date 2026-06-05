#pragma once

// ============================================================================
// binarytrain/parts/extractor.hpp
//
// PartExtractor: builds a PartDictionary from observed words and
// delimiters via an iterative peel algorithm.
//
// Workflow:
//
//   PartExtractor ex;
//   for (each StreamToken tok in corpus):
//     if (tok.type == Word)      ex.add_word(tok.value);
//     else                       ex.add_delimiter(tok.value);
//   PartDictionary dict = ex.finalize();
//
// Algorithm in finalize():
//
//   1. Seed the dictionary:
//      - Whole atoms: every distinct word with length 2..whole_max_len,
//        with frequency-driven adaptive prune to keep only the head
//        of the frequency distribution (Zipfian corpora produce a
//        small canonical-English set; uniform lexicons let the length
//        cap dominate).
//      - 78 alphabetic + 30 digit Letter singletons (positional)
//      - 26 letter + 10 digit single-character Whole atoms (coverage
//        backstop, added unconditionally)
//      - Every observed delimiter, frequency-sorted (no prune)
//
//   2. Peel loop. At the start of the first iteration every training
//      word that isn't already a Whole atom decomposes to nothing but
//      Letter singletons — a single "run" spanning the whole word and
//      touching both word boundaries. So the very first iteration is
//      exactly the classical frequency-based extraction over the
//      training vocabulary. Subsequent iterations refine the residual.
//
//      Repeat:
//        a. Re-decompose every training word with the current dict.
//        b. Build a pool of every singleton run of length >= 2 along
//           with its (touches_word_start, touches_word_end) flags.
//        c. If the pool is empty, stop.
//        d. For each L in {7, 6, 5, 4, 3, 2}:
//           - For each pool entry (substring s of length n):
//               * Every L-substring of s contributes to mid_bin.
//               * If touches_word_start, the L-prefix s[0..L)
//                 also contributes to start_bin.
//               * If touches_word_end, the L-suffix s[n-L..n)
//                 also contributes to end_bin.
//           - adaptive_prune_count each bin (elbow + scaled_mm cascade).
//           - Promote winners to the dictionary as Start / Mid / End.
//        e. If the iteration added zero new parts, stop (no progress).
//
// The iteration terminates when no length-≥2 singleton runs exist
// (every remaining unclaimed position is a 1-character Letter
// singleton — the genuine backstop) or when no further promotions are
// possible.
// ============================================================================

#include "binarytrain/parts/dictionary.hpp"
#include "binarytrain/parts/part.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace binarytrain::parts {

struct ExtractorConfig {
  // Length cap on whole-atom candidates. Defensive safety bound — the
  // adaptive_prune_count cascade applied to the frequency vector is
  // the operative selector. The cap only dominates when the input has
  // no frequency signal (uniform-frequency lexicon, e.g. a raw word
  // list). 16 covers essentially every legitimate English word.
  std::size_t whole_max_len = 16;

  // If false, skip adding Letter singletons. Default true — they're
  // the coverage backstop ensuring any input character is representable.
  bool add_letter_singletons = true;

  // Safety cap on the number of peel iterations. The algorithm
  // terminates naturally when the singleton-run pool is empty or no
  // new parts are added in an iteration; this is a worst-case fuse.
  std::size_t max_peel_iterations = 20;
};

class PartExtractor {
public:
  explicit PartExtractor(ExtractorConfig cfg = {}) : cfg_(cfg) {}

  // Feed a word (assumed already lowercased ASCII).
  void add_word(std::string_view word);

  // Feed a delimiter token (kept literally).
  void add_delimiter(std::string_view delim);

  // Build the final dictionary. Single-use.
  PartDictionary finalize();

  // Diagnostic accessors.
  std::size_t observed_word_count() const noexcept { return word_count_; }
  std::size_t distinct_word_count() const noexcept {
    return all_word_freq_.size();
  }

  // Number of peel iterations the most recent finalize() ran. Useful
  // for diagnostics on convergence behavior. 0 before finalize() runs.
  std::size_t last_peel_iterations() const noexcept {
    return last_peel_iters_;
  }

private:
  ExtractorConfig cfg_;

  // All distinct training words (regardless of length) with their
  // observed counts. Used at finalize for whole-atom selection (by
  // length filter + cascade prune) and for the peel loop.
  std::unordered_map<std::string, std::uint64_t> all_word_freq_;

  // Per-value frequency counts for delimiters.
  std::unordered_map<std::string, std::uint64_t> delim_freq_;

  std::size_t word_count_ = 0;
  std::size_t last_peel_iters_ = 0;
};

}  // namespace binarytrain::parts
