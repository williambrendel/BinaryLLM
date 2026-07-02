#pragma once

// ============================================================================
// core/parts/decomposer.hpp
//
// Decomposition: given a PartDictionary and a word (or delimiter),
// produce the sequence of part IDs that represent it.
//
// One algorithm, two callers:
//
//   decompose_word          — public; used at inference time
//   find_singleton_runs     — internal helper for the extractor's peel
//                             loop; identifies positions left uncovered
//                             so they can become candidates in the
//                             next iteration
//
// Word decomposition algorithm:
//
//   1. Whole-word fast path: if word is a Whole atom, emit one ID and
//      return.
//
//   2. Peel in dict ID order: iterate parts in the dictionary's order
//      (= insertion order). For each non-Letter/Whole/Delimiter part:
//        Start — peel from front of word if not already start-claimed,
//                value equals word's prefix, and that range is free.
//                At most one Start claim per word.
//        End   — peel from back, mirror of Start. At most one End claim.
//        Mid   — peel every non-overlapping occurrence in the
//                remaining uncovered region. A Mid can fire multiple
//                times in one word.
//
//   3. Letter singleton fill: any character not covered by the peel
//      becomes a positional Letter atom using the WordPiece-style "##"
//      marker (x##, ##x##, ##x).
//
// To get a different priority order over the same parts, REORDER THE
// DICT — see dict_reorder.hpp. The decomposer itself is strategy-
// agnostic.
//
// L range matches the extractor — see kMinPartLength / kMaxPartLength
// in part.hpp.
//
// Delimiter decomposition: if the value is exactly in Kind::Delimiter,
// emit it. If not (rare — happens only at inference for unseen
// delimiter strings), fall back to character-by-character emission.
// ============================================================================

#include "dictionary.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace core::parts {

// Decompose a single word into part IDs.
// `word` is assumed to be lowercased ASCII.
// Returns IDs in stream order (left-to-right in the word).
std::vector<std::uint32_t> decompose_word(
    const PartDictionary& dict,
    std::string_view word);

// Decompose a delimiter token into part IDs.
std::vector<std::uint32_t> decompose_delimiter(
    const PartDictionary& dict,
    std::string_view delim);

// Run information from a decomposition: contiguous range of word
// positions that were claimed by Letter singletons (i.e., no Start /
// End / Mid / Whole part covered them).
//
// run_start_position is inclusive; run_end_position is exclusive.
// touches_word_start / touches_word_end indicate whether the run
// reaches the word's boundaries (used by the peel pass to classify
// run kind).
struct SingletonRun {
  std::size_t start;
  std::size_t end;
  bool touches_word_start;
  bool touches_word_end;
};

// Same peel logic as decompose_word, but returns unclaimed ranges
// instead of emitted part IDs. Consistent semantics — same dict,
// same word, complementary outputs.
std::vector<SingletonRun> find_singleton_runs(
    const PartDictionary& dict,
    std::string_view word);

}  // namespace core::parts
