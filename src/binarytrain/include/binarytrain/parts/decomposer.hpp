#pragma once

// ============================================================================
// binarytrain/parts/decomposer.hpp
//
// Decomposition: given a PartDictionary and a word (or delimiter),
// produce the sequence of part IDs that represent it.
//
// Word decomposition algorithm:
//
//   1. Whole-word fast path: if the word is in Kind::Whole, emit its
//      one ID and return.
//
//   2. Start phase: try the word's longest prefix at length 4, then 3,
//      then 2, against Kind::Start. On hit, emit the part and consume
//      those characters from the front.
//
//   3. End phase: same logic on suffixes against Kind::End from the
//      back of what remains.
//
//   4. Mid phase: scan the middle portion left-to-right, trying lengths
//      5, then 4, then 3, then 2, against Kind::Mid. On hit, emit and
//      advance past the match. On miss, advance by 1 (the character
//      will become a Letter singleton). L=2 mids are looked up here to
//      support the coverage-peel pass in the extractor that promotes
//      L=2 mid candidates.
//
//   5. Singleton fill: any character not consumed by phases 2–4 emits
//      a Letter singleton with a position-tagged value using the
//      WordPiece-style "##" continuation marker:
//        - position 0 (and no Start emitted)        → "x##"
//        - last position (and no End emitted)       → "##x"
//        - everything else                          → "##x##"
//      The "##" separator is collision-free for our character set
//      (letters/digits/connectors none contain '#'), so every
//      (character, position) pair maps to a unique string. Position
//      tags ensure single-character coverage of any word.
//
// Output order: stream order in the original word. So a typical word
// produces [start (optional), mids and letters in left-to-right order,
// end (optional)].
//
// Delimiter decomposition: if the value is exactly in Kind::Delimiter,
// emit it. If not (rare — happens only at inference for unseen
// delimiter strings), fall back to character-by-character emission.
// ============================================================================

#include "binarytrain/parts/dictionary.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace binarytrain::parts {

// Decompose a single word into part IDs.
// `word` is assumed to be lowercased ASCII. Returns IDs in stream order.
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

// Find singleton runs in `word`'s decomposition. Same algorithm as
// decompose_word but returns the unclaimed ranges instead of the
// emitted part IDs.
//
// For peel iterations: gives the regions of each word where the
// current dictionary lacks coverage and where pruned-tail candidates
// could be promoted to fill in.
std::vector<SingletonRun> find_singleton_runs(
    const PartDictionary& dict,
    std::string_view word);

}  // namespace binarytrain::parts
