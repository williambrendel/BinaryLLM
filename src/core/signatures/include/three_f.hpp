#pragma once

// ============================================================================
// core/signatures/three_f.hpp
//
// 3F per-word context signature:
//
//   before[i]  — F-dim OR over current[0..i-1]   (prefix-OR, exclusive)
//   current[i] — F-dim bag of word at position i (Option B encoding,
//                materialized as a BinaryVecDynamic)
//   after[i]   — F-dim OR over current[i+1..N-1] (suffix-OR, exclusive)
//
// Built in O(N) with three passes:
//   1. encode each word's current sig from its part-ID bag.
//   2. forward sweep accumulating prefix-OR → fills `before`.
//   3. backward sweep accumulating suffix-OR → fills `after`.
//
// Scope is whatever the caller passes — a sentence range, a paragraph
// range, or the whole token stream. The 3F is built strictly against
// Word tokens inside the range (delimiters skipped). The caller is
// responsible for upstream segmentation; pair this with
// split_sentences / split_paragraphs.
//
// Output size equals the number of Word tokens in [start, end). One
// ThreeFSignature per word.
// ============================================================================

#include "binary_vec.hpp"
#include "dictionary.hpp"
#include "tokenize.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace core::signatures {

// Single runtime-dim signature vector type used throughout. Aliased
// here so callers don't need to think about which BinaryVec tier the
// dim falls into.
using SigVec = binarycore::binary_vec::BinaryVecDynamic;

struct ThreeFSignature {
  SigVec before;
  SigVec current;
  SigVec after;
};

// Convert a sorted-unique bag of part IDs (from encode_word) into a
// SigVec of the given dim. The bag is assumed sorted ascending; the
// resulting SigVec chunks are correspondingly sorted-unique.
SigVec bag_to_sig(const std::vector<std::uint32_t>& bag, std::size_t dim);

// OR `src` into `dst` in place. Both must have the same dim.
// O(|dst.chunks[k]| + |src.chunks[k]|) per chunk via sorted-merge with dedup.
void union_into(SigVec& dst, const SigVec& src);

// Build 3F signatures for Word tokens in the half-open range
// [start, end) of the token stream. Output has one entry per Word.
std::vector<ThreeFSignature> build_three_f(
    const core::parts::PartDictionary& dict,
    std::span<const core::parts::StreamToken> tokens,
    std::size_t start, std::size_t end);

// Convenience: full token span.
std::vector<ThreeFSignature> build_three_f(
    const core::parts::PartDictionary& dict,
    std::span<const core::parts::StreamToken> tokens);

}  // namespace core::signatures
