#pragma once

// ============================================================================
// core/signatures/encoder.hpp
//
// Encode the Word tokens of a chunk into 3F signatures.
//
// A 3F signature is a single BinaryVecDynamic of dimension 3*F laid
// out as three F-wide bands:
//
//   [0,   F)  before   — OR over current bags of words [0, i)
//   [F,  2F)  current  — Option B bag of word i
//   [2F, 3F)  after    — OR over current bags of words (i, N)
//
// Built in O(N) per chunk:
//   1. encode each word's current bag (offset +F).
//   2. forward sweep: prefix-OR of currents → write into before band.
//   3. backward sweep: suffix-OR of currents → write into after band.
//
// Scope is the half-open token range [start, end). Defaults cover the
// whole stream. The 3F is built strictly against Word tokens in the
// range (delimiters skipped). Pair with split_sentences /
// split_paragraphs for sentence/paragraph scopes.
//
// Output: one BinaryVecDynamic (dim = 3*F) per Word token in the range.
// ============================================================================

#include "binary_vec.hpp"
#include "dictionary.hpp"
#include "tokenize.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace core::signatures {

// Encode Word tokens in [start, end) into 3F signatures. start/end
// default to the full span; end is clamped to tokens.size().
std::vector<binarycore::binary_vec::BinaryVecDynamic> encode(
    const core::parts::PartDictionary& dict,
    std::span<const core::parts::StreamToken> tokens,
    std::size_t start = 0,
    std::size_t end = std::numeric_limits<std::size_t>::max());

}  // namespace core::signatures
