#pragma once

// ============================================================================
// binarycore/math/similarity.hpp
// ----------------------------------------------------------------------------
// Normalized similarity in [0, 1] between two binary vectors.
//
// Definition:
//   similarity(a, b) = 1 - hamming(a, b) / DIMS
//
// This is the binary equivalent of cosine similarity. It returns a float
// because it's a ratio.
//
//   1.0  → a and b are identical
//   0.5  → half the bits agree (worst case for random binary vectors)
//   0.0  → a and b are exact complements
//
// IMPORTANT — this is the only function in the math layer that returns
// floating-point. Inference paths (training loops, retrieval, ranking) should
// use the integer functions (hamming, dot) directly. This function exists
// for:
//   - human-readable scoring / debugging output
//   - comparing against legacy float-cosine baselines
//   - APIs that require a "score in [0,1]"
//
// If you call this in a hot loop, profile carefully — the integer-to-float
// conversion can become the bottleneck.
// ============================================================================

#include "binarycore/containers/binary_vec.hpp"
#include "binarycore/math/hamming.hpp"

#include <cstddef>

namespace binarycore {

// ----------------------------------------------------------------------------
// similarity(a, b)
// ----------------------------------------------------------------------------
// Note: NOT constexpr because float division at compile time has surprising
// quirks across compilers; we keep it runtime-only for predictability.
// ----------------------------------------------------------------------------
template <std::size_t DIMS>
float similarity(const BinaryVec<DIMS>& a, const BinaryVec<DIMS>& b) noexcept {
  // static_cast<float> makes the division floating-point. Without it,
  // std::size_t / std::size_t would be integer division (truncating).
  return 1.0f - static_cast<float>(hamming(a, b)) / static_cast<float>(DIMS);
}

}  // namespace binarycore
