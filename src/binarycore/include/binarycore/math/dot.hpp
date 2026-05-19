#pragma once

// ============================================================================
// binarycore/math/dot.hpp
// ----------------------------------------------------------------------------
// Binary dot product between two binary vectors.
//
// Definition:
//   dot(a, b) = number of bit positions where BOTH a and b have a 1
//             = popcount(a AND b)
//
// AND (&) produces a 1 in every position where both operands have a 1.
// Counting these 1-bits gives the number of "shared on bits", which is the
// binary analog of the standard inner product on the same bit positions.
//
// Range: [0, DIMS]
//   0    = no shared 1-bits (orthogonal under the binary inner product)
//   DIMS = a and b are identical and entirely ones
//
// Key identities:
//   dot(a, a) == popcount(a)        (a vector's "dot with itself" = its norm²)
//   dot(a, 0) == 0                   (the zero vector contributes nothing)
//   dot(a, b) == dot(b, a)           (symmetric)
//
// Note: this is NOT the same as cosine similarity. It's an unnormalized
// integer measure. To get a similarity in [0,1], use similarity() from
// similarity.hpp (which is Hamming-based).
// ============================================================================

#include "binarycore/containers/binary_vec.hpp"
#include "binarycore/math/popcount.hpp"

#include <cstddef>
#include <cstdint>

namespace binarycore {

// ----------------------------------------------------------------------------
// dot(a, b) — generic
// ----------------------------------------------------------------------------
// Sum of popcount(a.data[i] AND b.data[i]) across all chunks.
// ----------------------------------------------------------------------------
template <std::size_t DIMS>
constexpr std::size_t dot(const BinaryVec<DIMS>& a,
                          const BinaryVec<DIMS>& b) noexcept {
  std::size_t total = 0;
  for (std::size_t i = 0; i < BinaryVec<DIMS>::Chunks; ++i) {
    total += popcount(a.data[i] & b.data[i]);
  }
  return total;
}

// ============================================================================
// Explicit fast paths for canonical sizes
// ----------------------------------------------------------------------------
// Same structure as hamming, but with AND instead of XOR.
// ============================================================================

template <>
constexpr std::size_t dot<64>(const BinaryVec<64>& a,
                              const BinaryVec<64>& b) noexcept {
  return popcount(a.data[0] & b.data[0]);
}

template <>
constexpr std::size_t dot<128>(const BinaryVec<128>& a,
                               const BinaryVec<128>& b) noexcept {
  return popcount(a.data[0] & b.data[0]) +
         popcount(a.data[1] & b.data[1]);
}

template <>
constexpr std::size_t dot<256>(const BinaryVec<256>& a,
                               const BinaryVec<256>& b) noexcept {
  return popcount(a.data[0] & b.data[0]) +
         popcount(a.data[1] & b.data[1]) +
         popcount(a.data[2] & b.data[2]) +
         popcount(a.data[3] & b.data[3]);
}

template <>
constexpr std::size_t dot<512>(const BinaryVec<512>& a,
                               const BinaryVec<512>& b) noexcept {
  return popcount(a.data[0] & b.data[0]) +
         popcount(a.data[1] & b.data[1]) +
         popcount(a.data[2] & b.data[2]) +
         popcount(a.data[3] & b.data[3]) +
         popcount(a.data[4] & b.data[4]) +
         popcount(a.data[5] & b.data[5]) +
         popcount(a.data[6] & b.data[6]) +
         popcount(a.data[7] & b.data[7]);
}

}  // namespace binarycore
