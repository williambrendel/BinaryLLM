#pragma once

// ============================================================================
// binarycore/math/hamming.hpp
// ----------------------------------------------------------------------------
// Hamming distance between two binary vectors.
//
// Definition:
//   hamming(a, b) = number of bit positions where a and b differ
//                 = popcount(a XOR b)
//
// XOR (^) produces a 1 in every position where the two operands differ and
// a 0 where they match. Counting the 1-bits then gives the distance.
//
// Range: [0, DIMS]
//   0    = identical
//   DIMS = completely opposite (every bit flipped)
//
// Hamming is symmetric: hamming(a, b) == hamming(b, a).
// Hamming is integer: no floating point, no rounding error.
// ============================================================================

#include "binarycore/containers/binary_vec.hpp"
#include "binarycore/math/popcount.hpp"

#include <cstddef>
#include <cstdint>

namespace binarycore {

// ----------------------------------------------------------------------------
// hamming(a, b) — generic
// ----------------------------------------------------------------------------
// Sum of popcount(a.data[i] XOR b.data[i]) across all chunks.
//
// The XOR happens directly on the uint64_t chunks (not on a temporary
// BinaryVec) to avoid allocating an intermediate.
// ----------------------------------------------------------------------------
template <std::size_t DIMS>
constexpr std::size_t hamming(const BinaryVec<DIMS>& a,
                              const BinaryVec<DIMS>& b) noexcept {
  std::size_t total = 0;
  for (std::size_t i = 0; i < BinaryVec<DIMS>::Chunks; ++i) {
    total += popcount(a.data[i] ^ b.data[i]);
  }
  return total;
}

// ============================================================================
// Explicit fast paths for canonical sizes
// ----------------------------------------------------------------------------
// Straight-line code: each chunk XORed and popcounted independently, then
// summed. No loops, no branches, no temporary BinaryVec.
// ============================================================================

template <>
constexpr std::size_t hamming<64>(const BinaryVec<64>& a,
                                  const BinaryVec<64>& b) noexcept {
  // One XOR, one POPCNT. Two instructions total on modern x86.
  return popcount(a.data[0] ^ b.data[0]);
}

template <>
constexpr std::size_t hamming<128>(const BinaryVec<128>& a,
                                   const BinaryVec<128>& b) noexcept {
  return popcount(a.data[0] ^ b.data[0]) +
         popcount(a.data[1] ^ b.data[1]);
}

template <>
constexpr std::size_t hamming<256>(const BinaryVec<256>& a,
                                   const BinaryVec<256>& b) noexcept {
  return popcount(a.data[0] ^ b.data[0]) +
         popcount(a.data[1] ^ b.data[1]) +
         popcount(a.data[2] ^ b.data[2]) +
         popcount(a.data[3] ^ b.data[3]);
}

template <>
constexpr std::size_t hamming<512>(const BinaryVec<512>& a,
                                   const BinaryVec<512>& b) noexcept {
  return popcount(a.data[0] ^ b.data[0]) +
         popcount(a.data[1] ^ b.data[1]) +
         popcount(a.data[2] ^ b.data[2]) +
         popcount(a.data[3] ^ b.data[3]) +
         popcount(a.data[4] ^ b.data[4]) +
         popcount(a.data[5] ^ b.data[5]) +
         popcount(a.data[6] ^ b.data[6]) +
         popcount(a.data[7] ^ b.data[7]);
}

}  // namespace binarycore
