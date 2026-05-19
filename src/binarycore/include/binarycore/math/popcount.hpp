#pragma once

// ============================================================================
// binarycore/math/popcount.hpp
// ----------------------------------------------------------------------------
// Population count: the number of 1-bits in a value.
//
// This is the fundamental primitive for all our binary similarity math.
// Modern CPUs implement it as a dedicated instruction (POPCNT on x86,
// VCNT on ARM NEON), which is what makes Hamming and dot product so cheap.
//
// std::popcount (C++20) is a thin wrapper over the hardware instruction
// where available, falling back to a software implementation otherwise.
// We use it directly rather than writing our own, both for portability and
// because the compiler vendor's implementation is well-optimized.
//
// This file provides:
//   popcount(uint64_t)         — single chunk
//   popcount(BinaryVec<DIMS>)  — sum of chunk popcounts (generic + 4 specs)
// ============================================================================

#include "binarycore/containers/binary_vec.hpp"

#include <bit>       // std::popcount (C++20)
#include <cstddef>
#include <cstdint>

namespace binarycore {

// ----------------------------------------------------------------------------
// popcount(uint64_t)
// ----------------------------------------------------------------------------
// Counts the set bits in a single 64-bit word.
// Returns a std::size_t (the platform's largest unsigned integer type) so
// the caller doesn't have to worry about overflow when summing across many
// words.
//
// std::popcount returns int; we cast to std::size_t to keep the result
// consistent with our DIMS type, which is also std::size_t.
// ----------------------------------------------------------------------------
constexpr std::size_t popcount(uint64_t x) noexcept {
  return static_cast<std::size_t>(std::popcount(x));
}

// ----------------------------------------------------------------------------
// popcount(BinaryVec<DIMS>) — generic
// ----------------------------------------------------------------------------
// Total set bits across all chunks of a BinaryVec.
//
// The loop bound (Chunks) is a compile-time constant, so the compiler can
// fully unroll this at -O2. For the four canonical sizes (64, 128, 256, 512)
// we provide explicit specializations below that don't even use a loop.
// ----------------------------------------------------------------------------
template <std::size_t DIMS>
constexpr std::size_t popcount(const BinaryVec<DIMS>& v) noexcept {
  std::size_t total = 0;
  for (std::size_t i = 0; i < BinaryVec<DIMS>::Chunks; ++i) {
    total += popcount(v.data[i]);
  }
  return total;
}

// ============================================================================
// Explicit fast paths for canonical sizes
// ----------------------------------------------------------------------------
// Each specialization is straight-line code — no loop, no counter, no branch.
// The compiler sees N independent popcount calls and can schedule them in
// parallel to maximize instruction-level parallelism.
// ============================================================================

template <>
constexpr std::size_t popcount<64>(const BinaryVec<64>& v) noexcept {
  // One chunk → one POPCNT instruction on x86, one VCNT on ARM.
  return popcount(v.data[0]);
}

template <>
constexpr std::size_t popcount<128>(const BinaryVec<128>& v) noexcept {
  // Two independent popcounts, summed.
  return popcount(v.data[0]) + popcount(v.data[1]);
}

template <>
constexpr std::size_t popcount<256>(const BinaryVec<256>& v) noexcept {
  return popcount(v.data[0]) + popcount(v.data[1]) +
         popcount(v.data[2]) + popcount(v.data[3]);
}

template <>
constexpr std::size_t popcount<512>(const BinaryVec<512>& v) noexcept {
  // On AVX-512 + VPOPCNTQ, this entire operation can compile to a single
  // SIMD instruction across one 512-bit ZMM register. The compiler decides.
  return popcount(v.data[0]) + popcount(v.data[1]) + popcount(v.data[2]) +
         popcount(v.data[3]) + popcount(v.data[4]) + popcount(v.data[5]) +
         popcount(v.data[6]) + popcount(v.data[7]);
}

}  // namespace binarycore
