#pragma once

// ============================================================================
// binarycore::binary_vec::dense_binary_vec.hpp
//
// Tier 1 BinaryVec: dense bitset for dim in (0, 4096].
// Storage = std::array<uint64_t, ⌈Dim/64⌉>, value-initialized to 0.
// Each uint64 is a 64-bit "chunk" of the underlying bit space.
//
// Only mutators are set_bit / get_bit. No bit_intersection /
// bit_union materializing new vectors — Jaccard is computed in one
// fused loop: AND popcount + OR popcount per chunk.
//
// The main loop is 4-way unrolled with AND/OR INTERLEAVED at each
// offset. Loading a.chunks[k] and b.chunks[k] into named locals lets
// both popcounts reuse them before the offset advances — no register
// spill, no L1 re-load. Eight independent accumulators (4 for inter,
// 4 for uni) preserve ILP across offsets.
//
// A scalar remainder loop handles any chunks past the 4-way boundary.
// ============================================================================

#include <array>
#include <bit>          // std::popcount (C++20)
#include <cstddef>
#include <cstdint>

namespace binarycore::binary_vec {

template <std::size_t Dim>
struct DenseBinaryVec {
  static_assert(Dim > 0,
                "DenseBinaryVec: Dim must be > 0");
  static_assert(Dim <= 4096,
                "DenseBinaryVec: Dim must be <= 4096 "
                "(use SparseBinaryVec or BigSparseBinaryVec for larger)");

  static constexpr std::size_t num_chunks = (Dim + 63) / 64;
  std::array<std::uint64_t, num_chunks> chunks{};
};

template <std::size_t Dim>
inline void set_bit(DenseBinaryVec<Dim>& v, std::size_t i) {
  v.chunks[i >> 6] |= (std::uint64_t{1} << (i & 63));
}

template <std::size_t Dim>
inline bool get_bit(const DenseBinaryVec<Dim>& v, std::size_t i) {
  return (v.chunks[i >> 6] >> (i & 63)) & std::uint64_t{1};
}

template <std::size_t Dim>
float jaccard(const DenseBinaryVec<Dim>& a,
              const DenseBinaryVec<Dim>& b) {
  constexpr std::size_t N  = DenseBinaryVec<Dim>::num_chunks;
  constexpr std::size_t N4 = (N / 4) * 4;

  std::size_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
  std::size_t u0 = 0, u1 = 0, u2 = 0, u3 = 0;

  for (std::size_t k = 0; k < N4; k += 4) {
    // Load each offset's pair into locals so the compiler reuses
    // them across AND and OR without spilling or re-fetching.
    const std::uint64_t a0 = a.chunks[k    ], b0 = b.chunks[k    ];
    const std::uint64_t a1 = a.chunks[k + 1], b1 = b.chunks[k + 1];
    const std::uint64_t a2 = a.chunks[k + 2], b2 = b.chunks[k + 2];
    const std::uint64_t a3 = a.chunks[k + 3], b3 = b.chunks[k + 3];

    // Interleaved AND/OR per offset.
    i0 += static_cast<std::size_t>(std::popcount(a0 & b0));
    u0 += static_cast<std::size_t>(std::popcount(a0 | b0));
    i1 += static_cast<std::size_t>(std::popcount(a1 & b1));
    u1 += static_cast<std::size_t>(std::popcount(a1 | b1));
    i2 += static_cast<std::size_t>(std::popcount(a2 & b2));
    u2 += static_cast<std::size_t>(std::popcount(a2 | b2));
    i3 += static_cast<std::size_t>(std::popcount(a3 & b3));
    u3 += static_cast<std::size_t>(std::popcount(a3 | b3));
  }

  std::size_t inter = i0 + i1 + i2 + i3;
  std::size_t uni   = u0 + u1 + u2 + u3;

  for (std::size_t k = N4; k < N; ++k) {
    const std::uint64_t ax = a.chunks[k];
    const std::uint64_t bx = b.chunks[k];
    inter += static_cast<std::size_t>(std::popcount(ax & bx));
    uni   += static_cast<std::size_t>(std::popcount(ax | bx));
  }

  if (uni == 0) return 0.0f;
  return static_cast<float>(inter) / static_cast<float>(uni);
}

}  // namespace binarycore::binary_vec
