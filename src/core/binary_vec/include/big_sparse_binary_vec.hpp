#pragma once

// ============================================================================
// binarycore::binary_vec::big_sparse_binary_vec.hpp
//
// Tier 3 BinaryVec (compile-time variant): sparse representation
// for dim > 65535 known at compile time. Storage =
// std::array<SparseBinaryVec<65535>, num_chunks>, where
// num_chunks = ⌈Dim/65535⌉. Each chunk stores uint16_t indices LOCAL
// to its 65535-bit window.
//
// Only jaccard + intersection_size are exposed. Computed by
// accumulating per-chunk |a ∩ b| via SparseBinaryVec's
// intersection_size helper, then applying |a ∪ b| = |a| + |b| - |a ∩ b|
// once at the end.
//
// For runtime-known dim see big_sparse_binary_vec_dynamic.hpp.
// ============================================================================

#include "sparse_binary_vec.hpp"

#include <array>
#include <cstddef>

namespace binarycore::binary_vec {

template <std::size_t Dim>
struct BigSparseBinaryVec {
  static_assert(Dim > 65535,
                "BigSparseBinaryVec: Dim must be > 65535 "
                "(use SparseBinaryVec for smaller)");

  static constexpr std::size_t chunk_size = 65535;
  static constexpr std::size_t num_chunks = (Dim + chunk_size - 1) / chunk_size;

  std::array<SparseBinaryVec<chunk_size>, num_chunks> chunks{};
};

template <std::size_t Dim>
std::size_t intersection_size(const BigSparseBinaryVec<Dim>& a,
                              const BigSparseBinaryVec<Dim>& b) {
  std::size_t count = 0;
  for (std::size_t k = 0; k < BigSparseBinaryVec<Dim>::num_chunks; ++k) {
    count += intersection_size(a.chunks[k], b.chunks[k]);
  }
  return count;
}

template <std::size_t Dim>
float jaccard(const BigSparseBinaryVec<Dim>& a,
              const BigSparseBinaryVec<Dim>& b) {
  std::size_t inter = 0;
  std::size_t total_a = 0;
  std::size_t total_b = 0;
  for (std::size_t k = 0; k < BigSparseBinaryVec<Dim>::num_chunks; ++k) {
    inter   += intersection_size(a.chunks[k], b.chunks[k]);
    total_a += a.chunks[k].data.size();
    total_b += b.chunks[k].data.size();
  }
  const std::size_t uni = total_a + total_b - inter;
  if (uni == 0) return 0.0f;
  return static_cast<float>(inter) / static_cast<float>(uni);
}

}  // namespace binarycore::binary_vec
