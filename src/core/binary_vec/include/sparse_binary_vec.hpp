#pragma once

// ============================================================================
// binarycore::binary_vec::sparse_binary_vec.hpp
//
// Tier 2 BinaryVec: sparse representation for dim in (4096, 65535].
// Storage = sorted-strictly-ascending std::vector<uint16_t> of set
// bit indices. The user is responsible for maintaining the invariant
// (sorted, unique, all entries in [0, Dim)).
//
// API surface intentionally minimal:
//   - intersection_size(a, b)  — count of |a ∩ b| via two-pointer
//                                merge. No allocation. Reused both
//                                here and by BigSparseBinaryVec /
//                                BigSparseBinaryVecDynamic.
//   - jaccard(a, b)            — |a ∩ b| / (|a| + |b| - |a ∩ b|).
//                                Returns 0.0f for empty/empty (no NaN).
//
// No bit_intersection / bit_union materializing new vectors — Jaccard
// is the only ratio we need, and computing it via intersection_size
// avoids the intermediate allocation.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace binarycore::binary_vec {

template <std::size_t Dim>
struct SparseBinaryVec {
  static_assert(Dim > 0,
                "SparseBinaryVec: Dim must be > 0");
  static_assert(Dim <= 65535,
                "SparseBinaryVec: Dim must be <= 65535 "
                "(use BigSparseBinaryVec for larger)");

  std::vector<std::uint16_t> data;
};

// |a ∩ b| via two-pointer merge over sorted inputs. No allocation.
template <std::size_t Dim>
std::size_t intersection_size(const SparseBinaryVec<Dim>& a,
                              const SparseBinaryVec<Dim>& b) {
  std::size_t count = 0;
  std::size_t i = 0, j = 0;
  while (i < a.data.size() && j < b.data.size()) {
    if      (a.data[i] < b.data[j]) ++i;
    else if (a.data[i] > b.data[j]) ++j;
    else { ++count; ++i; ++j; }
  }
  return count;
}

// Jaccard = |a ∩ b| / |a ∪ b|. |a ∪ b| derived as |a|+|b|-|a∩b|.
// Empty/empty returns 0.0f by convention (no 0/0 NaN).
template <std::size_t Dim>
float jaccard(const SparseBinaryVec<Dim>& a,
              const SparseBinaryVec<Dim>& b) {
  const std::size_t inter = intersection_size(a, b);
  const std::size_t uni = a.data.size() + b.data.size() - inter;
  if (uni == 0) return 0.0f;
  return static_cast<float>(inter) / static_cast<float>(uni);
}

}  // namespace binarycore::binary_vec
