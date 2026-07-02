#pragma once

// ============================================================================
// binarycore::binary_vec::big_sparse_binary_vec_dynamic.hpp
//
// Runtime-dim counterpart to BigSparseBinaryVec<Dim>. Same chunk
// structure (each chunk is a SparseBinaryVec<65535>), but the chunk
// count is determined at construction from a runtime dim value.
//
// Use this whenever the bit-space dimension is only known at runtime
// — typically the input-signature dim computed once the dictionary
// is loaded. For dim ≤ 65535 the vector has exactly one chunk, so
// this type also serves as the universal runtime variant across the
// full size range (no need for SparseBinaryVecDynamic or
// DenseBinaryVecDynamic today).
//
// Precondition: any two-vector op assumes a.dim == b.dim (and
// therefore a.chunks.size() == b.chunks.size()). Not enforced at
// runtime — caller's contract, same posture as the templated variants.
// ============================================================================

#include "sparse_binary_vec.hpp"

#include <cstddef>
#include <vector>

namespace binarycore::binary_vec {

struct BigSparseBinaryVecDynamic {
  static constexpr std::size_t chunk_size = 65535;

  std::size_t dim = 0;
  std::vector<SparseBinaryVec<chunk_size>> chunks;

  BigSparseBinaryVecDynamic() = default;

  explicit BigSparseBinaryVecDynamic(std::size_t dim_)
      : dim(dim_),
        chunks((dim_ + chunk_size - 1) / chunk_size) {}
};

inline std::size_t intersection_size(const BigSparseBinaryVecDynamic& a,
                                     const BigSparseBinaryVecDynamic& b) {
  std::size_t count = 0;
  for (std::size_t k = 0; k < a.chunks.size(); ++k) {
    count += intersection_size(a.chunks[k], b.chunks[k]);
  }
  return count;
}

inline float jaccard(const BigSparseBinaryVecDynamic& a,
                     const BigSparseBinaryVecDynamic& b) {
  std::size_t inter = 0;
  std::size_t total_a = 0;
  std::size_t total_b = 0;
  for (std::size_t k = 0; k < a.chunks.size(); ++k) {
    inter   += intersection_size(a.chunks[k], b.chunks[k]);
    total_a += a.chunks[k].data.size();
    total_b += b.chunks[k].data.size();
  }
  const std::size_t uni = total_a + total_b - inter;
  if (uni == 0) return 0.0f;
  return static_cast<float>(inter) / static_cast<float>(uni);
}

}  // namespace binarycore::binary_vec
