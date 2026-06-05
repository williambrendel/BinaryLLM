#pragma once

// ============================================================================
// binarycore/sparse/sparse_vec.hpp
//
// Sparse vector type, value-templated. Stores only the non-zero entries:
//   - indices[]  — sorted ascending positions of non-zero entries
//   - values[]   — corresponding values, parallel to indices
//   - dim        — full vector dimension (number of positions, mostly zero)
//
// Layout is SoA (separate index and value arrays) for SIMD-friendliness on
// index-only operations (intersection counts, presence tests). Operations
// like dot product touch both arrays; the cache cost is acceptable because
// sparse vectors are small enough to fit in L1/L2 entirely.
//
// Indices must be sorted ascending and unique. Construction helpers enforce
// this invariant; mutating operations preserve it.
// ============================================================================

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace binarycore::sparse {

using index_type = uint32_t;

template <typename T>
class SparseVec {
public:
  using value_type = T;

  // Empty vector of given dimension.
  SparseVec() noexcept : dim_(0) {}
  explicit SparseVec(std::size_t dim) noexcept : dim_(dim) {}

  // Move-construct from index/value pairs (caller guarantees indices are
  // sorted ascending and unique).
  SparseVec(std::size_t dim,
            std::vector<index_type> idx,
            std::vector<T> vals) noexcept
      : indices_(std::move(idx)),
        values_(std::move(vals)),
        dim_(dim) {
    assert(indices_.size() == values_.size());
  }

  // Construct from unordered (col, value) pairs; sorts internally.
  template <typename Pair>
  static SparseVec from_pairs(std::size_t dim, std::vector<Pair> pairs) {
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.first < b.first; });
    std::vector<index_type> idx;
    std::vector<T> vals;
    idx.reserve(pairs.size());
    vals.reserve(pairs.size());
    for (auto& p : pairs) {
      idx.push_back(static_cast<index_type>(p.first));
      vals.push_back(p.second);
    }
    return SparseVec(dim, std::move(idx), std::move(vals));
  }

  std::size_t dim() const noexcept { return dim_; }
  std::size_t nnz() const noexcept { return indices_.size(); }

  const index_type* indices() const noexcept { return indices_.data(); }
  const T* values() const noexcept { return values_.data(); }

  index_type* indices_mut() noexcept { return indices_.data(); }
  T* values_mut() noexcept { return values_.data(); }

  // Internal accessors for ops to mutate / resize.
  std::vector<index_type>& indices_vec() noexcept { return indices_; }
  std::vector<T>& values_vec() noexcept { return values_; }

  // Iteration helpers.
  struct const_iterator {
    const index_type* idx;
    const T* val;
    std::size_t i;

    std::pair<index_type, T> operator*() const noexcept {
      return {idx[i], val[i]};
    }
    const_iterator& operator++() noexcept { ++i; return *this; }
    bool operator!=(const const_iterator& o) const noexcept { return i != o.i; }
  };

  const_iterator begin() const noexcept {
    return {indices_.data(), values_.data(), 0};
  }
  const_iterator end() const noexcept {
    return {indices_.data(), values_.data(), indices_.size()};
  }

private:
  std::vector<index_type> indices_;   // sorted ascending, unique
  std::vector<T> values_;             // parallel to indices_
  std::size_t dim_;                   // full dimension
};

}  // namespace binarycore::sparse
