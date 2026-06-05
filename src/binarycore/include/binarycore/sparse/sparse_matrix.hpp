#pragma once

// ============================================================================
// binarycore/sparse/sparse_matrix.hpp
//
// Compressed Sparse Row (CSR) matrix, value-templated. Three flat arrays:
//   - col_indices[]  — column index of each non-zero, packed row by row
//   - values[]       — value of each non-zero, parallel to col_indices
//   - row_ptr[]      — size rows+1; row_ptr[i] is the offset into the above
//                      where row i's data begins. row_ptr[rows] == nnz.
//
// Within each row, col_indices are sorted ascending. This invariant is
// required by every consumer (dot products, intersections, jaccard) and is
// preserved by all matrix-construction paths.
//
// Use RowView for non-owning per-row access:
//   const auto row_i = M.row(i);
//   for (std::size_t k = 0; k < row_i.nnz; ++k) {
//     auto col = row_i.indices[k];
//     auto val = row_i.values[k];
//   }
//
// Or pass row_i to free functions in ops.hpp.
//
// For building a matrix incrementally (one (i, j, value) triple at a time),
// use SparseMatrixBuilder which converts to CSR at finalization.
// ============================================================================

#include "binarycore/sparse/sparse_vec.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace binarycore::sparse {

// Non-owning view of one row of a SparseMatrix.
template <typename T>
struct RowView {
  const index_type* indices;   // sorted ascending, length nnz
  const T* values;             // parallel to indices, length nnz
  std::size_t nnz;
  std::size_t dim;             // full column dimension of the parent matrix
};

template <typename T>
class SparseMatrix {
public:
  using value_type = T;

  SparseMatrix() noexcept : rows_(0), cols_(0) {
    row_ptr_.push_back(0);
  }

  // Construct from full CSR arrays (move-in).
  // Caller guarantees:
  //   - col_indices.size() == values.size() (== nnz)
  //   - row_ptr.size() == rows + 1
  //   - row_ptr is monotone non-decreasing, row_ptr[0] == 0,
  //     row_ptr[rows] == nnz
  //   - within each row, col_indices are sorted ascending and unique
  SparseMatrix(std::size_t rows, std::size_t cols,
               std::vector<index_type> col_indices,
               std::vector<T> values,
               std::vector<std::size_t> row_ptr) noexcept
      : col_indices_(std::move(col_indices)),
        values_(std::move(values)),
        row_ptr_(std::move(row_ptr)),
        rows_(rows),
        cols_(cols) {
    assert(col_indices_.size() == values_.size());
    assert(row_ptr_.size() == rows + 1);
    assert(row_ptr_.empty() || row_ptr_.back() == values_.size());
  }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }
  std::size_t nnz() const noexcept { return values_.size(); }

  // Non-owning view of row i.
  RowView<T> row(std::size_t i) const noexcept {
    const std::size_t start = row_ptr_[i];
    const std::size_t end = row_ptr_[i + 1];
    return RowView<T>{
        col_indices_.data() + start,
        values_.data() + start,
        end - start,
        cols_
    };
  }

  // Number of non-zeros in row i.
  std::size_t row_nnz(std::size_t i) const noexcept {
    return row_ptr_[i + 1] - row_ptr_[i];
  }

  // Raw accessors (for serialization, low-level ops).
  const std::vector<index_type>& col_indices() const noexcept { return col_indices_; }
  const std::vector<T>& values() const noexcept { return values_; }
  const std::vector<std::size_t>& row_ptr() const noexcept { return row_ptr_; }

private:
  std::vector<index_type> col_indices_;
  std::vector<T> values_;
  std::vector<std::size_t> row_ptr_;
  std::size_t rows_;
  std::size_t cols_;
};

}  // namespace binarycore::sparse
