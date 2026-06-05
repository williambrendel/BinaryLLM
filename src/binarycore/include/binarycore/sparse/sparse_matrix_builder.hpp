#pragma once

// ============================================================================
// binarycore/sparse/sparse_matrix_builder.hpp
//
// Incremental builder for SparseMatrix. Use when you don't know the
// (row, col) entries upfront (e.g., accumulating C from corpus windows).
// Each row is a hash map of column -> value during accumulation; finalize()
// converts to CSR storage and sorts within each row.
//
// Workflow:
//   SparseMatrixBuilder<double> B(rows, cols);
//   for each window:
//     for each co-occurring pair (i, j):
//       B.add(i, j, 1.0);   // accumulates (sums on duplicate keys)
//   SparseMatrix<double> M = std::move(B).finalize();
//
// Cost: O(nnz) for add and O(nnz log avg_row_density) for finalize.
// ============================================================================

#include "binarycore/sparse/sparse_matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace binarycore::sparse {

template <typename T>
class SparseMatrixBuilder {
public:
  SparseMatrixBuilder(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data_(rows) {}

  // Add value to entry (row, col). On duplicate (row, col), sums the values.
  void add(std::size_t row, std::size_t col, T value) {
    auto& row_map = data_[row];
    auto it = row_map.find(static_cast<index_type>(col));
    if (it == row_map.end()) {
      row_map.emplace(static_cast<index_type>(col), value);
    } else {
      it->second += value;
    }
  }

  // Set value at entry (row, col), overwriting any previous value.
  void set(std::size_t row, std::size_t col, T value) {
    data_[row][static_cast<index_type>(col)] = value;
  }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }

  // Convert to CSR. After this call the builder is left in an empty state.
  SparseMatrix<T> finalize() && {
    std::vector<std::size_t> row_ptr(rows_ + 1, 0);
    std::size_t total = 0;
    for (std::size_t i = 0; i < rows_; ++i) {
      total += data_[i].size();
      row_ptr[i + 1] = total;
    }

    std::vector<index_type> col_indices(total);
    std::vector<T> values(total);

    for (std::size_t i = 0; i < rows_; ++i) {
      // Pull entries out into a temp vector for sorting by index.
      const std::size_t start = row_ptr[i];
      std::size_t k = start;
      for (auto& [col, v] : data_[i]) {
        col_indices[k] = col;
        values[k] = v;
        ++k;
      }
      // Sort this row by col_index ascending.
      // Pair-up indices and values, sort, write back.
      // For typical row sizes (dozens to hundreds), this is fast.
      const std::size_t n = data_[i].size();
      std::vector<std::size_t> perm(n);
      for (std::size_t j = 0; j < n; ++j) perm[j] = j;
      std::sort(perm.begin(), perm.end(),
                [&](std::size_t a, std::size_t b) {
                  return col_indices[start + a] < col_indices[start + b];
                });

      std::vector<index_type> sorted_idx(n);
      std::vector<T> sorted_val(n);
      for (std::size_t j = 0; j < n; ++j) {
        sorted_idx[j] = col_indices[start + perm[j]];
        sorted_val[j] = values[start + perm[j]];
      }
      for (std::size_t j = 0; j < n; ++j) {
        col_indices[start + j] = sorted_idx[j];
        values[start + j] = sorted_val[j];
      }
    }

    // Reset state.
    data_.clear();
    data_.shrink_to_fit();

    return SparseMatrix<T>(rows_, cols_, std::move(col_indices),
                           std::move(values), std::move(row_ptr));
  }

private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<std::unordered_map<index_type, T>> data_;
};

}  // namespace binarycore::sparse
