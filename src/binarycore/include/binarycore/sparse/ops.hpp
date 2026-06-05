#pragma once

// ============================================================================
// binarycore/sparse/ops.hpp
//
// Operations on sparse vectors and matrices. All ops assume sorted-ascending
// indices within each vector / row — the standard invariant for the sparse
// types in this module.
//
// Two-pointer merge is the inner loop behind most pairwise operations
// (dot, intersect_count, jaccard, hadamard, add). It is factored into the
// `detail::merge_traverse` helper so each public op is a small wrapper.
//
// Matrix-matrix product is provided only in the "A times B-transposed" form
// (`multiply_ABT`). Both operands are read row-by-row, which matches CSR's
// access pattern. We never need an explicit transpose for our pipeline.
// ============================================================================

#include "binarycore/sparse/sparse_matrix.hpp"
#include "binarycore/sparse/sparse_matrix_builder.hpp"
#include "binarycore/sparse/sparse_vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace binarycore::sparse {

// ---------------------------------------------------------------------------
// detail::merge_traverse
//
// Walk two sorted-ascending index lists in parallel, invoking callbacks
// at each event:
//   on_match(ia, ib, va, vb)   — same column in both
//   on_a_only(ia, va)          — column present in a but not b
//   on_b_only(ib, vb)          — column present in b but not a
//
// All callbacks take the position k_a / k_b into the respective vectors and
// the column index and value. Used by dot, jaccard, hadamard, etc.
// ---------------------------------------------------------------------------
namespace detail {

template <typename T, typename OnMatch, typename OnAOnly, typename OnBOnly>
void merge_traverse(const index_type* ia, const T* va, std::size_t na,
                    const index_type* ib, const T* vb, std::size_t nb,
                    OnMatch on_match,
                    OnAOnly on_a_only,
                    OnBOnly on_b_only) {
  std::size_t ka = 0, kb = 0;
  while (ka < na && kb < nb) {
    const index_type ca = ia[ka];
    const index_type cb = ib[kb];
    if (ca < cb) {
      on_a_only(ca, va[ka]);
      ++ka;
    } else if (cb < ca) {
      on_b_only(cb, vb[kb]);
      ++kb;
    } else {
      on_match(ca, va[ka], vb[kb]);
      ++ka;
      ++kb;
    }
  }
  while (ka < na) {
    on_a_only(ia[ka], va[ka]);
    ++ka;
  }
  while (kb < nb) {
    on_b_only(ib[kb], vb[kb]);
    ++kb;
  }
}

}  // namespace detail

// ===========================================================================
// Reductions over a pair of sparse vectors / rows
// ===========================================================================

// Inner product. Accumulates va * vb at matching indices.
template <typename T>
T dot(const index_type* ia, const T* va, std::size_t na,
      const index_type* ib, const T* vb, std::size_t nb) noexcept {
  T sum = T{};
  detail::merge_traverse<T>(
      ia, va, na, ib, vb, nb,
      [&](index_type, T a, T b) { sum += a * b; },
      [](index_type, T) {},
      [](index_type, T) {});
  return sum;
}

template <typename T>
T dot(const RowView<T>& a, const RowView<T>& b) noexcept {
  return dot<T>(a.indices, a.values, a.nnz, b.indices, b.values, b.nnz);
}

template <typename T>
T dot(const SparseVec<T>& a, const SparseVec<T>& b) noexcept {
  return dot<T>(a.indices(), a.values(), a.nnz(),
                b.indices(), b.values(), b.nnz());
}

// Number of indices present in both vectors (does not look at values).
template <typename T>
std::size_t intersect_count(const RowView<T>& a, const RowView<T>& b) noexcept {
  std::size_t n = 0;
  detail::merge_traverse<T>(
      a.indices, a.values, a.nnz, b.indices, b.values, b.nnz,
      [&](index_type, T, T) { ++n; },
      [](index_type, T) {},
      [](index_type, T) {});
  return n;
}

// Number of indices present in either vector (does not look at values).
template <typename T>
std::size_t union_count(const RowView<T>& a, const RowView<T>& b) noexcept {
  std::size_t n = 0;
  detail::merge_traverse<T>(
      a.indices, a.values, a.nnz, b.indices, b.values, b.nnz,
      [&](index_type, T, T) { ++n; },
      [&](index_type, T) { ++n; },
      [&](index_type, T) { ++n; });
  return n;
}

// Jaccard distance over the support sets (treating each vector as a binary
// presence pattern, ignoring values): 1 - |intersect| / |union|.
// Returns 0 when both vectors are empty.
template <typename T>
double jaccard_dist(const RowView<T>& a, const RowView<T>& b) noexcept {
  std::size_t i = 0, u = 0;
  detail::merge_traverse<T>(
      a.indices, a.values, a.nnz, b.indices, b.values, b.nnz,
      [&](index_type, T, T) { ++i; ++u; },
      [&](index_type, T) { ++u; },
      [&](index_type, T) { ++u; });
  if (u == 0) return 0.0;
  return 1.0 - static_cast<double>(i) / static_cast<double>(u);
}

// ===========================================================================
// Elementwise ops producing a new sparse vector
// ===========================================================================

// Hadamard (elementwise) product: result has entries only where both inputs
// have entries.
template <typename T>
SparseVec<T> hadamard(const RowView<T>& a, const RowView<T>& b) {
  std::vector<index_type> idx;
  std::vector<T> val;
  detail::merge_traverse<T>(
      a.indices, a.values, a.nnz, b.indices, b.values, b.nnz,
      [&](index_type c, T va, T vb) {
        idx.push_back(c);
        val.push_back(va * vb);
      },
      [](index_type, T) {},
      [](index_type, T) {});
  return SparseVec<T>(std::max(a.dim, b.dim), std::move(idx), std::move(val));
}

// Sum: result has entries where either input has entries; matching positions
// sum their values.
template <typename T>
SparseVec<T> add(const RowView<T>& a, const RowView<T>& b) {
  std::vector<index_type> idx;
  std::vector<T> val;
  detail::merge_traverse<T>(
      a.indices, a.values, a.nnz, b.indices, b.values, b.nnz,
      [&](index_type c, T va, T vb) {
        idx.push_back(c);
        val.push_back(va + vb);
      },
      [&](index_type c, T va) {
        idx.push_back(c);
        val.push_back(va);
      },
      [&](index_type c, T vb) {
        idx.push_back(c);
        val.push_back(vb);
      });
  return SparseVec<T>(std::max(a.dim, b.dim), std::move(idx), std::move(val));
}

// ===========================================================================
// In-place modification
// ===========================================================================

// Drop entries whose absolute value is at or below the threshold. Preserves
// sorted-ascending index ordering.
template <typename T>
void prune_below(SparseVec<T>& v, T threshold) {
  auto& idx = v.indices_vec();
  auto& val = v.values_vec();
  std::size_t w = 0;
  for (std::size_t r = 0; r < val.size(); ++r) {
    if (std::abs(val[r]) > threshold) {
      idx[w] = idx[r];
      val[w] = val[r];
      ++w;
    }
  }
  idx.resize(w);
  val.resize(w);
}

// ===========================================================================
// Matrix product: A * B^T
//
// Computes C[i, j] = dot(A[i, :], B[j, :]) for all (i, j).
// Both operands are read by row — CSR-friendly on both sides.
//
// Implementation: for each row i of A, walk each row j of B; record nonzero
// pair products via the builder. Skips computation when a row is empty.
// For square A * A^T (gram matrix), exploits symmetry: only compute (i, j)
// with j >= i and mirror.
//
// Cost: O(rows(A) * rows(B) * avg_merge_cost) in the worst case. For sparse
// rows with few matches the inner dot is fast; the outer two loops still
// dominate.
// ===========================================================================
template <typename T>
SparseMatrix<T> multiply_ABT(const SparseMatrix<T>& A,
                             const SparseMatrix<T>& B,
                             T threshold = T{}) {
  SparseMatrixBuilder<T> builder(A.rows(), B.rows());
  for (std::size_t i = 0; i < A.rows(); ++i) {
    const auto ra = A.row(i);
    if (ra.nnz == 0) continue;
    for (std::size_t j = 0; j < B.rows(); ++j) {
      const auto rb = B.row(j);
      if (rb.nnz == 0) continue;
      const T d = dot<T>(ra, rb);
      if (d > threshold || d < -threshold) {
        builder.set(i, j, d);
      }
    }
  }
  return std::move(builder).finalize();
}

// Symmetric optimization: C[i, j] = dot(A[i, :], A[j, :]) where C is
// symmetric, so we only compute upper triangle and mirror.
template <typename T>
SparseMatrix<T> gram(const SparseMatrix<T>& A, T threshold = T{}) {
  SparseMatrixBuilder<T> builder(A.rows(), A.rows());
  for (std::size_t i = 0; i < A.rows(); ++i) {
    const auto ri = A.row(i);
    if (ri.nnz == 0) continue;
    for (std::size_t j = i; j < A.rows(); ++j) {
      const auto rj = A.row(j);
      if (rj.nnz == 0) continue;
      const T d = dot<T>(ri, rj);
      if (d > threshold || d < -threshold) {
        builder.set(i, j, d);
        if (i != j) builder.set(j, i, d);
      }
    }
  }
  return std::move(builder).finalize();
}

}  // namespace binarycore::sparse
