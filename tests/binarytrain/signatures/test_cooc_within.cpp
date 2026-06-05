// ============================================================================
// tests/binarytrain/signatures/test_cooc_within.cpp
// ============================================================================

#include "binarytrain/signatures/cooc_within_builder.hpp"
#include "binarycore/sparse/sparse_matrix.hpp"
#include "doctest.h"

#include <cstdint>
#include <utility>
#include <vector>

using binarycore::sparse::SparseMatrix;
using binarytrain::signatures::CoocWithinBuilder;

namespace {

double lookup(const SparseMatrix<double>& m, std::uint32_t row,
              std::uint32_t col) {
  const auto r = m.row(row);
  for (std::size_t k = 0; k < r.nnz; ++k) {
    if (r.indices[k] == col) return r.values[k];
  }
  return 0.0;
}

}  // namespace

TEST_CASE("within: empty builder finalizes empty") {
  CoocWithinBuilder cb(10);
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.rows() == 10);
  CHECK(out.c_within.cols() == 10);
  CHECK(out.c_within.nnz() == 0);
  CHECK(out.window_count == 0);
  CHECK(out.total_word_count == 0);
}

TEST_CASE("within: single-part word has no within-pairs") {
  CoocWithinBuilder cb(10);
  cb.add_window({{5}});
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 0);
}

TEST_CASE("within: two distinct parts in one word -> one entry") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 2}});
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 1);
  // Upper triangle: stored at (min, max) = (1, 2)
  CHECK(lookup(out.c_within, 1, 2) == 1.0);
  CHECK(lookup(out.c_within, 2, 1) == 0.0);
}

TEST_CASE("within: canonicalizes to upper triangle regardless of input order") {
  CoocWithinBuilder cb(10);
  cb.add_window({{5, 3}});  // input order (5, 3) but stored at (3, 5)
  auto out = std::move(cb).finalize();
  CHECK(lookup(out.c_within, 3, 5) == 1.0);
  CHECK(lookup(out.c_within, 5, 3) == 0.0);
}

TEST_CASE("within: three distinct parts produce all 3 pairs") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 2, 3}});
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 3);
  CHECK(lookup(out.c_within, 1, 2) == 1.0);
  CHECK(lookup(out.c_within, 1, 3) == 1.0);
  CHECK(lookup(out.c_within, 2, 3) == 1.0);
}

TEST_CASE("within: self-pairs are skipped") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 1}});  // self-pair at positions 0,1 -> skip
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 0);
}

TEST_CASE("within: repeated parts mixed with distinct parts count multiply") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 1, 2}});
  // Position pairs (i<j): (0,1) = self-pair (skip),
  //                       (0,2) = (1,2) -> upper triangle
  //                       (1,2) = (1,2) -> upper triangle
  // Total: C[1, 2] += 2
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 1);
  CHECK(lookup(out.c_within, 1, 2) == 2.0);
}

TEST_CASE("within: multiple words in window each contribute independently") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 2}, {3, 4}, {1, 2}});  // word 0 and 2 both add to C[1,2]
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 2);
  CHECK(lookup(out.c_within, 1, 2) == 2.0);
  CHECK(lookup(out.c_within, 3, 4) == 1.0);
  CHECK(out.window_count == 1);
  CHECK(out.total_word_count == 3);
}

TEST_CASE("within: accumulates over multiple windows") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 2}});
  cb.add_window({{1, 2}});
  cb.add_window({{1, 2}});
  auto out = std::move(cb).finalize();
  CHECK(lookup(out.c_within, 1, 2) == 3.0);
  CHECK(out.window_count == 3);
}

TEST_CASE("within: word with one distinct + repeated part") {
  CoocWithinBuilder cb(10);
  cb.add_window({{2, 2, 2, 5}});
  // Position pairs:
  //   (0,1) self-pair skip
  //   (0,2) self-pair skip
  //   (0,3) (2,5) -> C[2,5] += 1
  //   (1,2) self-pair skip
  //   (1,3) (2,5) -> C[2,5] += 1
  //   (2,3) (2,5) -> C[2,5] += 1
  // Total: C[2, 5] += 3
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 1);
  CHECK(lookup(out.c_within, 2, 5) == 3.0);
}

TEST_CASE("within: empty items in window") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 2}, {}, {3, 4}});
  auto out = std::move(cb).finalize();
  CHECK(out.c_within.nnz() == 2);
  CHECK(lookup(out.c_within, 1, 2) == 1.0);
  CHECK(lookup(out.c_within, 3, 4) == 1.0);
  CHECK(out.total_word_count == 3);
}

TEST_CASE("within: finalized matrix has sorted columns per row") {
  CoocWithinBuilder cb(10);
  cb.add_window({{1, 5, 2, 7, 3}});
  auto out = std::move(cb).finalize();
  for (std::size_t i = 0; i < out.c_within.rows(); ++i) {
    const auto r = out.c_within.row(i);
    for (std::size_t k = 1; k < r.nnz; ++k) {
      CHECK(r.indices[k - 1] < r.indices[k]);
    }
  }
}
