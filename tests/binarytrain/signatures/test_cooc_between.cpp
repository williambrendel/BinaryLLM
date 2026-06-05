// ============================================================================
// tests/binarytrain/signatures/test_cooc_between.cpp
// ============================================================================

#include "binarytrain/signatures/cooc_between_builder.hpp"
#include "binarycore/sparse/sparse_matrix.hpp"
#include "doctest.h"

#include <cstdint>
#include <utility>
#include <vector>

using binarycore::sparse::SparseMatrix;
using binarytrain::signatures::CoocBetweenBuilder;

namespace {

// Look up a single entry from a finalized SparseMatrix. Returns 0 when
// the entry is absent.
double lookup(const SparseMatrix<double>& m, std::uint32_t row,
              std::uint32_t col) {
  const auto r = m.row(row);
  for (std::size_t k = 0; k < r.nnz; ++k) {
    if (r.indices[k] == col) return r.values[k];
  }
  return 0.0;
}

}  // namespace

// ---------------------------------------------------------------------------
// trivial / empty
// ---------------------------------------------------------------------------

TEST_CASE("between: empty builder finalizes empty matrix and zero freqs") {
  CoocBetweenBuilder cb(10);
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.rows() == 10);
  CHECK(out.c_between.cols() == 10);
  CHECK(out.c_between.nnz() == 0);
  CHECK(out.frequencies.size() == 10);
  for (auto f : out.frequencies) CHECK(f == 0);
  CHECK(out.window_count == 0);
  CHECK(out.total_word_count == 0);
  CHECK(out.total_part_count == 0);
}

TEST_CASE("between: empty window increments only window_count") {
  CoocBetweenBuilder cb(10);
  cb.add_window({});
  CHECK(cb.window_count() == 1);
  CHECK(cb.total_word_count() == 0);
  CHECK(cb.total_part_count() == 0);
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 0);
}

TEST_CASE("between: single-word window contributes only to frequencies") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{3, 4}});  // one word with two parts
  CHECK(cb.window_count() == 1);
  CHECK(cb.total_word_count() == 1);
  CHECK(cb.total_part_count() == 2);
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 0);
  CHECK(out.frequencies[3] == 1);
  CHECK(out.frequencies[4] == 1);
}

// ---------------------------------------------------------------------------
// basic accumulation
// ---------------------------------------------------------------------------

TEST_CASE("between: two single-part words add one (a,b) entry") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{1}, {2}});
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 1);
  CHECK(lookup(out.c_between, 1, 2) == 1.0);
  CHECK(lookup(out.c_between, 2, 1) == 0.0);  // asymmetric
  CHECK(out.frequencies[1] == 1);
  CHECK(out.frequencies[2] == 1);
}

TEST_CASE("between: matrix is asymmetric") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{1}, {2}});  // (1, 2) -> C[1, 2] += 1
  cb.add_window({{2}, {1}});  // (2, 1) -> C[2, 1] += 1
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 2);
  CHECK(lookup(out.c_between, 1, 2) == 1.0);
  CHECK(lookup(out.c_between, 2, 1) == 1.0);
}

TEST_CASE("between: full cartesian over multi-part words") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{1, 2}, {3, 4}, {5, 6}});
  // word pairs (i<j): (0,1), (0,2), (1,2)
  // (0,1): {1,2} x {3,4} -> (1,3), (1,4), (2,3), (2,4)
  // (0,2): {1,2} x {5,6} -> (1,5), (1,6), (2,5), (2,6)
  // (1,2): {3,4} x {5,6} -> (3,5), (3,6), (4,5), (4,6)
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 12);
  // Spot-check a few
  CHECK(lookup(out.c_between, 1, 3) == 1.0);
  CHECK(lookup(out.c_between, 2, 5) == 1.0);
  CHECK(lookup(out.c_between, 4, 6) == 1.0);
  // Reverse direction not present
  CHECK(lookup(out.c_between, 3, 1) == 0.0);
  CHECK(lookup(out.c_between, 5, 2) == 0.0);
  // Frequencies
  for (std::uint32_t p : {1u, 2u, 3u, 4u, 5u, 6u}) {
    CHECK(out.frequencies[p] == 1);
  }
  CHECK(out.total_word_count == 3);
  CHECK(out.total_part_count == 6);
}

TEST_CASE("between: same part at multiple positions accumulates multiply") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{1, 1}, {1, 1}});
  // word 0 has [1, 1], word 1 has [1, 1]
  // For (0, 1): 2 x 2 = 4 contributions to C[1, 1]
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 1);
  CHECK(lookup(out.c_between, 1, 1) == 4.0);
  CHECK(out.frequencies[1] == 4);
}

TEST_CASE("between: accumulates over multiple windows") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{1}, {2}});
  cb.add_window({{1}, {2}});
  cb.add_window({{1}, {2}});
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 1);
  CHECK(lookup(out.c_between, 1, 2) == 3.0);
  CHECK(out.window_count == 3);
  CHECK(out.frequencies[1] == 3);
  CHECK(out.frequencies[2] == 3);
}

TEST_CASE("between: self-pair at different positions counts to diagonal") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{5}, {5}});
  auto out = std::move(cb).finalize();
  CHECK(lookup(out.c_between, 5, 5) == 1.0);
  CHECK(out.frequencies[5] == 2);
}

TEST_CASE("between: empty items inside window are skipped") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{1}, {}, {2}});  // middle item is empty (e.g. stripped delim)
  // (0,1): empty pair, no contributions
  // (0,2): {1} x {2} -> (1, 2)
  // (1,2): empty pair, no contributions
  auto out = std::move(cb).finalize();
  CHECK(out.c_between.nnz() == 1);
  CHECK(lookup(out.c_between, 1, 2) == 1.0);
  CHECK(out.total_word_count == 3);  // empty items still counted
  CHECK(out.total_part_count == 2);
}

// ---------------------------------------------------------------------------
// stats
// ---------------------------------------------------------------------------

TEST_CASE("between: stats track correctly") {
  CoocBetweenBuilder cb(10);
  cb.add_window({{1, 2, 3}, {4, 5}});
  cb.add_window({{6}});
  CHECK(cb.window_count() == 2);
  CHECK(cb.total_word_count() == 3);
  CHECK(cb.total_part_count() == 6);
  CHECK(cb.feature_count() == 10);
  // Check live frequencies access (read-only)
  CHECK(cb.frequencies()[1] == 1);
  CHECK(cb.frequencies()[6] == 1);
  CHECK(cb.frequencies()[9] == 0);
}

TEST_CASE("between: config exposed") {
  CoocBetweenBuilder::Config cfg;
  cfg.include_delimiters = false;
  CoocBetweenBuilder cb(10, cfg);
  CHECK_FALSE(cb.config().include_delimiters);
  // Default constructor uses include_delimiters = true.
  CoocBetweenBuilder cb_default(10);
  CHECK(cb_default.config().include_delimiters);
}

// ---------------------------------------------------------------------------
// CSR invariants of the finalized matrix
// ---------------------------------------------------------------------------

TEST_CASE("between: finalized matrix has sorted column indices per row") {
  CoocBetweenBuilder cb(10);
  // Add entries in arbitrary order
  cb.add_window({{1, 2}, {5, 3}, {4}});
  // Word pairs: (0,1) {1,2}x{5,3}: (1,5),(1,3),(2,5),(2,3)
  //             (0,2) {1,2}x{4}: (1,4),(2,4)
  //             (1,2) {5,3}x{4}: (5,4),(3,4)
  auto out = std::move(cb).finalize();
  for (std::size_t i = 0; i < out.c_between.rows(); ++i) {
    const auto r = out.c_between.row(i);
    for (std::size_t k = 1; k < r.nnz; ++k) {
      CHECK(r.indices[k - 1] < r.indices[k]);
    }
  }
}
