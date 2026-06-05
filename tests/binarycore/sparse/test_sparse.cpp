#include "binarycore/sparse/ops.hpp"
#include "binarycore/sparse/sparse_matrix.hpp"
#include "binarycore/sparse/sparse_matrix_builder.hpp"
#include "binarycore/sparse/sparse_vec.hpp"
#include "doctest.h"

#include <utility>

using namespace binarycore::sparse;

// -----------------------------------------------------------------------------
// SparseVec basics
// -----------------------------------------------------------------------------

TEST_CASE("SparseVec construction and access") {
  SparseVec<double> v(10, {1, 3, 7}, {0.5, -1.0, 2.5});
  CHECK(v.dim() == 10);
  CHECK(v.nnz() == 3);
  CHECK(v.indices()[0] == 1);
  CHECK(v.values()[2] == doctest::Approx(2.5));
}

TEST_CASE("SparseVec from_pairs sorts and stores") {
  std::vector<std::pair<uint32_t, double>> pairs = {{7, 2.5}, {1, 0.5}, {3, -1.0}};
  auto v = SparseVec<double>::from_pairs(10, std::move(pairs));
  CHECK(v.indices()[0] == 1);
  CHECK(v.indices()[1] == 3);
  CHECK(v.indices()[2] == 7);
  CHECK(v.values()[1] == doctest::Approx(-1.0));
}

// -----------------------------------------------------------------------------
// SparseMatrix + Builder
// -----------------------------------------------------------------------------

TEST_CASE("SparseMatrixBuilder finalize produces sorted CSR") {
  SparseMatrixBuilder<double> b(3, 5);
  b.set(0, 4, 1.0);
  b.set(0, 1, 2.0);
  b.set(1, 0, 3.0);
  b.set(2, 2, 4.0);
  auto m = std::move(b).finalize();

  CHECK(m.rows() == 3);
  CHECK(m.cols() == 5);
  CHECK(m.nnz() == 4);

  auto row0 = m.row(0);
  CHECK(row0.nnz == 2);
  CHECK(row0.indices[0] == 1);   // sorted
  CHECK(row0.indices[1] == 4);
  CHECK(row0.values[0] == doctest::Approx(2.0));
  CHECK(row0.values[1] == doctest::Approx(1.0));

  auto row2 = m.row(2);
  CHECK(row2.nnz == 1);
  CHECK(row2.indices[0] == 2);
  CHECK(row2.values[0] == doctest::Approx(4.0));
}

TEST_CASE("SparseMatrixBuilder add accumulates duplicates") {
  SparseMatrixBuilder<double> b(2, 4);
  b.add(0, 1, 1.0);
  b.add(0, 1, 2.0);
  b.add(0, 1, 0.5);
  auto m = std::move(b).finalize();
  CHECK(m.row(0).values[0] == doctest::Approx(3.5));
}

// -----------------------------------------------------------------------------
// Ops: pairwise reductions
// -----------------------------------------------------------------------------

TEST_CASE("dot product over sparse rows") {
  SparseMatrixBuilder<double> b(2, 5);
  b.set(0, 0, 1.0);
  b.set(0, 2, 2.0);
  b.set(0, 4, 3.0);
  b.set(1, 0, 4.0);
  b.set(1, 1, 5.0);  // mismatched col, no contribution
  b.set(1, 2, 6.0);
  auto m = std::move(b).finalize();

  // dot = 1*4 + 2*6 = 16
  CHECK(dot<double>(m.row(0), m.row(1)) == doctest::Approx(16.0));
}

TEST_CASE("intersect_count and union_count") {
  SparseMatrixBuilder<double> b(2, 10);
  b.set(0, 1, 1.0); b.set(0, 3, 1.0); b.set(0, 5, 1.0);
  b.set(1, 1, 1.0); b.set(1, 2, 1.0); b.set(1, 5, 1.0); b.set(1, 7, 1.0);
  auto m = std::move(b).finalize();

  CHECK(intersect_count(m.row(0), m.row(1)) == 2);   // {1, 5}
  CHECK(union_count(m.row(0), m.row(1)) == 5);       // {1, 2, 3, 5, 7}
}

TEST_CASE("jaccard distance") {
  SparseMatrixBuilder<double> b(3, 10);
  b.set(0, 1, 1.0); b.set(0, 3, 1.0); b.set(0, 5, 1.0);
  b.set(1, 1, 1.0); b.set(1, 3, 1.0); b.set(1, 5, 1.0);  // identical to row 0
  b.set(2, 7, 1.0); b.set(2, 9, 1.0);                     // disjoint from row 0
  auto m = std::move(b).finalize();

  CHECK(jaccard_dist(m.row(0), m.row(1)) == doctest::Approx(0.0));
  CHECK(jaccard_dist(m.row(0), m.row(2)) == doctest::Approx(1.0));
}

TEST_CASE("jaccard distance on empty inputs returns 0") {
  SparseMatrix<double> empty(2, 5, {}, {}, {0, 0, 0});
  CHECK(jaccard_dist(empty.row(0), empty.row(1)) == doctest::Approx(0.0));
}

// -----------------------------------------------------------------------------
// Ops: elementwise
// -----------------------------------------------------------------------------

TEST_CASE("hadamard product picks only matching indices") {
  SparseMatrixBuilder<double> b(2, 10);
  b.set(0, 1, 2.0); b.set(0, 3, 3.0); b.set(0, 5, 4.0);
  b.set(1, 1, 10.0); b.set(1, 4, 5.0); b.set(1, 5, 100.0);
  auto m = std::move(b).finalize();

  auto h = hadamard<double>(m.row(0), m.row(1));
  CHECK(h.nnz() == 2);    // matches at columns 1 and 5
  CHECK(h.indices()[0] == 1);
  CHECK(h.indices()[1] == 5);
  CHECK(h.values()[0] == doctest::Approx(20.0));   // 2 * 10
  CHECK(h.values()[1] == doctest::Approx(400.0));  // 4 * 100
}

TEST_CASE("add merges and sums") {
  SparseMatrixBuilder<double> b(2, 10);
  b.set(0, 1, 1.0); b.set(0, 3, 2.0);
  b.set(1, 1, 10.0); b.set(1, 4, 4.0);
  auto m = std::move(b).finalize();

  auto s = add<double>(m.row(0), m.row(1));
  CHECK(s.nnz() == 3);
  CHECK(s.indices()[0] == 1); CHECK(s.values()[0] == doctest::Approx(11.0));
  CHECK(s.indices()[1] == 3); CHECK(s.values()[1] == doctest::Approx(2.0));
  CHECK(s.indices()[2] == 4); CHECK(s.values()[2] == doctest::Approx(4.0));
}

// -----------------------------------------------------------------------------
// Ops: in-place modification
// -----------------------------------------------------------------------------

TEST_CASE("prune_below drops small absolute values") {
  SparseVec<double> v(10, {1, 3, 5, 7}, {0.01, -2.0, 0.001, 3.5});
  prune_below(v, 0.1);
  CHECK(v.nnz() == 2);
  CHECK(v.indices()[0] == 3);
  CHECK(v.indices()[1] == 7);
  CHECK(v.values()[0] == doctest::Approx(-2.0));
  CHECK(v.values()[1] == doctest::Approx(3.5));
}

// -----------------------------------------------------------------------------
// Ops: matrix product (A * B^T) and gram
// -----------------------------------------------------------------------------

TEST_CASE("multiply_ABT computes pairwise row dot products") {
  SparseMatrixBuilder<double> ab(2, 4);
  ab.set(0, 0, 1.0); ab.set(0, 2, 2.0);
  ab.set(1, 1, 3.0); ab.set(1, 2, 4.0);
  auto A = std::move(ab).finalize();

  SparseMatrixBuilder<double> bb(3, 4);
  bb.set(0, 0, 5.0); bb.set(0, 2, 6.0);   // A[0]*B[0] = 1*5 + 2*6 = 17
  bb.set(1, 1, 7.0);                       // A[0]*B[1] = 0; A[1]*B[1] = 3*7 = 21
  bb.set(2, 2, 8.0);                       // A[0]*B[2] = 16; A[1]*B[2] = 32
  auto B = std::move(bb).finalize();

  auto C = multiply_ABT<double>(A, B);
  CHECK(C.rows() == 2);
  CHECK(C.cols() == 3);

  auto r0 = C.row(0);
  // expect: (0, 17), (2, 16)
  CHECK(r0.nnz == 2);
  CHECK(r0.indices[0] == 0); CHECK(r0.values[0] == doctest::Approx(17.0));
  CHECK(r0.indices[1] == 2); CHECK(r0.values[1] == doctest::Approx(16.0));

  auto r1 = C.row(1);
  // expect: (0, 24), (1, 21), (2, 32)  (A[1] hits B[0] at col 2)
  CHECK(r1.nnz == 3);
  CHECK(r1.indices[0] == 0); CHECK(r1.values[0] == doctest::Approx(24.0));
  CHECK(r1.indices[1] == 1); CHECK(r1.values[1] == doctest::Approx(21.0));
  CHECK(r1.indices[2] == 2); CHECK(r1.values[2] == doctest::Approx(32.0));
}

TEST_CASE("gram is symmetric") {
  SparseMatrixBuilder<double> b(3, 4);
  b.set(0, 0, 1.0); b.set(0, 1, 2.0);
  b.set(1, 0, 3.0); b.set(1, 1, 4.0);
  b.set(2, 2, 5.0); b.set(2, 3, 6.0);
  auto A = std::move(b).finalize();

  auto G = gram<double>(A);
  CHECK(G.rows() == 3);
  CHECK(G.cols() == 3);

  // Pull all entries and check symmetry.
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      // Look up G[i, j] and G[j, i] via row scan.
      auto find_val = [&](std::size_t r, std::size_t c) -> double {
        auto row = G.row(r);
        for (std::size_t k = 0; k < row.nnz; ++k) {
          if (row.indices[k] == c) return row.values[k];
        }
        return 0.0;
      };
      CHECK(find_val(i, j) == doctest::Approx(find_val(j, i)));
    }
  }
}

TEST_CASE("multiply_ABT threshold drops small entries") {
  SparseMatrixBuilder<double> ab(1, 3);
  ab.set(0, 0, 0.01); ab.set(0, 1, 1.0); ab.set(0, 2, 0.01);
  auto A = std::move(ab).finalize();

  SparseMatrixBuilder<double> bb(2, 3);
  bb.set(0, 0, 0.1); bb.set(0, 2, 0.1);   // dot = 0.001 + 0.001 = 0.002
  bb.set(1, 1, 1.0);                       // dot = 1.0
  auto B = std::move(bb).finalize();

  auto C = multiply_ABT<double>(A, B, /*threshold=*/0.5);
  CHECK(C.nnz() == 1);                     // only the dot=1 entry survives
  CHECK(C.row(0).indices[0] == 1);
  CHECK(C.row(0).values[0] == doctest::Approx(1.0));
}
