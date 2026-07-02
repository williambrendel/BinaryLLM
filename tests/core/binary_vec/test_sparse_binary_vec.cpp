// ============================================================================
// tests/binarycore/binary_vec/test_sparse_binary_vec.cpp
//
// Tests for sparse_binary_vec.hpp (tier 2). Covers default construction,
// aggregate init, intersection_size two-pointer count, and the jaccard
// convention on a wide range of input shapes.
// ============================================================================

#include "sparse_binary_vec.hpp"
#include "doctest.h"

#include <cstdint>
#include <vector>

using binarycore::binary_vec::SparseBinaryVec;
using binarycore::binary_vec::intersection_size;
using binarycore::binary_vec::jaccard;

TEST_CASE("sparse: default construction is empty") {
  SparseBinaryVec<100> v;
  CHECK(v.data.empty());
}

TEST_CASE("sparse: aggregate init from sorted-unique indices") {
  SparseBinaryVec<100> v{{1, 3, 5, 7}};
  CHECK(v.data == std::vector<std::uint16_t>{1, 3, 5, 7});
}

TEST_CASE("sparse: intersection_size two-pointer count") {
  SparseBinaryVec<100> a{{1, 3, 5, 7, 9}};
  SparseBinaryVec<100> b{{3, 7, 11, 13}};
  CHECK(intersection_size(a, b) == 2u);
}

TEST_CASE("sparse: intersection_size disjoint -> 0") {
  SparseBinaryVec<100> a{{1, 3}};
  SparseBinaryVec<100> b{{5, 7}};
  CHECK(intersection_size(a, b) == 0u);
}

TEST_CASE("sparse: intersection_size identical -> size") {
  SparseBinaryVec<100> a{{1, 3, 5}};
  CHECK(intersection_size(a, a) == 3u);
}

TEST_CASE("sparse: intersection_size with empty -> 0") {
  SparseBinaryVec<100> a{{1, 3, 5}};
  SparseBinaryVec<100> b;
  CHECK(intersection_size(a, b) == 0u);
  CHECK(intersection_size(b, a) == 0u);
}

TEST_CASE("sparse: jaccard half overlap = 2/5") {
  SparseBinaryVec<100> a{{1, 3, 5, 7}};
  SparseBinaryVec<100> b{{3, 5, 9}};
  // |inter|=2, |a|+|b|-|inter| = 4+3-2 = 5
  CHECK(jaccard(a, b) == doctest::Approx(2.0f / 5.0f));
}

TEST_CASE("sparse: jaccard identical -> 1") {
  SparseBinaryVec<100> a{{1, 3, 5}};
  CHECK(jaccard(a, a) == doctest::Approx(1.0f));
}

TEST_CASE("sparse: jaccard disjoint -> 0") {
  SparseBinaryVec<100> a{{1, 3}};
  SparseBinaryVec<100> b{{5, 7}};
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
}

TEST_CASE("sparse: jaccard empty/empty -> 0 by convention") {
  SparseBinaryVec<100> a, b;
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
}

TEST_CASE("sparse: jaccard one empty -> 0") {
  SparseBinaryVec<100> a{{1, 2, 3}};
  SparseBinaryVec<100> b;
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
  CHECK(jaccard(b, a) == doctest::Approx(0.0f));
}

TEST_CASE("sparse: jaccard subset = |smaller|/|larger|") {
  SparseBinaryVec<100> a{{1, 3, 5}};
  SparseBinaryVec<100> b{{1, 3, 5, 7, 9}};
  CHECK(jaccard(a, b) == doctest::Approx(0.6f));  // 3/5
}

TEST_CASE("sparse: jaccard near tier-2 ceiling") {
  SparseBinaryVec<65000> a{{100, 5000, 30000, 64999}};
  SparseBinaryVec<65000> b{{200, 5000, 30000, 50000}};
  // |inter|=2, |union|=6, J=1/3
  CHECK(jaccard(a, b) == doctest::Approx(1.0f / 3.0f));
}
