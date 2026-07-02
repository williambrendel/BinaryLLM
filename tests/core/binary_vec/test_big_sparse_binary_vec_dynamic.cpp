// ============================================================================
// tests/binarycore/binary_vec/test_big_sparse_binary_vec_dynamic.cpp
//
// Tests for big_sparse_binary_vec_dynamic.hpp (tier 3 runtime).
// Covers construction with various runtime dims (including dims
// smaller than chunk_size — single-chunk case), and parity with
// the compile-time variant on the same data shape.
// ============================================================================

#include "big_sparse_binary_vec_dynamic.hpp"
#include "doctest.h"

#include <cstdint>

using binarycore::binary_vec::BigSparseBinaryVecDynamic;
using binarycore::binary_vec::intersection_size;
using binarycore::binary_vec::jaccard;

TEST_CASE("big_sparse_dynamic: default construction is empty") {
  BigSparseBinaryVecDynamic v;
  CHECK(v.dim == 0u);
  CHECK(v.chunks.empty());
}

TEST_CASE("big_sparse_dynamic: explicit dim allocates correct chunk count") {
  BigSparseBinaryVecDynamic a(131070);  // 2 * 65535
  CHECK(a.dim == 131070u);
  CHECK(a.chunks.size() == 2u);

  BigSparseBinaryVecDynamic b(131071);  // crosses to a 3rd chunk
  CHECK(b.chunks.size() == 3u);

  BigSparseBinaryVecDynamic c(65448);   // smaller than one chunk; still 1
  CHECK(c.chunks.size() == 1u);

  BigSparseBinaryVecDynamic d(1);       // tiny; still 1 chunk
  CHECK(d.chunks.size() == 1u);
}

TEST_CASE("big_sparse_dynamic: jaccard matches compile-time variant") {
  // Same data shape as the test in test_big_sparse_binary_vec.cpp.
  BigSparseBinaryVecDynamic a(131070), b(131070);
  a.chunks[0].data = {10, 100, 1000};
  a.chunks[1].data = {50, 500, 5000};
  b.chunks[0].data = {100, 200};
  b.chunks[1].data = {500, 5000, 7000};
  // |inter|=3, |a|=6, |b|=5, |union|=8 -> J = 3/8
  CHECK(jaccard(a, b) == doctest::Approx(3.0f / 8.0f));
}

TEST_CASE("big_sparse_dynamic: empty/empty -> 0") {
  BigSparseBinaryVecDynamic a(131070), b(131070);
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
}

TEST_CASE("big_sparse_dynamic: identical -> 1") {
  BigSparseBinaryVecDynamic a(131070);
  a.chunks[0].data = {10, 100};
  a.chunks[1].data = {50, 5000};
  CHECK(jaccard(a, a) == doctest::Approx(1.0f));
}

TEST_CASE("big_sparse_dynamic: intersection_size helper") {
  BigSparseBinaryVecDynamic a(131070), b(131070);
  a.chunks[0].data = {10, 20, 30};
  b.chunks[0].data = {20, 30, 40};
  a.chunks[1].data = {100, 200};
  b.chunks[1].data = {200, 300};
  CHECK(intersection_size(a, b) == 3u);  // {20, 30} + {200}
}

TEST_CASE("big_sparse_dynamic: single-chunk case works for small dim") {
  // The dynamic variant covers the full size range — for any
  // dim ≤ 65535 it just allocates one chunk and behaves like a
  // wrapped SparseBinaryVec.
  BigSparseBinaryVecDynamic a(50000), b(50000);
  CHECK(a.chunks.size() == 1u);
  a.chunks[0].data = {100, 1000, 49999};
  b.chunks[0].data = {1000, 49999};
  // |inter|=2, |a|=3, |b|=2, |union|=3 -> J=2/3
  CHECK(jaccard(a, b) == doctest::Approx(2.0f / 3.0f));
}
