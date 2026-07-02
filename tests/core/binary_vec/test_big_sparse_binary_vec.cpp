// ============================================================================
// tests/binarycore/binary_vec/test_big_sparse_binary_vec.cpp
//
// Tests for big_sparse_binary_vec.hpp (tier 3 compile-time).
// Covers num_chunks math, per-chunk routing for intersection_size
// and jaccard, and the empty-chunk-pair case.
// ============================================================================

#include "big_sparse_binary_vec.hpp"
#include "doctest.h"

#include <cstdint>

using binarycore::binary_vec::BigSparseBinaryVec;
using binarycore::binary_vec::intersection_size;
using binarycore::binary_vec::jaccard;

TEST_CASE("big_sparse: num_chunks computation") {
  static_assert(BigSparseBinaryVec<65536>::num_chunks  == 2);  // ceil(65536/65535)
  static_assert(BigSparseBinaryVec<131070>::num_chunks == 2);  // 2*65535
  static_assert(BigSparseBinaryVec<131071>::num_chunks == 3);  // crosses to chunk 3
  static_assert(BigSparseBinaryVec<200000>::num_chunks == 4);
  CHECK(true);
}

TEST_CASE("big_sparse: jaccard aggregates across chunks") {
  BigSparseBinaryVec<131070> a, b;
  a.chunks[0].data = {10, 100, 1000};   // 3
  a.chunks[1].data = {50, 500, 5000};   // 3
  b.chunks[0].data = {100, 200};        // 2, shares 1
  b.chunks[1].data = {500, 5000, 7000}; // 3, shares 2
  // |inter| = 3, |a| = 6, |b| = 5, |union| = 8 -> J = 3/8
  CHECK(jaccard(a, b) == doctest::Approx(3.0f / 8.0f));
}

TEST_CASE("big_sparse: intersection_size aggregates across chunks") {
  BigSparseBinaryVec<131070> a, b;
  a.chunks[0].data = {10, 100, 1000};
  a.chunks[1].data = {50, 500, 5000};
  b.chunks[0].data = {100, 200};
  b.chunks[1].data = {500, 5000, 7000};
  CHECK(intersection_size(a, b) == 3u);
}

TEST_CASE("big_sparse: empty/empty -> 0") {
  BigSparseBinaryVec<131070> a, b;
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
}

TEST_CASE("big_sparse: identical -> 1") {
  BigSparseBinaryVec<131070> a;
  a.chunks[0].data = {10, 100};
  a.chunks[1].data = {50, 5000};
  CHECK(jaccard(a, a) == doctest::Approx(1.0f));
}

TEST_CASE("big_sparse: jaccard with one all-empty chunk pair") {
  // Chunk 0 has data; chunk 1 is empty on both sides. Empty chunks
  // contribute 0 to all three sums and should not affect J.
  BigSparseBinaryVec<131070> a, b;
  a.chunks[0].data = {10, 20, 30};
  b.chunks[0].data = {20, 30, 40};
  // |inter|=2, |a|=3, |b|=3, |union|=4 -> J = 0.5
  CHECK(jaccard(a, b) == doctest::Approx(0.5f));
}
