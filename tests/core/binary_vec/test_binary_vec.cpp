// ============================================================================
// tests/binarycore/binary_vec/test_binary_vec.cpp
//
// Tests for binary_vec.hpp (the public entry point). Covers the
// BinaryVec<Dim> compile-time tier dispatch and the BinaryVecDynamic
// runtime alias. Tier-specific behavior is tested in the per-tier
// test files; this file only verifies that the aliases resolve to
// the right types and remain usable through the alias.
// ============================================================================

#include "binary_vec.hpp"
#include "doctest.h"

#include <type_traits>

using namespace binarycore::binary_vec;

TEST_CASE("BinaryVec<Dim> dispatches to the correct tier") {
  static_assert(std::is_same_v<BinaryVec<1>,       DenseBinaryVec<1>>);
  static_assert(std::is_same_v<BinaryVec<2048>,    DenseBinaryVec<2048>>);
  static_assert(std::is_same_v<BinaryVec<4096>,    DenseBinaryVec<4096>>);
  static_assert(std::is_same_v<BinaryVec<4097>,    SparseBinaryVec<4097>>);
  static_assert(std::is_same_v<BinaryVec<65448>,   SparseBinaryVec<65448>>);
  static_assert(std::is_same_v<BinaryVec<65535>,   SparseBinaryVec<65535>>);
  static_assert(std::is_same_v<BinaryVec<65536>,   BigSparseBinaryVec<65536>>);
  static_assert(std::is_same_v<BinaryVec<200000>,  BigSparseBinaryVec<200000>>);
  CHECK(true);
}

TEST_CASE("BinaryVecDynamic aliases BigSparseBinaryVecDynamic") {
  static_assert(std::is_same_v<BinaryVecDynamic, BigSparseBinaryVecDynamic>);
  CHECK(true);
}

TEST_CASE("BinaryVecDynamic usable through the alias") {
  BinaryVecDynamic v(131070);
  CHECK(v.dim == 131070u);
  CHECK(v.chunks.size() == 2u);
}

TEST_CASE("BinaryVecDynamic jaccard via alias") {
  BinaryVecDynamic a(131070), b(131070);
  a.chunks[0].data = {1, 2, 3};
  b.chunks[0].data = {2, 3, 4};
  // |inter|=2, |a|=3, |b|=3, |union|=4 -> J=0.5
  CHECK(jaccard(a, b) == doctest::Approx(0.5f));
}
