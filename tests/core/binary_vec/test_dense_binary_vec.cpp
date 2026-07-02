// ============================================================================
// tests/binarycore/binary_vec/test_dense_binary_vec.cpp
//
// Tests for dense_binary_vec.hpp (tier 1). Covers default zeroing,
// num_chunks math, set/get_bit at word boundaries, and jaccard
// across sizes that exercise both the 4-way unrolled loop and the
// scalar remainder loop (each remainder mod 4 in {0, 1, 2, 3}).
// ============================================================================

#include "dense_binary_vec.hpp"
#include "doctest.h"

#include <cstdint>

using binarycore::binary_vec::DenseBinaryVec;
using binarycore::binary_vec::set_bit;
using binarycore::binary_vec::get_bit;
using binarycore::binary_vec::jaccard;

TEST_CASE("dense: default is all-zero") {
  DenseBinaryVec<200> v;
  for (auto c : v.chunks) CHECK(c == 0u);
}

TEST_CASE("dense: num_chunks computed correctly") {
  static_assert(DenseBinaryVec<1>::num_chunks    == 1);
  static_assert(DenseBinaryVec<64>::num_chunks   == 1);
  static_assert(DenseBinaryVec<65>::num_chunks   == 2);
  static_assert(DenseBinaryVec<128>::num_chunks  == 2);
  static_assert(DenseBinaryVec<200>::num_chunks  == 4);  // ceil(200/64)
  static_assert(DenseBinaryVec<4096>::num_chunks == 64);
  CHECK(true);
}

TEST_CASE("dense: set/get_bit at word boundaries") {
  DenseBinaryVec<200> v;
  for (std::size_t i : {0u, 63u, 64u, 65u, 127u, 128u, 199u}) set_bit(v, i);
  for (std::size_t i : {0u, 63u, 64u, 65u, 127u, 128u, 199u}) CHECK(get_bit(v, i));
  for (std::size_t i : {1u, 62u, 66u, 100u, 198u}) CHECK_FALSE(get_bit(v, i));
}

TEST_CASE("dense: jaccard identical -> 1") {
  DenseBinaryVec<200> a;
  set_bit(a, 5); set_bit(a, 100); set_bit(a, 199);
  CHECK(jaccard(a, a) == doctest::Approx(1.0f));
}

TEST_CASE("dense: jaccard disjoint -> 0") {
  DenseBinaryVec<200> a, b;
  set_bit(a, 5); set_bit(a, 100);
  set_bit(b, 6); set_bit(b, 101);
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
}

TEST_CASE("dense: jaccard half overlap = 0.5") {
  DenseBinaryVec<200> a, b;
  set_bit(a, 1); set_bit(a, 2); set_bit(a, 3);
  set_bit(b, 2); set_bit(b, 3); set_bit(b, 4);
  CHECK(jaccard(a, b) == doctest::Approx(0.5f));
}

TEST_CASE("dense: jaccard all-zero / all-zero -> 0") {
  DenseBinaryVec<200> a, b;
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
}

TEST_CASE("dense: jaccard one all-zero -> 0") {
  DenseBinaryVec<200> a, b;
  set_bit(a, 5);
  CHECK(jaccard(a, b) == doctest::Approx(0.0f));
  CHECK(jaccard(b, a) == doctest::Approx(0.0f));
}

TEST_CASE("dense: jaccard at 4096-bit ceiling") {
  DenseBinaryVec<4096> v;
  set_bit(v, 0); set_bit(v, 2048); set_bit(v, 4095);
  CHECK(jaccard(v, v) == doctest::Approx(1.0f));
}

TEST_CASE("dense: jaccard 4-way unrolled loop + remainder agree") {
  // 300 bits -> 5 chunks -> N4 = 4, remainder = 1
  DenseBinaryVec<300> a, b;
  set_bit(a, 0); set_bit(a, 64); set_bit(a, 128); set_bit(a, 192); set_bit(a, 256);
  set_bit(b, 0); set_bit(b, 64); set_bit(b, 128); set_bit(b, 192); set_bit(b, 256);
  CHECK(jaccard(a, b) == doctest::Approx(1.0f));  // identical across all 5 chunks

  DenseBinaryVec<300> c;
  set_bit(c, 0); set_bit(c, 256);  // shares only first and last chunk
  // |inter|=2, |a|=5, |c|=2, |union|=5
  CHECK(jaccard(a, c) == doctest::Approx(2.0f / 5.0f));
}

TEST_CASE("dense: jaccard at sizes covering each remainder mod 4") {
  // num_chunks = 1 (remainder 1, N4 = 0): only remainder loop runs
  {
    DenseBinaryVec<32> a, b;
    set_bit(a, 0); set_bit(a, 16); set_bit(a, 31);
    set_bit(b, 0); set_bit(b, 31);  // shares 2
    // |inter|=2, |a|=3, |b|=2, |union|=3
    CHECK(jaccard(a, b) == doctest::Approx(2.0f / 3.0f));
  }
  // num_chunks = 2 (remainder 2, N4 = 0): only remainder loop runs
  {
    DenseBinaryVec<128> a, b;
    set_bit(a, 0); set_bit(a, 100);
    set_bit(b, 0);  // shares 1
    CHECK(jaccard(a, b) == doctest::Approx(0.5f));  // 1/2
  }
  // num_chunks = 4 (remainder 0, N4 = 4): only unrolled loop runs
  {
    DenseBinaryVec<256> a, b;
    set_bit(a, 0); set_bit(a, 255);
    set_bit(b, 0); set_bit(b, 255);
    CHECK(jaccard(a, b) == doctest::Approx(1.0f));
  }
  // num_chunks = 3 (remainder 3, N4 = 0): only remainder loop runs
  {
    DenseBinaryVec<192> a, b;
    set_bit(a, 0); set_bit(a, 64); set_bit(a, 128);
    set_bit(b, 0); set_bit(b, 128);
    // |inter|=2, |a|=3, |b|=2, |union|=3
    CHECK(jaccard(a, b) == doctest::Approx(2.0f / 3.0f));
  }
}
