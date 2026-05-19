#include "binarycore/containers/binary_vec.hpp"
#include "doctest.h"

using namespace binarycore;

TEST_CASE("BinaryVec default-constructs to all zeros") {
  BinaryVec64 v;
  for (std::size_t i = 0; i < 64; ++i) {
    CHECK_FALSE(v.get(i));
  }
}

TEST_CASE("BinaryVec set/get/clear individual bits") {
  BinaryVec128 v;
  v.set(0);
  v.set(63);
  v.set(64);
  v.set(127);
  CHECK(v.get(0));
  CHECK(v.get(63));
  CHECK(v.get(64));
  CHECK(v.get(127));
  CHECK_FALSE(v.get(1));
  CHECK_FALSE(v.get(126));

  v.clear(63);
  CHECK_FALSE(v.get(63));
}

TEST_CASE("BinaryVec assign() handles both true and false") {
  BinaryVec64 v;
  v.assign(5, true);
  CHECK(v.get(5));
  v.assign(5, false);
  CHECK_FALSE(v.get(5));
}

TEST_CASE("BinaryVec bitwise AND / OR / XOR / NOT") {
  BinaryVec64 a, b;
  a.set(0);
  a.set(1);
  a.set(2);
  b.set(1);
  b.set(2);
  b.set(3);

  auto and_ = a & b;
  CHECK(and_.get(1));
  CHECK(and_.get(2));
  CHECK_FALSE(and_.get(0));
  CHECK_FALSE(and_.get(3));

  auto or_ = a | b;
  CHECK(or_.get(0));
  CHECK(or_.get(1));
  CHECK(or_.get(2));
  CHECK(or_.get(3));

  auto xor_ = a ^ b;
  CHECK(xor_.get(0));
  CHECK(xor_.get(3));
  CHECK_FALSE(xor_.get(1));
  CHECK_FALSE(xor_.get(2));

  BinaryVec64 zero;
  auto not_zero = ~zero;
  for (std::size_t i = 0; i < 64; ++i) CHECK(not_zero.get(i));
}

TEST_CASE("BinaryVec in-place bitwise operations") {
  BinaryVec64 a, b;
  a.set(0);
  a.set(1);
  b.set(1);
  b.set(2);

  BinaryVec64 c = a;
  c &= b;
  CHECK(c.get(1));
  CHECK_FALSE(c.get(0));

  c = a;
  c |= b;
  CHECK(c.get(0));
  CHECK(c.get(1));
  CHECK(c.get(2));

  c = a;
  c ^= b;
  CHECK(c.get(0));
  CHECK(c.get(2));
  CHECK_FALSE(c.get(1));
}

TEST_CASE("BinaryVec equality") {
  BinaryVec128 a, b;
  a.set(0);
  a.set(99);
  b.set(0);
  b.set(99);
  CHECK(a == b);
  CHECK_FALSE(a != b);
  b.set(50);
  CHECK_FALSE(a == b);
  CHECK(a != b);
}

TEST_CASE("BinaryVec sizes: 64, 128, 256, 512") {
  CHECK(BinaryVec64::Dims == 64);
  CHECK(BinaryVec64::Chunks == 1);
  CHECK(BinaryVec128::Dims == 128);
  CHECK(BinaryVec128::Chunks == 2);
  CHECK(BinaryVec256::Dims == 256);
  CHECK(BinaryVec256::Chunks == 4);
  CHECK(BinaryVec512::Dims == 512);
  CHECK(BinaryVec512::Chunks == 8);
}

TEST_CASE("BinaryVec aggregate initialization from chunk values") {
  BinaryVec128 v{{0xFFFFFFFFFFFFFFFFULL, 0x0ULL}};
  CHECK(v.get(0));
  CHECK(v.get(63));
  CHECK_FALSE(v.get(64));
}
