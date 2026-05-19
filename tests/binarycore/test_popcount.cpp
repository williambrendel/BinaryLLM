#include "binarycore/math/popcount.hpp"
#include "doctest.h"

using namespace binarycore;

TEST_CASE("popcount on uint64_t") {
  CHECK(popcount(uint64_t{0}) == 0);
  CHECK(popcount(uint64_t{1}) == 1);
  CHECK(popcount(uint64_t{0xFF}) == 8);
  CHECK(popcount(~uint64_t{0}) == 64);
  CHECK(popcount(uint64_t{0x5555555555555555ULL}) == 32);
}

TEST_CASE("popcount on BinaryVec — empty / full") {
  CHECK(popcount(BinaryVec64{}) == 0);
  CHECK(popcount(BinaryVec128{}) == 0);
  CHECK(popcount(BinaryVec256{}) == 0);
  CHECK(popcount(BinaryVec512{}) == 0);

  CHECK(popcount(~BinaryVec64{}) == 64);
  CHECK(popcount(~BinaryVec128{}) == 128);
  CHECK(popcount(~BinaryVec256{}) == 256);
  CHECK(popcount(~BinaryVec512{}) == 512);
}

TEST_CASE("popcount on BinaryVec with sparse bits") {
  BinaryVec128 v;
  v.set(0);
  v.set(63);
  v.set(64);
  v.set(127);
  CHECK(popcount(v) == 4);
}
