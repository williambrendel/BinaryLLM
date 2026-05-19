#include "binarycore/math/hamming.hpp"
#include "doctest.h"

using namespace binarycore;

TEST_CASE("hamming: identical vectors → 0") {
  BinaryVec64 a, b;
  CHECK(hamming(a, b) == 0);

  a.set(3);
  a.set(10);
  b.set(3);
  b.set(10);
  CHECK(hamming(a, b) == 0);
}

TEST_CASE("hamming: counts differing bit positions") {
  BinaryVec64 a, b;
  a.set(0);
  a.set(1);
  a.set(2);
  b.set(2);
  b.set(3);
  b.set(4);
  CHECK(hamming(a, b) == 4);  // {0,1,3,4} differ
  CHECK(hamming(b, a) == 4);  // symmetric
}

TEST_CASE("hamming: max distance = all bits flipped") {
  CHECK(hamming(BinaryVec64{}, ~BinaryVec64{}) == 64);
  CHECK(hamming(BinaryVec128{}, ~BinaryVec128{}) == 128);
  CHECK(hamming(BinaryVec256{}, ~BinaryVec256{}) == 256);
  CHECK(hamming(BinaryVec512{}, ~BinaryVec512{}) == 512);
}

TEST_CASE("hamming: works across chunk boundaries") {
  BinaryVec128 a, b;
  a.set(63);   // last bit of chunk 0
  a.set(64);   // first bit of chunk 1
  b.set(64);   // only chunk 1 set
  CHECK(hamming(a, b) == 1);
}
