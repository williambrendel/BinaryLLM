#include "binarycore/math/dot.hpp"
#include "doctest.h"

using namespace binarycore;

TEST_CASE("dot: counts shared 1-bits") {
  BinaryVec64 a, b;
  a.set(0);
  a.set(1);
  a.set(2);
  b.set(2);
  b.set(3);
  b.set(4);
  CHECK(dot(a, b) == 1);  // only bit 2 in common
  CHECK(dot(b, a) == 1);  // symmetric
}

TEST_CASE("dot(x, x) == popcount(x)") {
  BinaryVec128 v;
  v.set(0);
  v.set(50);
  v.set(127);
  CHECK(dot(v, v) == 3);
}

TEST_CASE("dot(x, 0) == 0") {
  BinaryVec64 v;
  v.set(5);
  v.set(40);
  CHECK(dot(v, BinaryVec64{}) == 0);
  CHECK(dot(BinaryVec64{}, v) == 0);
}

TEST_CASE("dot at all sizes") {
  BinaryVec256 a, b;
  a.set(0);
  a.set(128);
  a.set(255);
  b.set(0);
  b.set(255);
  CHECK(dot(a, b) == 2);
}
