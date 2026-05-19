#include "binarycore/math/similarity.hpp"
#include "doctest.h"

using namespace binarycore;

TEST_CASE("similarity: identical vectors → 1.0") {
  CHECK(similarity(BinaryVec64{}, BinaryVec64{}) == doctest::Approx(1.0f));
}

TEST_CASE("similarity: opposite vectors → 0.0") {
  CHECK(similarity(BinaryVec64{}, ~BinaryVec64{}) == doctest::Approx(0.0f));
  CHECK(similarity(BinaryVec128{}, ~BinaryVec128{}) == doctest::Approx(0.0f));
}

TEST_CASE("similarity: one bit different → 1 - 1/DIMS") {
  BinaryVec64 a, b;
  a.set(0);
  CHECK(similarity(a, b) == doctest::Approx(1.0f - 1.0f / 64.0f));

  BinaryVec128 c, d;
  c.set(0);
  CHECK(similarity(c, d) == doctest::Approx(1.0f - 1.0f / 128.0f));
}
