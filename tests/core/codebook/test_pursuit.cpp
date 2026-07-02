// ============================================================================
// tests/core/codebook/test_pursuit.cpp
//
// α-threshold pursuit (spec §4.3): hard (α=1, argmax) vs graded (α<1, a set),
// candidate correctness (disjoint atoms never fire), coverage/residual, budget.
// ============================================================================

#include "pursuit.hpp"

#include <cstdint>
#include <vector>

#include "doctest.h"

using core::codebook::Atom;
using core::codebook::Codebook;
using core::codebook::pursuit_encode;
using core::codebook::PursuitConfig;

namespace {
// Uniform weights → match reduces to plain containment |res∩φ| / |φ|.
std::vector<std::uint16_t> ones(std::size_t dim) {
  return std::vector<std::uint16_t>(dim, 1);
}
}  // namespace

TEST_CASE("pursuit: α=1 fires the single best match (argmax)") {
  Codebook cb(50, 8);
  cb.add(Atom{0, 1, 2, 3});  // 0: full match to signature
  cb.add(Atom{2, 3, 4, 5});  // 1: half match
  cb.add(Atom{10, 11, 12});  // 2: disjoint
  const std::vector<std::uint32_t> sig = {0, 1, 2, 3};

  PursuitConfig hard{1, 1, 32};
  const auto fired = pursuit_encode(cb, sig, ones(50), hard);
  CHECK(fired == std::vector<std::uint32_t>{0});  // only the argmax
}

TEST_CASE("pursuit: α<1 fires a graded set") {
  Codebook cb(50, 8);
  cb.add(Atom{0, 1, 2, 3});  // match 4/4 = 1.0  (best)
  cb.add(Atom{2, 3, 4, 5});  // match 2/4 = 0.5
  cb.add(Atom{10, 11, 12});  // disjoint
  const std::vector<std::uint32_t> sig = {0, 1, 2, 3};

  // α = 1/2: atom 1 (0.5) meets 0.5·best=0.5 → both fire in one iteration.
  PursuitConfig graded{1, 2, 32};
  const auto fired = pursuit_encode(cb, sig, ones(50), graded);
  CHECK(fired == std::vector<std::uint32_t>{0, 1});
}

TEST_CASE("pursuit: disjoint atoms never fire") {
  Codebook cb(50, 8);
  cb.add(Atom{0, 1, 2});
  cb.add(Atom{20, 21, 22});  // shares nothing with the signature
  const std::vector<std::uint32_t> sig = {0, 1, 2};
  const auto fired = pursuit_encode(cb, sig, ones(50), PursuitConfig{1, 2, 32});
  CHECK(fired == std::vector<std::uint32_t>{0});
}

TEST_CASE("pursuit: covers residual across iterations") {
  Codebook cb(50, 8);
  cb.add(Atom{0, 1, 2, 3});  // covers left half of signature
  cb.add(Atom{4, 5, 6, 7});  // covers right half
  const std::vector<std::uint32_t> sig = {0, 1, 2, 3, 4, 5, 6, 7};
  // α=1: iteration 1 fires one atom (tie → both are equal best 1.0, so both);
  // use α slightly <1 to allow the second-half atom on the next residual too.
  const auto fired = pursuit_encode(cb, sig, ones(50), PursuitConfig{1, 1, 32});
  // Both atoms are needed to cover the signature; both perfectly contained.
  CHECK(fired == std::vector<std::uint32_t>{0, 1});
}

TEST_CASE("pursuit: budget caps the fired count") {
  Codebook cb(100, 8);
  for (std::uint32_t g = 0; g < 10; ++g) cb.add(Atom{g});  // 10 singletons
  std::vector<std::uint32_t> sig;
  for (std::uint32_t g = 0; g < 10; ++g) sig.push_back(g);
  PursuitConfig cfg{1, 2, 3};  // budget = 3
  const auto fired = pursuit_encode(cb, sig, ones(100), cfg);
  CHECK(fired.size() <= 3);
}

TEST_CASE("pursuit: empty signature / empty codebook → no firing") {
  Codebook cb(50, 8);
  cb.add(Atom{0, 1});
  CHECK(pursuit_encode(cb, {}, ones(50)).empty());
  Codebook empty(50, 8);
  CHECK(pursuit_encode(empty, {0, 1}, ones(50)).empty());
}
