// ============================================================================
// tests/core/sparse/test_info_jaccard.cpp
//
// Contracts for the surprisal-weighted similarity primitives (spec §7.5.0):
// info_content, weighted_dot, info_weighted_jaccard (cached + uncached),
// surprisal, rescale_weights_to_max. Positions are sorted-ascending set-bit
// index lists; weights are indexed by position.
// ============================================================================

#include "info_jaccard.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "doctest.h"

using binarycore::sparse::info_content;
using binarycore::sparse::info_weighted_jaccard;
using binarycore::sparse::rescale_weights_to_max;
using binarycore::sparse::surprisal;
using binarycore::sparse::weighted_dot;

namespace {
// w[i] = (i+1)/10 → info_content({0,2,4}) = 0.1 + 0.3 + 0.5 = 0.9.
const std::vector<float> kW = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
}  // namespace

TEST_CASE("info_content") {
  const std::vector<std::uint32_t> a = {0, 2, 4};

  SUBCASE("empty vector -> 0") {
    CHECK(info_content<float>(nullptr, 0, kW.data()) == 0.0f);
  }
  SUBCASE("sums weights at set positions") {
    CHECK(info_content<float>(a.data(), a.size(), kW.data()) ==
          doctest::Approx(0.9f));
  }
  SUBCASE("only zero-weight positions -> 0") {
    const std::vector<float> zero(8, 0.0f);
    CHECK(info_content<float>(a.data(), a.size(), zero.data()) == 0.0f);
  }
  SUBCASE("equals weighted_dot with self") {
    CHECK(info_content<float>(a.data(), a.size(), kW.data()) ==
          weighted_dot<float>(a.data(), a.size(), a.data(), a.size(),
                              kW.data()));
  }
}

TEST_CASE("info_weighted_jaccard") {
  const std::vector<std::uint32_t> a = {0, 1, 2, 3};
  const std::vector<std::uint32_t> b = {2, 3, 4, 5};  // partial overlap {2,3}
  const std::vector<std::uint32_t> d = {5, 6, 7};     // disjoint from a

  SUBCASE("both empty -> 0") {
    CHECK(info_weighted_jaccard<float>(nullptr, 0, nullptr, 0, kW.data()) ==
          0.0);
  }
  SUBCASE("identical vectors -> 1") {
    CHECK(info_weighted_jaccard<float>(a.data(), a.size(), a.data(), a.size(),
                                       kW.data()) == doctest::Approx(1.0));
  }
  SUBCASE("disjoint supports -> 0") {
    CHECK(info_weighted_jaccard<float>(a.data(), a.size(), d.data(), d.size(),
                                       kW.data()) == 0.0);
  }
  SUBCASE("bounded in [0, 1] on a representative partial overlap") {
    const double j = info_weighted_jaccard<float>(
        a.data(), a.size(), b.data(), b.size(), kW.data());
    CHECK(j > 0.0);
    CHECK(j < 1.0);
  }
  SUBCASE("symmetric in its arguments") {
    const double ab = info_weighted_jaccard<float>(
        a.data(), a.size(), b.data(), b.size(), kW.data());
    const double ba = info_weighted_jaccard<float>(
        b.data(), b.size(), a.data(), a.size(), kW.data());
    CHECK(ab == doctest::Approx(ba));
  }
  SUBCASE("uniform weights equals plain set Jaccard") {
    const std::vector<float> ones(8, 1.0f);
    // |a ∩ b| = 2, |a ∪ b| = 6 -> 1/3.
    CHECK(info_weighted_jaccard<float>(a.data(), a.size(), b.data(), b.size(),
                                       ones.data()) ==
          doctest::Approx(2.0 / 6.0));
  }
  SUBCASE("scale invariance under uniform rescaling") {
    std::vector<float> w5;
    for (float w : kW) w5.push_back(5.0f * w);
    const double j1 = info_weighted_jaccard<float>(
        a.data(), a.size(), b.data(), b.size(), kW.data());
    const double j5 = info_weighted_jaccard<float>(
        a.data(), a.size(), b.data(), b.size(), w5.data());
    CHECK(j1 == doctest::Approx(j5));
  }
  SUBCASE("short-circuits when shared bits are all zero-weight") {
    // a and b share {2,3}; zero those weights -> wdot 0 -> jaccard 0.
    std::vector<float> w = kW;
    w[2] = w[3] = 0.0f;
    CHECK(info_weighted_jaccard<float>(a.data(), a.size(), b.data(), b.size(),
                                       w.data()) == 0.0);
  }
  SUBCASE("cached: matches uncached on partial overlap") {
    const float ia = info_content<float>(a.data(), a.size(), kW.data());
    const float ib = info_content<float>(b.data(), b.size(), kW.data());
    const double cached = info_weighted_jaccard<float>(
        a.data(), a.size(), ia, b.data(), b.size(), ib, kW.data());
    const double uncached = info_weighted_jaccard<float>(
        a.data(), a.size(), b.data(), b.size(), kW.data());
    CHECK(cached == doctest::Approx(uncached));
  }
  SUBCASE("cached: short-circuits on zero wdot regardless of Ia/Ib") {
    // Disjoint -> wdot 0 -> 0 even with bogus Ia/Ib.
    CHECK(info_weighted_jaccard<float>(a.data(), a.size(), 999.0f, d.data(),
                                       d.size(), 999.0f, kW.data()) == 0.0);
  }
}

TEST_CASE("info_weighted_jaccard: integer weight path") {
  // Quantized-multiplicity inference path (§7.2): integer W, ratio double.
  const std::vector<std::uint32_t> a = {0, 1, 2};
  const std::vector<std::uint32_t> b = {1, 2, 3};
  const std::vector<std::uint32_t> w = {5, 3, 2, 7};  // integer multiplicities
  // wdot = w[1]+w[2] = 5; Ia = 10, Ib = 12; denom = 17 -> 5/17.
  CHECK(info_weighted_jaccard<std::uint32_t>(a.data(), a.size(), b.data(),
                                             b.size(), w.data()) ==
        doctest::Approx(5.0 / 17.0));
}

TEST_CASE("surprisal") {
  SUBCASE("p=1 -> 0") { CHECK(surprisal<double>(1.0) == doctest::Approx(0.0)); }
  SUBCASE("p=0.5 -> log(2)") {
    CHECK(surprisal<double>(0.5) == doctest::Approx(std::log(2.0)));
  }
  SUBCASE("p=1/e -> 1") {
    CHECK(surprisal<double>(1.0 / std::exp(1.0)) == doctest::Approx(1.0));
  }
  SUBCASE("p=0 -> +infinity") {
    CHECK(std::isinf(surprisal<double>(0.0)));
    CHECK(surprisal<double>(0.0) > 0.0);
  }
  SUBCASE("negative p -> +infinity (defensive)") {
    CHECK(std::isinf(surprisal<double>(-0.5)));
  }
  SUBCASE("monotonically decreasing in p over (0, 1]") {
    CHECK(surprisal<double>(0.01) > surprisal<double>(0.1));
    CHECK(surprisal<double>(0.1) > surprisal<double>(0.5));
    CHECK(surprisal<double>(0.5) > surprisal<double>(1.0));
  }
}

TEST_CASE("rescale_weights_to_max") {
  SUBCASE("max becomes 1.0; ratios preserved") {
    std::vector<float> w = {1.0f, 2.0f, 4.0f};
    rescale_weights_to_max(w);
    CHECK(w[2] == doctest::Approx(1.0f));
    CHECK(w[0] == doctest::Approx(0.25f));
    CHECK(w[1] == doctest::Approx(0.5f));
  }
  SUBCASE("idempotent (already-rescaled stays the same)") {
    std::vector<float> w = {0.25f, 0.5f, 1.0f};
    rescale_weights_to_max(w);
    CHECK(w[0] == doctest::Approx(0.25f));
    CHECK(w[2] == doctest::Approx(1.0f));
  }
  SUBCASE("all-zero weights left untouched") {
    std::vector<float> w = {0.0f, 0.0f, 0.0f};
    rescale_weights_to_max(w);
    CHECK(w[0] == 0.0f);
    CHECK(w[1] == 0.0f);
    CHECK(w[2] == 0.0f);
  }
  SUBCASE("empty container is no-op") {
    std::vector<float> w;
    rescale_weights_to_max(w);
    CHECK(w.empty());
  }
  SUBCASE("zero-mixed-in vector keeps zeros at zero") {
    std::vector<float> w = {0.0f, 2.0f, 0.0f, 4.0f};
    rescale_weights_to_max(w);
    CHECK(w[0] == 0.0f);
    CHECK(w[2] == 0.0f);
    CHECK(w[3] == doctest::Approx(1.0f));
  }
  SUBCASE("preserves jaccard (composes scale invariance + rescale)") {
    const std::vector<std::uint32_t> a = {0, 1, 2, 3};
    const std::vector<std::uint32_t> b = {2, 3, 4, 5};
    std::vector<float> w = kW;
    const double j_orig = info_weighted_jaccard<float>(
        a.data(), a.size(), b.data(), b.size(), w.data());
    rescale_weights_to_max(w);
    const double j_rescaled = info_weighted_jaccard<float>(
        a.data(), a.size(), b.data(), b.size(), w.data());
    CHECK(j_orig == doctest::Approx(j_rescaled));
  }
}
