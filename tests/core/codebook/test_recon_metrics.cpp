// ============================================================================
// tests/core/codebook/test_recon_metrics.cpp
//
// Surprisal-weighted, block-aware reconstruction BER (spec §7.5.2): exact
// per-band recall/precision on a controlled fixture, and the identity-vs-
// context expectation (BER_C recall ≥ BER_L/R).
// ============================================================================

#include "recon_metrics.hpp"

#include <cstdint>
#include <vector>

#include "doctest.h"

using core::codebook::recon_metrics;

TEST_CASE("recon_metrics: exact per-band recall/precision") {
  // F=4 → L=[0,4) C=[4,8) R=[8,12). C bits weigh 10 (identity), L/R weigh 1.
  const std::uint32_t F = 4;
  const std::vector<std::uint32_t> w = {1, 1, 0, 0, 10, 10, 0, 0, 1, 1, 0, 0};
  const std::vector<std::uint32_t> truth = {0, 1, 4, 5, 8, 9};
  const std::vector<std::uint32_t> recon = {0, 4, 5};  // L partial, C full, R none

  const auto m = recon_metrics(recon, truth, w, F);

  CHECK(m.recall_C == doctest::Approx(1.0));      // identity recovered
  CHECK(m.precision_C == doctest::Approx(1.0));
  CHECK(m.recall_L == doctest::Approx(0.5));      // one of two L bits
  CHECK(m.precision_L == doctest::Approx(1.0));
  CHECK(m.recall_R == doctest::Approx(0.0));      // R missed
  CHECK(m.precision_R == doctest::Approx(1.0));   // but no phantom R bits
  CHECK(m.recall == doctest::Approx(21.0 / 24.0));
  CHECK(m.precision == doctest::Approx(1.0));
  CHECK(m.fn() == doctest::Approx(1.0 - 21.0 / 24.0));
  CHECK(m.fp() == doctest::Approx(0.0));
}

TEST_CASE("recon_metrics: C reconstructs better than context (identity guarantee)") {
  const std::uint32_t F = 4;
  const std::vector<std::uint32_t> w = {1, 1, 0, 0, 10, 10, 0, 0, 1, 1, 0, 0};
  const std::vector<std::uint32_t> truth = {0, 1, 4, 5, 8, 9};
  const std::vector<std::uint32_t> recon = {0, 4, 5};
  const auto m = recon_metrics(recon, truth, w, F);
  CHECK(m.recall_C > m.recall_L);
  CHECK(m.recall_C > m.recall_R);
}

TEST_CASE("recon_metrics: perfect decode → all recall/precision 1") {
  const std::uint32_t F = 4;
  const std::vector<std::uint32_t> w(12, 2);
  const std::vector<std::uint32_t> truth = {1, 5, 9};
  const auto m = recon_metrics(truth, truth, w, F);  // recon == truth
  CHECK(m.recall == doctest::Approx(1.0));
  CHECK(m.precision == doctest::Approx(1.0));
  CHECK(m.recall_C == doctest::Approx(1.0));
}
