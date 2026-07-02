// ============================================================================
// tests/core/adaptive_threshold/test_thresholds.cpp
//
// Doctest coverage for the utility-level threshold primitives. The
// composite cascade (adaptive_prune_count) is covered in detail in
// test_adaptive_prune.cpp; this file owns the per-utility behavior
// (entropy_effective_count, gap_ratio_effective_count) plus a few
// composite spot checks.
// ============================================================================

#include "adaptive_threshold.hpp"
#include "doctest.h"

#include <vector>

using core::adaptive_threshold::adaptive_prune_count;
using core::adaptive_threshold::adaptive_prune_threshold;
using core::adaptive_threshold::entropy_effective_count;
using core::adaptive_threshold::gap_ratio_effective_count;
using core::adaptive_threshold::kNoMaxCutIndex;

// ─────────────────────────────────────────────────────────────────────────────
// entropy_effective_count
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("entropy: empty input returns 0") {
  std::vector<double> empty_input;
  CHECK(entropy_effective_count(empty_input) == 0);
}

TEST_CASE("entropy: all zeros returns 0") {
  std::vector<double> v = {0.0, 0.0, 0.0};
  CHECK(entropy_effective_count(v) == 0);
}

TEST_CASE("entropy: single positive value returns 1") {
  std::vector<double> v = {0.9};
  CHECK(entropy_effective_count(v) == 1);
}

TEST_CASE("entropy: true delta returns 1") {
  // Only one positive value → l=1 → must return 1.
  std::vector<double> v = {0.999, 0.0, 0.0};
  CHECK(entropy_effective_count(v) == 1);
}

TEST_CASE("entropy: near-delta with tiny tail still returns >= 2 (ceil)") {
  // Any positive tail pushes exp(H) strictly above 1; ceil rounds up.
  // This is intentional: we don't want to silently drop the tail just
  // because the head dominates.
  std::vector<double> v = {0.999, 0.0001, 0.0001};
  CHECK(entropy_effective_count(v) == 2);
}

TEST_CASE("entropy: uniform over k items returns k") {
  // Uniform: H = log(k), exp(H) = k exactly.
  std::vector<double> v = {0.25, 0.25, 0.25, 0.25};
  CHECK(entropy_effective_count(v) == 4);
}

TEST_CASE("entropy: trailing non-positive values truncate the scan") {
  std::vector<double> v = {0.5, 0.5, 0.0, 0.0, 0.0};
  // Effective input is just {0.5, 0.5} → uniform over 2 → returns 2.
  CHECK(entropy_effective_count(v) == 2);
}

TEST_CASE("entropy: max_cut_index caps the scan") {
  std::vector<double> v = {0.25, 0.25, 0.25, 0.25};
  // Cap at 2 → only the first two contribute. Their sum normalizes to
  // a uniform pair. exp(H) = 2 → return 2.
  CHECK(entropy_effective_count(v, 2) == 2);
}

TEST_CASE("entropy: peaky distribution gives small k") {
  // 0.8 / 0.05 / 0.05 / 0.05 / 0.05 — sums to 1, dominated by first.
  std::vector<double> v = {0.8, 0.05, 0.05, 0.05, 0.05};
  // H ≈ log(1) + (-0.8*log(0.8) - 4*0.05*log(0.05)) / 1
  //   ≈ 0 + (0.1785 + 0.5994) ≈ 0.778
  // exp(H) ≈ 2.18 → ceil → 3.
  CHECK(entropy_effective_count(v) == 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// gap_ratio_effective_count
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ratio: empty input returns 0") {
  std::vector<double> empty_input;
  CHECK(gap_ratio_effective_count(empty_input) == 0);
}

TEST_CASE("ratio: single element returns 1") {
  std::vector<double> v = {0.9};
  CHECK(gap_ratio_effective_count(v) == 1);
}

TEST_CASE("ratio: plateau then cliff cuts at the cliff (min_gap=3)") {
  // ratios: 1.037, 1.025, 4.0, 1.333
  // The 4.0 cliff between indices 2 and 3 clears min_gap=3.
  std::vector<double> v = {0.85, 0.82, 0.80, 0.20, 0.15};
  CHECK(gap_ratio_effective_count(v, 3.0) == 3);
}

TEST_CASE("ratio: one dominant score keeps only it") {
  std::vector<double> v = {0.9, 0.01, 0.01, 0.01};
  // 0.9 / 0.01 = 90 >> 3 → cut at 1.
  CHECK(gap_ratio_effective_count(v, 3.0) == 1);
}

TEST_CASE("ratio: cliff exactly at min_gap threshold cuts (>=)") {
  std::vector<double> v = {0.9, 0.3, 0.3, 0.3};
  // 0.9 / 0.3 = 3.0 exactly equals min_gap.
  CHECK(gap_ratio_effective_count(v, 3.0) == 1);
}

TEST_CASE("ratio: smooth descent gives no cut, returns n") {
  // ratios all ≈ 1.06, well below min_gap=3.
  std::vector<double> v = {0.85, 0.80, 0.75, 0.70};
  CHECK(gap_ratio_effective_count(v, 3.0) == 4);
}

TEST_CASE("ratio: multiple cliffs — steepest qualifying wins") {
  // ratios: 3.6, 1.25, 10.0. min_gap=3, so 3.6 qualifies (cut at 1)
  // and 10.0 qualifies more steeply (cut at 3). Final: 3.
  std::vector<double> v = {0.90, 0.25, 0.20, 0.02};
  CHECK(gap_ratio_effective_count(v, 3.0) == 3);
}

TEST_CASE("ratio: tied cliffs — earlier wins (strict > on max_gap)") {
  // ratios: 2.0, 1.0, 2.0. Two equal cliffs at i=1 and i=3.
  // Strict > means the second 2.0 does NOT overwrite cut_idx=1.
  std::vector<double> v = {1.0, 0.5, 0.5, 0.25};
  CHECK(gap_ratio_effective_count(v, 2.0) == 1);
}

TEST_CASE("ratio: cosine-tuned min_gap=1.5 catches moderate cliffs") {
  // Top score 0.866, second 0.540 → ratio 1.60, clears 1.5.
  std::vector<double> v = {0.866, 0.540, 0.530, 0.520};
  CHECK(gap_ratio_effective_count(v, 1.5) == 1);
}

TEST_CASE("ratio: noise-floor stops the scan") {
  // Loop iterates with cap default = n = 3.
  //   i=1: ac=0.85 > eps, OK. ap=0.85, ac=sorted_desc[1]=0.80. g=1.0625.
  //   i=2: ac=0.80 > eps, OK. ap=0.80, ac=sorted_desc[2]=1e-12. g=8e11.
  //         8e11 > max_gap(1.0625) → max_gap=8e11, qualifies (>= 3) → cut=2.
  //   Loop exits.
  //   Final: min(cut=2, n=3) = 2.
  std::vector<double> v = {0.85, 0.80, 1e-12};
  CHECK(gap_ratio_effective_count(v, 3.0) == 2);
}

TEST_CASE("ratio: max_cut_index caps the scan") {
  std::vector<double> v = {0.9, 0.3, 0.1, 0.05, 0.01};
  // Cap at 2 → only ratio 0.9/0.3=3.0 is considered. Cut at 1.
  // Beyond that, 0.3/0.1=3.0 would also qualify (and tie), but capped.
  CHECK(gap_ratio_effective_count(v, 3.0, 1e-10, 2) == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// adaptive_prune_count — composite spot checks
// (full coverage in test_adaptive_prune.cpp)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("adaptive: empty input returns 0") {
  std::vector<double> empty_input;
  CHECK(adaptive_prune_count(empty_input) == 0);
}

TEST_CASE("adaptive: sharp cliff dominates") {
  // Peaky: elbow should fire immediately.
  std::vector<double> v = {0.9, 0.01, 0.01, 0.01};
  CHECK(adaptive_prune_count(v) == 1);
}

TEST_CASE("adaptive: all-positive smooth — no cut") {
  // Near-uniform with shallow slope — no significant elbow,
  // scaled_mm fallback returns n (factor saturates).
  std::vector<double> v = {1.0, 0.99, 0.98, 0.97};
  CHECK(adaptive_prune_count(v) == 4);
}

// ─────────────────────────────────────────────────────────────────────────────
// adaptive_prune_threshold
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("threshold: empty input returns 0.0 (no-cut sentinel)") {
  std::vector<double> empty_input;
  CHECK(adaptive_prune_threshold(empty_input) == 0.0);
}

TEST_CASE("threshold: no cut returns 0.0 (filter 'value > 0' passes positives)") {
  std::vector<double> v = {1.0, 0.99, 0.98, 0.97};
  // Smooth, no cut, k = n = 4 → 0.0.
  CHECK(adaptive_prune_threshold(v) == 0.0);
}

TEST_CASE("threshold: returns first dropped value") {
  std::vector<double> v = {0.9, 0.01, 0.01, 0.01};
  // adaptive_prune_count returns 1, so threshold = v[1] = 0.01.
  // Anything strictly > 0.01 survives the cut.
  CHECK(adaptive_prune_threshold(v) == doctest::Approx(0.01));
}
