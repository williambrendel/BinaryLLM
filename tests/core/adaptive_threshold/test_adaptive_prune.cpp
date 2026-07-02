// ============================================================================
// tests/core/adaptive_threshold/test_adaptive_prune.cpp
//
// Tests for the four utilities and the canonical composite
// adaptive_prune in adaptive_threshold.hpp.
//
// Coverage:
//   - elbow_effective_count: trivial inputs, curved distributions,
//     cliffs, significance gate behavior, noise filtering
//   - scaled_mm_effective_count: linear slopes (slope-aware), uniform,
//     all-zero, saturation point
//   - adaptive_prune_count: full cascade — every test from the elbow
//     prune patch, plus the linear-fallback cases that exercise scaled_mm
//   - adaptive_prune_threshold: value-form correctness
// ============================================================================

#include "adaptive_threshold.hpp"
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

using core::adaptive_threshold::adaptive_prune_count;
using core::adaptive_threshold::adaptive_prune_threshold;
using core::adaptive_threshold::elbow_effective_count;
using core::adaptive_threshold::kElbowSig;
using core::adaptive_threshold::kMMFactor;
using core::adaptive_threshold::scaled_mm_effective_count;

namespace {

std::vector<double> linear_descending(std::size_t n, double max, double slope) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = max - static_cast<double>(i) * slope;
  }
  return v;
}

std::vector<double> uniform_plus_cliff(
    std::size_t n, std::size_t cliff_at, double high, double low) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < cliff_at; ++i) v[i] = high;
  for (std::size_t i = cliff_at; i < n; ++i) v[i] = low;
  return v;
}

}  // namespace

// =============================================================================
// elbow_effective_count — utility tests
// =============================================================================

TEST_CASE("elbow util: empty / trivial return n") {
  std::vector<double> v0{};
  CHECK(elbow_effective_count(v0) == 0);
  std::vector<double> v1{42.0};
  CHECK(elbow_effective_count(v1) == 1);
  std::vector<double> v2{10.0, 1.0};
  CHECK(elbow_effective_count(v2) == 2);
}

TEST_CASE("elbow util: flat input returns n (no elbow)") {
  std::vector<double> v(2048, 5.0);
  CHECK(elbow_effective_count(v) == 2048);
}

TEST_CASE("elbow util: linear input returns n (no significant elbow)") {
  auto v = linear_descending(2048, 2048.0, 1.0);
  // Linear has zero curvature; chord-distance is FP noise. Sig gate
  // rejects it -> returns n to signal "no elbow".
  CHECK(elbow_effective_count(v) == 2048);
}

TEST_CASE("elbow util: cliff at 1900 returns 1900") {
  auto v = uniform_plus_cliff(2048, 1900, 100.0, 1.0);
  CHECK(elbow_effective_count(v) == 1900);
}

TEST_CASE("elbow util: 1/sqrt returns the natural elbow") {
  std::vector<double> v(2048);
  for (int i = 0; i < int(v.size()); ++i) v[i] = 1.0 / std::sqrt(double(i + 1));
  CHECK(elbow_effective_count(v) == 102);
}

TEST_CASE("elbow util: peak returns 1") {
  std::vector<double> v(2048, 1.0);
  v[0] = 1000.0;
  CHECK(elbow_effective_count(v) == 1);
}

TEST_CASE("elbow util: uniform + tiny noise is rejected (no significant elbow)") {
  std::vector<double> v(2048, 100.0);
  for (int i = 0; i < int(v.size()); ++i) v[i] += (i % 3 - 1) * 0.01;
  std::sort(v.begin(), v.end(), std::greater<double>());
  // sig=1% of median (≈100) = 1.0; noise diffs are ~0.007 — well below.
  CHECK(elbow_effective_count(v) == 2048);
}

// =============================================================================
// scaled_mm_effective_count — utility tests
// =============================================================================

TEST_CASE("scaled_mm: empty returns 0") {
  std::vector<double> v{};
  CHECK(scaled_mm_effective_count(v) == 0);
}

TEST_CASE("scaled_mm: all-zero returns n") {
  std::vector<double> v(100, 0.0);
  CHECK(scaled_mm_effective_count(v) == 100);
}

TEST_CASE("scaled_mm: uniform saturates to n") {
  std::vector<double> v(2048, 5.0);
  CHECK(scaled_mm_effective_count(v) == 2048);
}

TEST_CASE("scaled_mm: linear slope=1 returns ~1320") {
  auto v = linear_descending(2048, 2048.0, 1.0);
  // MM ~= 600 for v[i] = 2048-i; factor 2.2 -> 1320.
  CHECK(scaled_mm_effective_count(v) == 1320);
}

TEST_CASE("scaled_mm: linear slope=0.5 returns ~1888") {
  auto v = linear_descending(2048, 2048.0, 0.5);
  CHECK(scaled_mm_effective_count(v) == 1888);
}

TEST_CASE("scaled_mm: shallow linear saturates to n") {
  auto v = linear_descending(2048, 2048.0, 0.1);
  CHECK(scaled_mm_effective_count(v) == 2048);
}

TEST_CASE("scaled_mm: factor 2.0 vs 2.2 cuts differently") {
  auto v = linear_descending(2048, 2048.0, 1.0);
  CHECK(scaled_mm_effective_count(v, 2.0) == 1200);
  CHECK(scaled_mm_effective_count(v, 2.2) == 1320);
}

// =============================================================================
// adaptive_prune_count — composite cascade
// =============================================================================

TEST_CASE("adaptive: trivial inputs") {
  std::vector<double> v0{};
  CHECK(adaptive_prune_count(v0) == 0);
  std::vector<double> v1{42.0};
  CHECK(adaptive_prune_count(v1) == 1);
  std::vector<double> v2{10.0, 1.0};
  CHECK(adaptive_prune_count(v2) == 2);
}

TEST_CASE("adaptive: flat input returns n") {
  std::vector<double> v(2048, 5.0);
  CHECK(adaptive_prune_count(v) == 2048);
}

TEST_CASE("adaptive: all-zero returns n") {
  std::vector<double> v(100, 0.0);
  CHECK(adaptive_prune_count(v) == 100);
}

// --- linear: elbow rejects, scaled_mm takes over ---

TEST_CASE("adaptive: linear slope=1 -> scaled_mm path") {
  auto v = linear_descending(2048, 2048.0, 1.0);
  CHECK(adaptive_prune_count(v) == 1320);
}

TEST_CASE("adaptive: linear slope=0.5 -> larger k") {
  auto v = linear_descending(2048, 2048.0, 0.5);
  CHECK(adaptive_prune_count(v) == 1888);
}

TEST_CASE("adaptive: shallow linear saturates to n") {
  auto v = linear_descending(2048, 2048.0, 0.1);
  CHECK(adaptive_prune_count(v) == 2048);
}

TEST_CASE("adaptive: linear N=100 slope=1 -> ~66") {
  auto v = linear_descending(100, 100.0, 1.0);
  CHECK(adaptive_prune_count(v) == 66);
}

// --- noise: median-based sig filters false positives ---

TEST_CASE("adaptive: uniform + tiny noise -> no cut") {
  std::vector<double> v(2048, 100.0);
  for (int i = 0; i < int(v.size()); ++i) v[i] += (i % 3 - 1) * 0.01;
  std::sort(v.begin(), v.end(), std::greater<double>());
  CHECK(adaptive_prune_count(v) == 2048);
}

TEST_CASE("adaptive: uniform + small noise -> no cut") {
  std::vector<double> v(2048, 100.0);
  for (int i = 0; i < int(v.size()); ++i) v[i] += (i % 7 - 3) * 0.1;
  std::sort(v.begin(), v.end(), std::greater<double>());
  CHECK(adaptive_prune_count(v) == 2048);
}

TEST_CASE("adaptive: uniform + moderate noise -> no cut") {
  std::vector<double> v(2048, 100.0);
  for (int i = 0; i < int(v.size()); ++i) v[i] += (i % 11 - 5) * 1.0;
  std::sort(v.begin(), v.end(), std::greater<double>());
  CHECK(adaptive_prune_count(v) == 2048);
}

// --- curved: elbow path ---

TEST_CASE("adaptive: 1/sqrt(i+1) -> 102") {
  std::vector<double> v(2048);
  for (int i = 0; i < int(v.size()); ++i) v[i] = 1.0 / std::sqrt(double(i + 1));
  CHECK(adaptive_prune_count(v) == 102);
}

TEST_CASE("adaptive: Zipf 1/(i+1) -> 44") {
  std::vector<double> v(2048);
  for (int i = 0; i < int(v.size()); ++i) v[i] = 1.0 / double(i + 1);
  CHECK(adaptive_prune_count(v) == 44);
}

TEST_CASE("adaptive: exp(-0.01*i) -> 302") {
  std::vector<double> v(2048);
  for (int i = 0; i < int(v.size()); ++i) v[i] = std::exp(-double(i) * 0.01);
  CHECK(adaptive_prune_count(v) == 302);
}

TEST_CASE("adaptive: 1/(i+1)^2 sharp power law") {
  std::vector<double> v(2048);
  for (int i = 0; i < int(v.size()); ++i)
    v[i] = 1.0 / std::pow(double(i + 1), 2.0);
  std::size_t k = adaptive_prune_count(v);
  CHECK(k < 20);
  CHECK(k >= 1);
}

// --- cliffs ---

TEST_CASE("adaptive: uniform + cliff at 1900") {
  auto v = uniform_plus_cliff(2048, 1900, 100.0, 1.0);
  CHECK(adaptive_prune_count(v) == 1900);
}

TEST_CASE("adaptive: uniform + cliff at 1500") {
  auto v = uniform_plus_cliff(2048, 1500, 100.0, 1.0);
  CHECK(adaptive_prune_count(v) == 1500);
}

TEST_CASE("adaptive: uniform + cliff at 1000") {
  auto v = uniform_plus_cliff(2048, 1000, 100.0, 1.0);
  CHECK(adaptive_prune_count(v) == 1000);
}

TEST_CASE("adaptive: uniform + cliff at 500 (median in floor)") {
  auto v = uniform_plus_cliff(2048, 500, 100.0, 1.0);
  CHECK(adaptive_prune_count(v) == 500);
}

TEST_CASE("adaptive: uniform + cliff at 100 (early)") {
  auto v = uniform_plus_cliff(2048, 100, 100.0, 1.0);
  CHECK(adaptive_prune_count(v) == 100);
}

TEST_CASE("adaptive: cliff at 30 (N=100)") {
  auto v = uniform_plus_cliff(100, 30, 100.0, 1.0);
  CHECK(adaptive_prune_count(v) == 30);
}

TEST_CASE("adaptive: linear decay + cliff at 1800 catches the cliff") {
  std::vector<double> v(2048);
  for (int i = 0; i < 1800; ++i) v[i] = 2048.0 - double(i);
  for (int i = 1800; i < 2048; ++i) v[i] = 1.0;
  CHECK(adaptive_prune_count(v) == 1800);
}

// --- peak / multi-level ---

TEST_CASE("adaptive: single peak returns 1") {
  std::vector<double> v(2048, 1.0);
  v[0] = 1000.0;
  CHECK(adaptive_prune_count(v) == 1);
}

TEST_CASE("adaptive: 3 levels picks deepest cliff") {
  std::vector<double> v(2048);
  for (int i = 0; i < 30; ++i) v[i] = 100.0;
  for (int i = 30; i < 1000; ++i) v[i] = 50.0;
  for (int i = 1000; i < 2048; ++i) v[i] = 1.0;
  CHECK(adaptive_prune_count(v) == 1000);
}

// --- parameter sensitivity ---

TEST_CASE("adaptive: factor only affects fallback path") {
  std::vector<double> v(2048);
  for (int i = 0; i < int(v.size()); ++i) v[i] = 1.0 / std::sqrt(double(i + 1));
  // Curved -> elbow path -> factor unused.
  CHECK(adaptive_prune_count(v, kElbowSig, 2.0) ==
        adaptive_prune_count(v, kElbowSig, 2.2));
  CHECK(adaptive_prune_count(v, kElbowSig, 2.2) ==
        adaptive_prune_count(v, kElbowSig, 5.0));
}

TEST_CASE("adaptive: factor affects linear fallback") {
  auto v = linear_descending(2048, 2048.0, 1.0);
  CHECK(adaptive_prune_count(v, kElbowSig, 2.0) == 1200);
  CHECK(adaptive_prune_count(v, kElbowSig, 2.2) == 1320);
  CHECK(adaptive_prune_count(v, kElbowSig, 2.5) == 1500);
}

// =============================================================================
// adaptive_prune_threshold
// =============================================================================

TEST_CASE("threshold: returns first dropped value at cliff") {
  auto v = uniform_plus_cliff(2048, 1900, 100.0, 1.0);
  CHECK(adaptive_prune_threshold(v) == 1.0);
}

TEST_CASE("threshold: returns 0.0 when no cut") {
  std::vector<double> v(2048, 5.0);
  CHECK(adaptive_prune_threshold(v) == 0.0);
}

TEST_CASE("threshold: peak returns the rest value") {
  std::vector<double> v(2048, 1.0);
  v[0] = 1000.0;
  CHECK(adaptive_prune_threshold(v) == 1.0);
}

TEST_CASE("threshold: empty returns 0.0") {
  std::vector<double> v{};
  CHECK(adaptive_prune_threshold(v) == 0.0);
}
