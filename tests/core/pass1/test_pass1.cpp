// ============================================================================
// tests/core/pass1/test_pass1.cpp
//
// Pass1Learner core (Step 3b): seed → stream → refresh recovers atoms that
// cover structured data. Checks the D-cap invariant, utilization, surprisal-
// weighted reconstruction recall, and determinism.
// ============================================================================

#include "pass1.hpp"

#include "frequent_directions.hpp"

#include <cstdint>
#include <vector>

#include "doctest.h"

using binarycore::binary_vec::BigSparseBinaryVecDynamic;
using core::pass1::Pass1Learner;
using core::pass1::Signature;
using core::sketch::FrequentDirections;

namespace {

constexpr std::size_t kDim = 200;
constexpr std::size_t kNumPatterns = 40;

// 40 disjoint 4-bit patterns: pattern p occupies bits {5p .. 5p+3}.
std::vector<Signature> make_patterns() {
  std::vector<Signature> pats;
  for (std::uint32_t p = 0; p < kNumPatterns; ++p)
    pats.push_back({5 * p, 5 * p + 1, 5 * p + 2, 5 * p + 3});
  return pats;
}

BigSparseBinaryVecDynamic to_vec(const Signature& sig) {
  BigSparseBinaryVecDynamic v(kDim);
  for (std::uint32_t g : sig)
    v.chunks[0].data.push_back(static_cast<std::uint16_t>(g));
  return v;
}

}  // namespace

TEST_CASE("Pass1Learner: seed+stream recovers covering atoms") {
  const auto pats = make_patterns();
  const std::vector<std::uint32_t> weights(kDim, 3);  // uniform

  // FD over the distinct patterns (for leverage seeding).
  FrequentDirections fd(kDim, 64);
  for (const auto& p : pats) fd.add(to_vec(p));
  fd.finalize();

  Pass1Learner::Config cfg;
  cfg.K = 32;
  cfg.D = 8;
  cfg.refresh_every = 100;
  Pass1Learner learner(kDim, weights, cfg);
  learner.seed(fd, pats);  // warmup = the distinct patterns

  for (int rep = 0; rep < 30; ++rep)
    for (const auto& p : pats) learner.observe(p);
  learner.finalize();

  const auto& cb = learner.codebook();

  SUBCASE("D-cap invariant holds for every atom") {
    for (std::size_t k = 0; k < cb.size(); ++k)
      CHECK(cb.atom(k).size() <= cfg.D);
  }
  SUBCASE("utilization is high (most seeded atoms fire)") {
    CHECK(learner.utilization() > 0.8);
  }
  SUBCASE("reconstruction recall is high on the training patterns") {
    // K=32 atoms can cover 32 of the 40 disjoint patterns → recall ≳ 0.8.
    CHECK(learner.mean_recall(pats) > 0.6);
  }
}

TEST_CASE("Pass1Learner: deterministic under a fixed seed") {
  const auto pats = make_patterns();
  const std::vector<std::uint32_t> weights(kDim, 3);
  FrequentDirections fd(kDim, 64);
  for (const auto& p : pats) fd.add(to_vec(p));
  fd.finalize();

  auto run = [&] {
    Pass1Learner::Config cfg;
    cfg.K = 32;
    cfg.D = 8;
    cfg.refresh_every = 100;
    cfg.rng_seed = 777;
    Pass1Learner l(kDim, weights, cfg);
    l.seed(fd, pats);
    for (int rep = 0; rep < 10; ++rep)
      for (const auto& p : pats) l.observe(p);
    l.finalize();
    std::vector<core::codebook::Atom> atoms;
    for (std::size_t k = 0; k < l.codebook().size(); ++k)
      atoms.push_back(l.codebook().atom(k));
    return atoms;
  };
  CHECK(run() == run());
}
