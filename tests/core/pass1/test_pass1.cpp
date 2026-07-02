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

namespace {

// Total pairwise mask overlap Σ_{j<k} popcount(φ_j ∩ φ_k) across the codebook.
long total_overlap(const core::codebook::Codebook& cb) {
  long sum = 0;
  for (std::size_t j = 0; j < cb.size(); ++j)
    for (std::size_t k = j + 1; k < cb.size(); ++k) {
      const auto& a = cb.atom(j);
      const auto& b = cb.atom(k);
      std::size_t ia = 0, ib = 0;
      while (ia < a.size() && ib < b.size()) {
        if (a[ia] < b[ib]) ++ia;
        else if (b[ib] < a[ia]) ++ib;
        else { ++sum; ++ia; ++ib; }
      }
    }
  return sum;
}

// 20 patterns sharing a common hub {0,1,2,3} plus 3 distinct bits each.
std::vector<Signature> make_hub_patterns() {
  std::vector<Signature> pats;
  for (std::uint32_t p = 0; p < 20; ++p)
    pats.push_back({0, 1, 2, 3, 10u + p, 40u + p, 70u + p});
  return pats;
}

core::codebook::Codebook learn_hub(double lambda) {
  const auto pats = make_hub_patterns();
  const std::vector<std::uint32_t> weights(kDim, 3);
  FrequentDirections fd(kDim, 32);
  for (const auto& p : pats) fd.add(to_vec(p));
  fd.finalize();

  Pass1Learner::Config cfg;
  cfg.K = 20;
  cfg.D = 8;
  cfg.refresh_every = 100000;  // no mid-stream refresh → finalize sees full counts
  cfg.decay = 1.0;
  cfg.lambda = lambda;
  Pass1Learner l(kDim, weights, cfg);
  l.seed(fd, pats);
  for (int rep = 0; rep < 30; ++rep)
    for (const auto& p : pats) l.observe(p);
  l.finalize();
  return l.codebook();
}

}  // namespace

TEST_CASE("Pass1Learner: incoherence penalty lowers pairwise overlap") {
  // Without separation, every atom keeps the shared hub → high overlap.
  const long overlap_off = total_overlap(learn_hub(0.0));
  // With a strong penalty, the congested hub bits get spread out / dropped.
  const long overlap_on = total_overlap(learn_hub(2.0));
  CHECK(overlap_off > 0);
  CHECK(overlap_on < overlap_off / 2);  // substantially lower-degree overlap
}

namespace {

std::size_t atoms_with_bit(const core::codebook::Codebook& cb,
                           std::uint32_t bit) {
  std::size_t n = 0;
  for (std::size_t k = 0; k < cb.size(); ++k) {
    const auto& a = cb.atom(k);
    if (std::find(a.begin(), a.end(), bit) != a.end()) ++n;
  }
  return n;
}

// 20 patterns that all share the hub bit 0 (in every signature → dominant FD
// direction), plus two distinct bits each.
std::vector<Signature> make_hubbed() {
  std::vector<Signature> pats;
  for (std::uint32_t p = 0; p < 20; ++p) pats.push_back({0, 10u + p, 30u + p});
  return pats;
}

core::codebook::Codebook learn_hubbed(double strip_fraction) {
  const auto pats = make_hubbed();
  const std::vector<std::uint32_t> weights(kDim, 3);
  FrequentDirections fd(kDim, 32);
  for (const auto& p : pats) fd.add(to_vec(p));
  fd.finalize();

  Pass1Learner::Config cfg;
  cfg.K = 20;
  cfg.D = 8;
  cfg.refresh_every = 100;
  cfg.lambda = 0.0;  // isolate hub-stripping from the incoherence penalty
  Pass1Learner l(kDim, weights, cfg);
  if (strip_fraction > 0.0) l.strip_hubs(fd, strip_fraction);
  l.seed(fd, pats);
  for (int rep = 0; rep < 30; ++rep)
    for (const auto& p : pats) l.observe(p);
  l.finalize();
  return l.codebook();
}

}  // namespace

TEST_CASE("Pass1Learner: hub-stripping removes the diffuse hub bit") {
  // Without stripping (and with separation off) the hub bit is in atoms.
  CHECK(atoms_with_bit(learn_hubbed(0.0), 0) > 0);
  // Stripping the most-promiscuous bits excludes the hub from every codeword.
  CHECK(atoms_with_bit(learn_hubbed(0.05), 0) == 0);
}

TEST_CASE("Pass1Learner: re-seed revives dead codewords, raising recall") {
  const auto pats = make_patterns();  // 40 disjoint patterns
  const std::vector<std::uint32_t> weights(kDim, 3);
  FrequentDirections fd(kDim, 64);
  for (const auto& p : pats) fd.add(to_vec(p));
  fd.finalize();

  auto train = [&](std::size_t reseed_cap) {
    Pass1Learner::Config cfg;
    cfg.K = 40;
    cfg.D = 8;
    cfg.refresh_every = 40;
    cfg.reseed_pool_cap = reseed_cap;  // 0 disables re-seed
    Pass1Learner l(kDim, weights, cfg);
    // Seed only the first 28 patterns → atoms 28..39 start dead.
    std::vector<Signature> partial(pats.begin(), pats.begin() + 28);
    l.seed(fd, partial);
    for (int rep = 0; rep < 40; ++rep)
      for (const auto& p : pats) l.observe(p);  // stream ALL 40
    l.finalize();
    return l.mean_recall(pats);
  };

  const double recall_no_reseed = train(0);
  const double recall_reseed = train(4096);
  CHECK(recall_reseed > recall_no_reseed + 0.1);
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
