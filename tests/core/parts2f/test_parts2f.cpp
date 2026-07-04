// ============================================================================
// tests/core/parts2f/test_parts2f.cpp — Pass-1 context-part discovery (§14).
//
// Note on synthetic structure: growth scores a candidate bit by
//   cs_i · W_e,   cs_i = log(|S| / count_in_S),
// so a bit present in ALL survivors has cs=0 and is not added — the algorithm
// bundles *partially* co-occurring bits (nested support), not perfect cliques.
// Fixtures below use nested/partial support accordingly.
// ============================================================================

#include "doctest.h"
#include "parts2f.hpp"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

using namespace core::parts2f;

namespace {

Dataset make_ds(std::vector<std::vector<std::uint32_t>> sigs, std::size_t dim) {
  Dataset ds;
  ds.dim = dim;
  ds.sigs = std::move(sigs);
  ds.build_index();
  ds.w.assign(dim, 0.0);
  const double N = static_cast<double>(ds.sigs.size());
  for (std::size_t e = 0; e < dim; ++e)
    if (ds.c_e[e] > 0) ds.w[e] = std::log(N / static_cast<double>(ds.c_e[e]));
  return ds;
}

// Nested partial support on bits {0,1,2}: bit 0 in `a` sigs, bit 1 in the first
// `b`≤a of them, bit 2 in the first `c`≤b — so cs>0 for 1 and 2 within bit-0's
// survivors. Plus `noise` unique singletons so no bit fills every signature.
Dataset partial_ds(std::size_t a, std::size_t b, std::size_t c, std::size_t noise) {
  std::vector<std::vector<std::uint32_t>> sigs;
  for (std::size_t i = 0; i < a; ++i) {
    std::vector<std::uint32_t> s{0};
    if (i < b) s.push_back(1);
    if (i < c) s.push_back(2);
    sigs.push_back(std::move(s));
  }
  for (std::size_t i = 0; i < noise; ++i) sigs.push_back({static_cast<std::uint32_t>(50 + i)});
  return make_ds(std::move(sigs), 50 + noise + 4);
}

}  // namespace

TEST_CASE("support floor / rollback — no single-signature parts") {
  std::vector<std::vector<std::uint32_t>> sigs;
  for (int i = 0; i < 30; ++i) sigs.push_back({0, 1});
  sigs.push_back({0, 1, 5});  // one lone extra bit
  for (int i = 0; i < 10; ++i) sigs.push_back({static_cast<std::uint32_t>(20 + i)});
  Dataset ds = make_ds(std::move(sigs), 40);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 16; cfg.c_atom = 0;  // grown layer only
  auto parts = build_parts(ds, cfg);
  REQUIRE(!parts.empty());
  for (const auto& p : parts) CHECK(p.support >= cfg.s_min);  // lone signature never isolated
}

TEST_CASE("atomic completion (Fix 2): every uncovered bit gets a support-exempt 1-bit atom") {
  std::vector<std::vector<std::uint32_t>> sigs;
  for (int i = 0; i < 30; ++i) sigs.push_back({0, 1});
  sigs.push_back({0, 1, 5});  // bit 5 occurs once — below the support floor, never grown
  Dataset ds = make_ds(std::move(sigs), 8);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 16; cfg.c_atom = 1;  // atoms on
  auto parts = build_parts(ds, cfg);
  bool atom5 = false;
  for (const auto& p : parts)
    if (p.bits == std::vector<std::uint32_t>{5}) { atom5 = true; CHECK(p.t_p == doctest::Approx(1.0)); }
  CHECK(atom5);  // the rare bit is reconstructable via its atom, despite support 1
}

TEST_CASE("growth bundles partially co-occurring bits into a multi-bit part") {
  Dataset ds = partial_ds(40, 30, 20, 30);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 16;
  auto parts = build_parts(ds, cfg);
  REQUIRE(!parts.empty());
  const auto& p0 = parts.front();
  CHECK(p0.bits.size() >= 2);                 // real multi-bit clique, not a single bit
  bool has0 = false;
  for (std::uint32_t b : p0.bits) { CHECK(b <= 2); if (b == 0) has0 = true; }  // grown from bit-0 anchor
  CHECK(has0);
  CHECK(p0.support >= cfg.s_min);
}

TEST_CASE("universal-bit absorb: a perfectly co-occurring clique bundles into one part") {
  // {0,1,2} appear together in exactly the same 40 signatures (cs=0 for 1,2 within
  // bit-0's survivors) — must still be bundled via the universal-absorb sweep.
  std::vector<std::vector<std::uint32_t>> sigs;
  for (int i = 0; i < 40; ++i) sigs.push_back({0, 1, 2});
  for (int i = 0; i < 40; ++i) sigs.push_back({static_cast<std::uint32_t>(50 + i)});  // keep w>0
  Dataset ds = make_ds(std::move(sigs), 100);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 8;
  auto parts = build_parts(ds, cfg);
  REQUIRE(!parts.empty());
  CHECK(parts.front().bits == std::vector<std::uint32_t>{0, 1, 2});  // whole clique, one part
}

TEST_CASE("t0 prior: closed form 1 − w_last/ic(p), monotone in the last bit's share") {
  Dataset ds = partial_ds(40, 30, 20, 30);
  std::vector<std::uint32_t> p{0, 1, 2};
  const double ic = ds.info(p);
  CHECK(t0_prior(ds, p, ds.w[2]) == doctest::Approx(1.0 - ds.w[2] / ic));
  // the more of the part's info sits in the (miss-able) last bit, the lower t0 —
  // a part that may miss a big chunk fires on weaker containment.
  CHECK(t0_prior(ds, p, ic * 0.9) < t0_prior(ds, p, ic * 0.1));
}

TEST_CASE("train/fire parity — the stored t_p is the firing boundary") {
  Dataset ds = partial_ds(40, 30, 20, 30);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 4;
  auto parts = build_parts(ds, cfg);
  REQUIRE(!parts.empty());
  const auto& p = parts.front();
  const double info_p = ds.info(p.bits);
  CHECK(ds.cont(p.bits, p.bits, info_p) >= p.t_p);  // a signature ⊇ p sits at cont=1 ≥ t_p
  CHECK(p.t_p >= 0.0);
  CHECK(p.t_p <= 1.0);
}

TEST_CASE("reseed is not a halt — structure past a rejected hub is still found") {
  std::vector<std::vector<std::uint32_t>> sigs;
  for (int i = 0; i < 30; ++i) sigs.push_back({0, static_cast<std::uint32_t>(100 + i)});  // hub + unique
  for (int i = 0; i < 30; ++i) { std::vector<std::uint32_t> s{3}; if (i < 20) s.push_back(4); sigs.push_back(std::move(s)); }
  Dataset ds = make_ds(std::move(sigs), 140);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 32;
  auto parts = build_parts(ds, cfg);
  CHECK(parts.size() >= 2);  // did not halt at the hub
  bool covers3 = false;
  for (const auto& p : parts)
    for (std::uint32_t b : p.bits) if (b == 3) covers3 = true;
  CHECK(covers3);            // the {3,4} region past the hub is discovered
}

TEST_CASE("uniform weights: cont reduces to popcount(f∧p)/popcount(p)") {
  Dataset ds = partial_ds(40, 30, 20, 30);
  for (double& w : ds.w) w = 1.0;  // force uniform across all ids
  std::vector<std::uint32_t> p{0, 1, 2}, f{0, 1};
  CHECK(ds.cont(f, p, ds.info(p)) == doctest::Approx(2.0 / 3.0));  // 2 of 3 bits present
}

TEST_CASE("determinism — fixed input yields identical dictionaries") {
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 32;
  Dataset ds1 = partial_ds(40, 30, 20, 30);
  Dataset ds2 = partial_ds(40, 30, 20, 30);
  auto a = build_parts(ds1, cfg);
  auto b = build_parts(ds2, cfg);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].bits == b[i].bits);
}

TEST_CASE("Variant B: positive coverage, no duplicate parts (coverage subsumes incoherence)") {
  Dataset ds = partial_ds(40, 30, 20, 40);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 64;
  BuildStats st;
  auto parts = build_parts(ds, cfg, &st);
  REQUIRE(!parts.empty());
  CHECK(parts.front().alpha > 0.0);  // admitted on positive coverage gain G
  CHECK(st.covered_info > 0.0);
  for (std::size_t i = 0; i < parts.size(); ++i)      // covered bits contribute 0 → no re-discovery
    for (std::size_t j = i + 1; j < parts.size(); ++j)
      CHECK(parts[i].bits != parts[j].bits);
}

TEST_CASE("Variant A: fixed importance α_p=1, no e / negative-importance computed") {
  Dataset ds = partial_ds(40, 30, 20, 30);
  Config cfg; cfg.variant = 'A'; cfg.s_min = 5; cfg.K_max = 16;
  auto parts = build_parts(ds, cfg);
  REQUIRE(!parts.empty());
  for (const auto& p : parts) {
    CHECK(p.alpha == doctest::Approx(1.0));
    CHECK(p.t_p > 0.0);
    CHECK(p.t_p <= 1.0);
  }
}

TEST_CASE("CPRT round-trips") {
  Dataset ds = partial_ds(40, 30, 20, 30);
  Config cfg; cfg.variant = 'B'; cfg.s_min = 5; cfg.K_max = 16;
  auto parts = build_parts(ds, cfg);
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  save_cprt(ss, parts, cfg, 20);
  Config cfg2; std::uint32_t two_f = 0;
  auto loaded = load_cprt(ss, cfg2, two_f);
  CHECK(two_f == 20);
  CHECK(cfg2.variant == 'B');
  REQUIRE(loaded.size() == parts.size());
  for (std::size_t i = 0; i < parts.size(); ++i) {
    CHECK(loaded[i].bits == parts[i].bits);
    CHECK(loaded[i].t_p == doctest::Approx(parts[i].t_p));
    CHECK(loaded[i].alpha == doctest::Approx(parts[i].alpha));
  }
}
