// ============================================================================
// core/parts2f/src/parts2f.cpp — Pass-1 context-part discovery (Variants A/B).
// ============================================================================

#include "parts2f.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace core::parts2f {
namespace {

constexpr std::uint32_t kNoBit = 0xFFFFFFFFu;

// sorted intersection: keep elements of `survivors` that are also in `post`.
std::vector<std::uint32_t> intersect_sorted(const std::vector<std::uint32_t>& survivors,
                                            const std::vector<std::uint32_t>& post) {
  std::vector<std::uint32_t> out;
  std::size_t i = 0, j = 0;
  while (i < survivors.size() && j < post.size()) {
    if (survivors[i] < post[j]) ++i;
    else if (post[j] < survivors[i]) ++j;
    else { out.push_back(survivors[i]); ++i; ++j; }
  }
  return out;
}

// bits of `a` (sorted) present in `b` (sorted) but NOT in `covered` (sorted).
std::vector<std::uint32_t> new_covered(const std::vector<std::uint32_t>& part,
                                       const std::vector<std::uint32_t>& sig,
                                       const std::vector<std::uint32_t>& covered) {
  std::vector<std::uint32_t> inter = intersect_sorted(part, sig);  // p ∩ x_f
  std::vector<std::uint32_t> out;
  std::size_t j = 0;
  for (std::uint32_t b : inter) {
    while (j < covered.size() && covered[j] < b) ++j;
    if (j < covered.size() && covered[j] == b) continue;  // already covered
    out.push_back(b);
  }
  return out;
}

}  // namespace

void Dataset::build_index() {
  inv.assign(dim, {});
  c_e.assign(dim, 0);
  for (std::uint32_t f = 0; f < sigs.size(); ++f)
    for (std::uint32_t e : sigs[f]) inv[e].push_back(f);
  for (std::size_t e = 0; e < dim; ++e) c_e[e] = inv[e].size();
}

double t0_prior(const Dataset& ds, const std::vector<std::uint32_t>& part, double w_last) {
  const double ic = ds.info(part);
  if (ic <= 0.0) return 1.0;
  return 1.0 - w_last / ic;
}

std::vector<std::uint32_t> grow_part(const Dataset& ds, const Config& cfg, std::uint32_t seed,
                                     const std::function<double(std::uint32_t)>& weight_of,
                                     std::vector<std::uint32_t>& survivors, double& w_last) {
  static thread_local std::vector<double> sumw;
  static thread_local std::vector<std::uint32_t> cnt;
  static thread_local std::vector<char> inP;
  if (sumw.size() < ds.dim) { sumw.assign(ds.dim, 0.0); cnt.assign(ds.dim, 0); inP.assign(ds.dim, 0); }

  std::vector<std::uint32_t> p{seed};
  inP[seed] = 1;
  survivors = ds.inv[seed];  // sorted signature ids containing the seed
  w_last = ds.w[seed];

  std::vector<std::uint32_t> touched;
  while (p.size() < cfg.D) {
    // tally each candidate bit's (survivor count, summed survivor weight)
    touched.clear();
    for (std::uint32_t f : survivors)
      for (std::uint32_t i : ds.sigs[f])
        if (!inP[i]) {
          if (cnt[i] == 0) touched.push_back(i);
          ++cnt[i];
          sumw[i] += weight_of(f);
        }
    std::sort(touched.begin(), touched.end());  // ascending id → strict-> keeps lowest id on ties
    const double lS = std::log(static_cast<double>(survivors.size()));
    double best = 0.0;
    std::uint32_t bi = kNoBit;
    for (std::uint32_t i : touched) {
      const double cs = lS - std::log(static_cast<double>(cnt[i]));  // conditional surprisal in W
      const double score = cs * sumw[i];
      if (score > best) { best = score; bi = i; }
    }
    for (std::uint32_t i : touched) { cnt[i] = 0; sumw[i] = 0.0; }
    if (bi == kNoBit) break;  // no positive-score candidate

    std::vector<std::uint32_t> next = intersect_sorted(survivors, ds.inv[bi]);
    if (next.size() < cfg.s_min) break;  // test-then-commit rollback: preserve support floor
    p.push_back(bi);
    inP[bi] = 1;
    survivors.swap(next);
    w_last = ds.w[bi];  // surprisal of the last KEPT bit
  }
  for (std::uint32_t i : p) inP[i] = 0;
  std::sort(p.begin(), p.end());
  return p;
}

// Gather signatures sharing ≥1 bit with `part` (firing candidates), dedup via `seen`.
static std::vector<std::uint32_t> firing_candidates(const Dataset& ds,
                                                    const std::vector<std::uint32_t>& part,
                                                    std::vector<char>& seen) {
  std::vector<std::uint32_t> cand;
  for (std::uint32_t b : part)
    for (std::uint32_t f : ds.inv[b])
      if (!seen[f]) { seen[f] = 1; cand.push_back(f); }
  for (std::uint32_t f : cand) seen[f] = 0;
  return cand;
}

// ---------------------------------------------------------------------------
// Variant B — greedy submodular set-cover on bit-level residuals (§6).
// ---------------------------------------------------------------------------
static std::vector<Part> build_variant_b(const Dataset& ds, const Config& cfg, BuildStats* stats) {
  const std::size_t N = ds.sigs.size();
  std::vector<std::vector<std::uint32_t>> cov(N);   // covered bits per signature (sorted)
  std::vector<double> ri(N);                         // residual info = info(res_f)
  double total_info = 0.0;
  for (std::size_t f = 0; f < N; ++f) { ri[f] = ds.info(ds.sigs[f]); total_info += ri[f]; }

  std::vector<double> We(ds.dim, 0.0);               // W_e = Σ_{f: e∈res_f} ri[f]
  for (std::size_t f = 0; f < N; ++f)
    for (std::uint32_t e : ds.sigs[f]) We[e] += ri[f];

  std::vector<char> exhausted(ds.dim, 0), seen(N, 0);
  std::vector<std::uint32_t> active;
  for (std::uint32_t e = 0; e < ds.dim; ++e)
    if (ds.c_e[e] >= cfg.s_min) active.push_back(e);

  std::vector<Part> parts;
  double covered_info = 0.0;
  while (parts.size() < cfg.K_max) {
    // seed = heaviest residual bit with support ≥ s_min, not exhausted
    std::uint32_t seed = kNoBit;
    double best = 0.0;
    for (std::uint32_t e : active)
      if (!exhausted[e] && We[e] > best) { best = We[e]; seed = e; }
    if (seed == kNoBit) break;

    std::vector<std::uint32_t> S;
    double w_last = 0.0;
    auto weight_of = [&](std::uint32_t f) { return ri[f]; };
    std::vector<std::uint32_t> p = grow_part(ds, cfg, seed, weight_of, S, w_last);
    const double info_p = ds.info(p);
    const double t_p = t0_prior(ds, p, w_last);  // §6.3 default (solver optional)

    // fire set + coverage gain G
    std::vector<std::uint32_t> cand = firing_candidates(ds, p, seen);
    std::vector<std::uint32_t> fired;
    double G = 0.0;
    for (std::uint32_t f : cand) {
      if (ds.cont(ds.sigs[f], p, info_p) < t_p) continue;
      fired.push_back(f);
      G += ds.info(new_covered(p, ds.sigs[f], cov[f]));
    }

    RoundStat rs{seed, static_cast<std::uint32_t>(p.size()), static_cast<std::uint32_t>(S.size()),
                 static_cast<std::uint32_t>(fired.size()), t_p, G, false};
    const bool admit = (G > cfg.g_min) && (t_p < cfg.t_max) && (fired.size() >= cfg.s_min);
    if (admit) {
      for (std::uint32_t f : fired) {
        std::vector<std::uint32_t> nc = new_covered(p, ds.sigs[f], cov[f]);
        if (nc.empty()) continue;
        double dm = 0.0;
        for (std::uint32_t b : nc) dm += ds.w[b];
        const double old_ri = ri[f];
        for (std::uint32_t b : nc) We[b] -= old_ri;          // b leaves res_f → drop its full contribution
        ri[f] = old_ri - dm;
        // merge nc into cov[f]
        std::vector<std::uint32_t> merged;
        merged.reserve(cov[f].size() + nc.size());
        std::set_union(cov[f].begin(), cov[f].end(), nc.begin(), nc.end(),
                       std::back_inserter(merged));
        cov[f].swap(merged);
        // remaining residual bits e still in res_f: contribution shrank by dm
        std::size_t j = 0;
        for (std::uint32_t e : ds.sigs[f]) {
          while (j < cov[f].size() && cov[f][j] < e) ++j;
          if (j < cov[f].size() && cov[f][j] == e) continue;  // covered → not in residual
          We[e] -= dm;
        }
        covered_info += dm;
      }
      parts.push_back({p, G, t_p, static_cast<std::uint32_t>(S.size())});
      rs.admitted = true;
    } else {
      exhausted[seed] = 1;
    }
    if (stats) stats->rounds.push_back(rs);
  }
  if (stats) { stats->covered_info = covered_info; stats->total_info = total_info; }
  return parts;
}

// ---------------------------------------------------------------------------
// Variant A — boosted-threshold (§5). Sample weights + α_p=1, t_p solved.
// Threshold solved over the firing-candidate containment distribution (§5.2);
// admission by loss drop ΔL<0 restricted to the touched (candidate) set — the
// s≈0 majority carries a constant exp(1) factor that cancels in ΔL (see notes).
// ---------------------------------------------------------------------------
static double margin(double s, double t) {
  return (s > t) ? (s - t) / (1.0 - t) : (s - t) / t;
}

// solve t minimizing Σ_cand w_f exp(−margin(s_f,t)); grid over candidate conts, clamped.
static double solve_threshold(const std::vector<double>& conts, const std::vector<double>& wf,
                              double t0, double delta, double t_max) {
  const double lo = std::max(1e-3, t0 - delta), hi = std::min(t_max, 1.0 - 1e-3);
  double best_t = std::min(std::max(t0, lo), hi), best_L = 1e300;
  const int G = 64;
  for (int g = 0; g <= G; ++g) {
    const double t = lo + (hi - lo) * g / G;
    double L = 0.0;
    for (std::size_t i = 0; i < conts.size(); ++i) L += wf[i] * std::exp(-margin(conts[i], t));
    if (L < best_L) { best_L = L; best_t = t; }
  }
  return best_t;
}

static std::vector<Part> build_variant_a(const Dataset& ds, const Config& cfg, BuildStats* stats) {
  const std::size_t N = ds.sigs.size();
  std::vector<double> wf(N, 1.0 / static_cast<double>(N));  // AdaBoost sample distribution
  std::vector<char> exhausted(ds.dim, 0), seen(N, 0);
  std::vector<std::uint32_t> active;
  for (std::uint32_t e = 0; e < ds.dim; ++e)
    if (ds.c_e[e] >= cfg.s_min) active.push_back(e);

  std::vector<Part> parts;
  while (parts.size() < cfg.K_max) {
    std::uint32_t seed = kNoBit;
    double best = 0.0;
    for (std::uint32_t e : active) {
      if (exhausted[e]) continue;
      double s = 0.0;
      for (std::uint32_t f : ds.inv[e]) s += wf[f];
      if (s > best) { best = s; seed = e; }
    }
    if (seed == kNoBit) break;

    std::vector<std::uint32_t> S;
    double w_last = 0.0;
    auto weight_of = [&](std::uint32_t f) { return wf[f]; };
    std::vector<std::uint32_t> p = grow_part(ds, cfg, seed, weight_of, S, w_last);
    const double info_p = ds.info(p);
    const double t0 = t0_prior(ds, p, w_last);

    std::vector<std::uint32_t> cand = firing_candidates(ds, p, seen);
    std::vector<double> conts(cand.size()), cw(cand.size());
    for (std::size_t i = 0; i < cand.size(); ++i) {
      conts[i] = ds.cont(ds.sigs[cand[i]], p, info_p);
      cw[i] = wf[cand[i]];
    }
    const double t_p = solve_threshold(conts, cw, t0, cfg.delta, cfg.t_max);  // §5.2 (A always solves)

    // ΔL over candidates (constant s≈0 term cancels); firing count guard.
    double dL = 0.0;
    std::uint32_t fired = 0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      const double m = margin(conts[i], t_p);
      dL += cw[i] * (std::exp(-m) - 1.0);
      if (conts[i] > t_p) ++fired;
    }
    RoundStat rs{seed, static_cast<std::uint32_t>(p.size()), static_cast<std::uint32_t>(S.size()),
                 fired, t_p, dL, false};
    const bool admit = (dL < 0.0) && (t_p < cfg.t_max) && (fired >= cfg.s_min);
    if (admit) {
      for (std::uint32_t f : cand) {
        const double s = ds.cont(ds.sigs[f], p, info_p);
        wf[f] *= std::exp(-margin(s, t_p));  // α_p = 1
      }
      double Z = 0.0;
      for (double v : wf) Z += v;
      if (Z > 0.0) for (double& v : wf) v /= Z;
      parts.push_back({p, 1.0, t_p, static_cast<std::uint32_t>(S.size())});
      rs.admitted = true;
    } else {
      exhausted[seed] = 1;
    }
    if (stats) stats->rounds.push_back(rs);
  }
  return parts;
}

std::vector<Part> build_parts(const Dataset& ds, const Config& cfg, BuildStats* stats) {
  return cfg.variant == 'A' ? build_variant_a(ds, cfg, stats) : build_variant_b(ds, cfg, stats);
}

// ---------------------------------------------------------------------------
// CPRT dictionary artifact (§15). magic 'CPRT', v1.
// ---------------------------------------------------------------------------
namespace {
constexpr std::uint32_t kMagic = 0x54525043;  // 'CPRT' little-endian
template <class T> void wr(std::ostream& os, T v) { os.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> T rd(std::istream& is) { T v{}; is.read(reinterpret_cast<char*>(&v), sizeof(T)); return v; }
}  // namespace

void save_cprt(std::ostream& os, const std::vector<Part>& parts, const Config& cfg,
               std::uint32_t two_f) {
  wr<std::uint32_t>(os, kMagic);
  wr<std::uint32_t>(os, 1);  // version
  wr<std::uint32_t>(os, two_f);
  wr<std::uint32_t>(os, static_cast<std::uint32_t>(parts.size()));
  wr<std::uint32_t>(os, cfg.D);
  wr<char>(os, cfg.variant);
  wr<std::uint32_t>(os, cfg.s_min);
  for (const auto& p : parts) {
    wr<std::uint32_t>(os, static_cast<std::uint32_t>(p.bits.size()));
    wr<std::uint32_t>(os, p.support);
    wr<double>(os, p.alpha);
    wr<double>(os, p.t_p);
    for (std::uint32_t b : p.bits) wr<std::uint32_t>(os, b);
  }
}

std::vector<Part> load_cprt(std::istream& is, Config& cfg_out, std::uint32_t& two_f_out) {
  if (rd<std::uint32_t>(is) != kMagic) throw std::runtime_error("parts2f: bad CPRT magic");
  rd<std::uint32_t>(is);  // version
  two_f_out = rd<std::uint32_t>(is);
  const std::uint32_t K = rd<std::uint32_t>(is);
  cfg_out.D = rd<std::uint32_t>(is);
  cfg_out.variant = rd<char>(is);
  cfg_out.s_min = rd<std::uint32_t>(is);
  std::vector<Part> parts(K);
  for (auto& p : parts) {
    const std::uint32_t n = rd<std::uint32_t>(is);
    p.support = rd<std::uint32_t>(is);
    p.alpha = rd<double>(is);
    p.t_p = rd<double>(is);
    p.bits.resize(n);
    for (std::uint32_t& b : p.bits) b = rd<std::uint32_t>(is);
  }
  return parts;
}

}  // namespace core::parts2f
