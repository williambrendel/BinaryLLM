#pragma once

// ============================================================================
// core/parts2f — Pass-1 context-part discovery over the 2F = [L | R] block.
//
// Authoritative two-variant spec (BinaryLLM Pass 1). Discover a compact
// dictionary of reusable context-parts so each near-unique context signature is
// reconstructed by the logical OR of the few parts that fire on it:
//     x_f  ≈  OR_{p : cont(x_f,p) ≥ t_p}  p
// scored in the surprisal-weighted metric. Two selection strategies share the
// same part growth (§3.1) and per-part threshold t_p (§3.2):
//
//   Variant B (default) — greedy submodular set-cover on sparse bit-level
//     residuals res_f = x_f \ cov_f; admit on marginal surprisal-coverage G.
//   Variant A — AdaBoost loop with the classifier weight fixed α_p = 1 and the
//     per-part threshold t_p solved from the exponential loss; admit on ΔL < 0.
//
// Matrix-free: no N×2F, no N², no F×F ever formed. Floats only at build time.
// ============================================================================

#include "info_jaccard.hpp"

#include <cstdint>
#include <functional>
#include <istream>
#include <ostream>
#include <vector>

namespace core::parts2f {

// A discovered context-part codeword.
struct Part {
  std::vector<std::uint32_t> bits;  // sorted set-bit ids over the 2F space, popcount ≤ D
  double alpha = 1.0;               // ensemble importance: 1 (Variant A) or coverage G (Variant B)
  double t_p = 1.0;                 // firing threshold (used identically in training and inference)
  std::uint32_t support = 0;        // exact-AND survivor count at grow time (≥ s_min)
};

struct Config {
  std::uint32_t K_max = 8192;  // dictionary budget
  std::uint32_t D = 256;       // max part popcount
  std::uint32_t s_min = 20;    // support floor
  double t_max = 0.98;         // degeneracy guard on t_p (a part "declines" above this)
  double delta = 0.1;          // clamp band for t_p around its t0 prior (Variant A)
  double g_min = 1e-9;         // minimum coverage gain to admit a part (Variant B)
  char variant = 'B';          // 'A' or 'B'
  bool solve_threshold = false;  // Variant B: use the α=1 solver instead of the t0 prior
};

// Prepared 2F sample set: sparse signatures + frozen surprisal + inverted index.
// `w` is indexed by global bit id (size = dim); only 2F (L/R) ids are populated.
struct Dataset {
  std::size_t dim = 0;
  std::vector<std::vector<std::uint32_t>> sigs;  // each sorted
  std::vector<std::vector<std::uint32_t>> inv;   // bit id -> sorted signature ids (posting lists)
  std::vector<double> w;                         // surprisal weight per bit id
  std::vector<std::uint64_t> c_e;                // support count per bit (= |inv[e]|)

  void build_index();  // fill inv and c_e from sigs (call once after sigs/dim/w are set)

  double info(const std::vector<std::uint32_t>& a) const {
    return binarycore::sparse::info_content(a.data(), a.size(), w.data());
  }
  double wdot(const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) const {
    return binarycore::sparse::weighted_dot(a.data(), a.size(), b.data(), b.size(), w.data());
  }
  // cont(f,p) = weighted_dot(x_f,p)/info_content(p)  ∈ [0,1]
  double cont(const std::vector<std::uint32_t>& x, const std::vector<std::uint32_t>& p,
              double info_p) const {
    return info_p > 0.0 ? wdot(x, p) / info_p : 0.0;
  }
};

// Per-round diagnostics (stats report, §15).
struct RoundStat {
  std::uint32_t seed;
  std::uint32_t width;
  std::uint32_t support;      // exact-AND survivors
  std::uint32_t fired;        // signatures firing at t_p
  double t_p;
  double signal;              // admission signal: G (B) or ΔL (A)
  bool admitted;
};
struct BuildStats {
  std::vector<RoundStat> rounds;
  double covered_info = 0.0;  // total surprisal explained (Variant B)
  double total_info = 0.0;
};

// Growth (§3.1): grow a part from `seed`, scoring candidate bit i by
//   (conditional surprisal within survivors) × (Σ survivor weight carrying i),
// exact-membership survivor filter, test-then-commit rollback at s_min.
// weight_of(f) returns the per-signature weight (w_f for A, info(res_f) for B).
// Returns the sorted part; fills `survivors` (exact-AND support) and `w_last`.
std::vector<std::uint32_t> grow_part(const Dataset& ds, const Config& cfg, std::uint32_t seed,
                                     const std::function<double(std::uint32_t)>& weight_of,
                                     std::vector<std::uint32_t>& survivors, double& w_last);

// t0 prior (§3.2): 1 − w_last / info_content(p).
double t0_prior(const Dataset& ds, const std::vector<std::uint32_t>& part, double w_last);

// Main entry: dispatches on cfg.variant.
std::vector<Part> build_parts(const Dataset& ds, const Config& cfg, BuildStats* stats = nullptr);

// CPRT dictionary artifact (§15).
void save_cprt(std::ostream& os, const std::vector<Part>& parts, const Config& cfg,
               std::uint32_t two_f);
std::vector<Part> load_cprt(std::istream& is, Config& cfg_out, std::uint32_t& two_f_out);

}  // namespace core::parts2f
