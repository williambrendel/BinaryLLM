// ============================================================================
// core/codebook/src/recon_metrics.cpp
// ============================================================================

#include "recon_metrics.hpp"

namespace core::codebook {

namespace {

// Σ weights over positions of `a` within [lo, hi).
std::uint64_t info_band(const std::vector<std::uint32_t>& a,
                        const std::vector<std::uint32_t>& w, std::uint32_t lo,
                        std::uint32_t hi) {
  std::uint64_t s = 0;
  for (std::uint32_t e : a)
    if (e >= lo && e < hi) s += w[e];
  return s;
}

// Σ weights over positions in a∩b within [lo, hi) (two-pointer merge).
std::uint64_t wdot_band(const std::vector<std::uint32_t>& a,
                        const std::vector<std::uint32_t>& b,
                        const std::vector<std::uint32_t>& w, std::uint32_t lo,
                        std::uint32_t hi) {
  std::uint64_t s = 0;
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] < b[j]) ++i;
    else if (b[j] < a[i]) ++j;
    else {
      if (a[i] >= lo && a[i] < hi) s += w[a[i]];
      ++i;
      ++j;
    }
  }
  return s;
}

void band_scores(const std::vector<std::uint32_t>& recon,
                 const std::vector<std::uint32_t>& truth,
                 const std::vector<std::uint32_t>& w, std::uint32_t lo,
                 std::uint32_t hi, double& recall, double& precision) {
  const std::uint64_t wd = wdot_band(recon, truth, w, lo, hi);
  const std::uint64_t it = info_band(truth, w, lo, hi);
  const std::uint64_t ir = info_band(recon, w, lo, hi);
  // Empty band → nothing to recall / no false positives → 1.
  recall = it ? static_cast<double>(wd) / static_cast<double>(it) : 1.0;
  precision = ir ? static_cast<double>(wd) / static_cast<double>(ir) : 1.0;
}

}  // namespace

ReconMetrics recon_metrics(const std::vector<std::uint32_t>& recon,
                           const std::vector<std::uint32_t>& truth,
                           const std::vector<std::uint32_t>& weights,
                           std::uint32_t F) {
  ReconMetrics m;
  band_scores(recon, truth, weights, 0, 3 * F, m.recall, m.precision);
  band_scores(recon, truth, weights, 0, F, m.recall_L, m.precision_L);
  band_scores(recon, truth, weights, F, 2 * F, m.recall_C, m.precision_C);
  band_scores(recon, truth, weights, 2 * F, 3 * F, m.recall_R, m.precision_R);
  return m;
}

}  // namespace core::codebook
