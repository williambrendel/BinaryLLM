#pragma once

// ============================================================================
// core/codebook/recon_metrics.hpp
//
// Surprisal-weighted, block-aware reconstruction fidelity (spec §7.5.2).
//
// The decode is the non-learned union of fired codeword masks (Codebook::
// decode — integer, no float, no learned decoder). This scores that decode
// against the true 3F signature in the shipped surprisal metric:
//
//   weighted_recall    = weighted_dot(recon, true) / info_content(true)
//   weighted_precision = weighted_dot(recon, true) / info_content(recon)
//   FN = 1 − recall    (missed true bits — the real reconstruction failure)
//   FP = 1 − precision (phantom bits the masks add)
//
// Restricted to each typed band gives BER_C / BER_L / BER_R. Bands over the
// 3F space: L = [0,F), C = [F,2F), R = [2F,3F). Expectation: BER_C recall → 1
// (identity, near-lossless), BER_L/R graded (OR-pooled context, lossy by
// design). Empty band → recall/precision defined as 1 (nothing to miss/add).
// ============================================================================

#include <cstdint>
#include <vector>

namespace core::codebook {

struct ReconMetrics {
  double recall = 0.0, precision = 0.0;                    // overall (3F)
  double recall_L = 0.0, recall_C = 0.0, recall_R = 0.0;   // per band
  double precision_L = 0.0, precision_C = 0.0, precision_R = 0.0;

  double fn() const { return 1.0 - recall; }
  double fp() const { return 1.0 - precision; }
};

// recon / truth: sorted-unique 3F position lists. weights: uint32 surprisal
// multiplicities indexed by 3F position. F: single-band width (3F = 3·F).
ReconMetrics recon_metrics(const std::vector<std::uint32_t>& recon,
                           const std::vector<std::uint32_t>& truth,
                           const std::vector<std::uint32_t>& weights,
                           std::uint32_t F);

}  // namespace core::codebook
