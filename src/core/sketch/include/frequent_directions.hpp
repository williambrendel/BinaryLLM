#pragma once

// ============================================================================
// core/sketch/frequent_directions.hpp
//
// Frequent Directions (Liberty 2013; Ghashami–Liberty–Phillips–Woodruff
// 2016) over the 3F signature stream — the one-pass, bounded-memory factor
// that replaces any N² token graph or F×F correlation matrix (spec §4.4,
// §7.3).
//
// Maintains an ℓ×d sketch B with the deterministic covariance guarantee
//
//     0 ⪯ AᵀA − BᵀB ⪯ (‖A‖_F² / ℓ) · I
//
// over the streamed rows A. Uses the "fast" 2ℓ-row buffer so the O(ℓ²d)
// densification runs once per ℓ rows (a free algorithmic 2×).
//
// Downstream (pass 1) reads structure off this factor, never off a
// materialized correlation matrix:
//   - bit-bit correlation  = ⟨column i, column j⟩ = (BᵀB)[i,j]
//   - promiscuity / energy = column_energy(j) = (BᵀB)[j,j]
//
// Float32, build-time only (§7.2). The heavy linear algebra runs through the
// swappable core::math kernels.
// ============================================================================

#include "binary_vec.hpp"

#include <cstddef>
#include <vector>

namespace core::sketch {

using binarycore::binary_vec::BinaryVecDynamic;

class FrequentDirections {
 public:
  // dim = 3F signature width; ell = sketch rank (≥ expected effective rank).
  FrequentDirections(std::size_t dim, std::size_t ell);

  // Append one signature row (its set bits become 1.0f). Streaming, O(nnz)
  // amortized plus the periodic densification.
  void add(const BinaryVecDynamic& sig);

  // Final shrink + rotate the sketch onto its principal axes (rows ordered
  // by descending singular value). Call once; accessors below require it.
  void finalize();

  std::size_t dim() const noexcept { return d_; }
  std::size_t ell() const noexcept { return ell_; }
  std::size_t rank() const noexcept { return rank_; }  // valid after finalize
  float singular_value(std::size_t i) const { return sv_[i]; }

  // Sketch factor, rank_ × dim row-major (after finalize). Row i is
  // σ_i · v_iᵀ, so factorᵀ·factor = BᵀB.
  const std::vector<float>& factor() const noexcept { return B_; }

  // (BᵀB)[i][j] — the FD estimate of the i,j bit-correlation.
  double bit_correlation(std::size_t i, std::size_t j) const;
  double column_energy(std::size_t j) const { return bit_correlation(j, j); }

 private:
  void shrink();  // 2ℓ buffer → top ℓ rows (subtract the ℓ-th eigenvalue)

  std::size_t d_;
  std::size_t ell_;
  std::size_t m_;               // buffer height = 2ℓ
  std::vector<float> B_;        // m_×d_ while streaming; rank_×d_ after finalize
  std::size_t filled_ = 0;      // occupied buffer rows
  std::size_t rank_ = 0;
  std::vector<float> sv_;       // singular values of the final sketch
  bool finalized_ = false;
};

}  // namespace core::sketch
