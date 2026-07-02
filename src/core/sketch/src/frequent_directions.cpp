// ============================================================================
// core/sketch/src/frequent_directions.cpp
// ============================================================================

#include "frequent_directions.hpp"

#include "dense.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace core::sketch {

namespace {
constexpr std::size_t kChunk = BinaryVecDynamic::chunk_size;
constexpr float kSvEps = 1e-6f;
}  // namespace

FrequentDirections::FrequentDirections(std::size_t dim, std::size_t ell)
    : d_(dim), ell_(ell), m_(2 * ell) {
  if (ell_ == 0) throw std::invalid_argument("FrequentDirections: ell == 0");
  B_.assign(m_ * d_, 0.0f);
}

void FrequentDirections::add(const BinaryVecDynamic& sig) {
  if (finalized_) throw std::logic_error("FrequentDirections: add after finalize");
  if (filled_ == m_) shrink();
  float* row = B_.data() + filled_ * d_;  // freshly zeroed slot
  for (std::size_t k = 0; k < sig.chunks.size(); ++k) {
    const std::size_t base = k * kChunk;
    for (std::uint16_t local : sig.chunks[k].data) {
      const std::size_t g = base + local;
      if (g < d_) row[g] = 1.0f;
    }
  }
  ++filled_;
}

void FrequentDirections::shrink() {
  const std::size_t m = filled_;  // == m_ == 2ℓ when called from add()
  std::vector<float> G(m * m);
  core::math::gram(B_.data(), m, d_, G.data());

  std::vector<float> eval(m), evec(m * m);
  core::math::symeig_desc(G.data(), m, eval.data(), evec.data());

  // Fast-FD shrink: subtract the ℓ-th eigenvalue, keeping the top ℓ rows.
  const double delta = std::max(0.0f, eval[ell_]);

  // New sketch rows i<ℓ: σ'_i · v_iᵀ = (σ'_i/σ_i) · (w_iᵀ B). Build the
  // (ℓ×m) mixing matrix M[i][k] = (σ'_i/σ_i) · w_i[k], then B ← M·B.
  std::vector<float> M(ell_ * m);
  for (std::size_t i = 0; i < ell_; ++i) {
    const double lam = std::max(0.0f, eval[i]);
    const double sig = std::sqrt(lam);
    const double sp = std::sqrt(std::max(0.0, lam - delta));
    const double ci = (sig > 1e-12) ? sp / sig : 0.0;
    for (std::size_t k = 0; k < m; ++k)
      M[i * m + k] = static_cast<float>(ci * evec[k * m + i]);
  }

  std::vector<float> newB(ell_ * d_);
  core::math::matmul(M.data(), ell_, m, B_.data(), d_, newB.data());
  std::copy(newB.begin(), newB.end(), B_.begin());
  std::fill(B_.begin() + ell_ * d_, B_.begin() + m_ * d_, 0.0f);
  filled_ = ell_;
}

void FrequentDirections::finalize() {
  if (finalized_) return;
  if (filled_ > ell_) shrink();

  const std::size_t m = filled_;
  if (m == 0) {
    B_.clear();
    finalized_ = true;
    return;
  }

  // Rotate the retained rows onto principal axes so factor() rows are the
  // singular directions ordered by descending σ (rotation leaves BᵀB fixed).
  std::vector<float> G(m * m);
  core::math::gram(B_.data(), m, d_, G.data());
  std::vector<float> eval(m), evec(m * m);
  core::math::symeig_desc(G.data(), m, eval.data(), evec.data());

  rank_ = 0;
  sv_.clear();
  for (std::size_t i = 0; i < m; ++i) {
    const float s = std::sqrt(std::max(0.0f, eval[i]));
    if (s > kSvEps) {
      sv_.push_back(s);
      ++rank_;
    }
  }

  // principal factor row i = w_iᵀ B  → Wt[i][k] = w_i[k] = evec[k][i].
  std::vector<float> Wt(rank_ * m);
  for (std::size_t i = 0; i < rank_; ++i)
    for (std::size_t k = 0; k < m; ++k) Wt[i * m + k] = evec[k * m + i];

  std::vector<float> P(rank_ * d_);
  if (rank_ > 0) core::math::matmul(Wt.data(), rank_, m, B_.data(), d_, P.data());
  B_.swap(P);
  finalized_ = true;
}

double FrequentDirections::bit_correlation(std::size_t i, std::size_t j) const {
  double s = 0.0;
  for (std::size_t r = 0; r < rank_; ++r) {
    const float* row = B_.data() + r * d_;
    s += static_cast<double>(row[i]) * static_cast<double>(row[j]);
  }
  return s;
}

}  // namespace core::sketch
