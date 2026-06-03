// ============================================================================
// binaryoptim/src/dominant_set.cpp
//
// Replicator dynamics implementation of the gamma-scaled dominant-set solver.
// See dominant_set.hpp for the math.
// ============================================================================

#include "binaryoptim/dominant_set.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace binaryoptim {

template <typename T>
DominantSetResult<T> dominant_set(
    const T* A,
    const T* b,
    const T* beta,
    std::size_t N,
    const DominantSetOptions<T>& opts) {

  DominantSetResult<T> result;
  result.alpha.assign(N, T{});
  result.iterations = 0;
  result.converged = false;

  if (N == 0) return result;

  const T gamma = opts.gamma;

  // -------------------------------------------------------------------------
  // Build M = A - diag(beta) + kappa * 11^T.
  //   - kappa = max(beta) guarantees the diagonal kappa - beta[i] >= 0.
  //   - Adding kappa to every off-diagonal entry guarantees M >= 0 entrywise
  //     when A >= 0 (which is the regime the caller is expected to supply,
  //     e.g. a Hamming-derived affinity matrix).
  //   - The kappa * 11^T term is a constant on the simplex (rank-1 shift by
  //     a multiple of (sum alpha)^2 = gamma^2) so the argmax is preserved.
  //
  // M is stored as its own buffer because:
  //   - A's diagonal is ignored by the problem statement, and we need a
  //     diagonal that depends on beta.
  //   - We add kappa to every off-diagonal entry once, instead of inside
  //     the inner loop on each iteration.
  // -------------------------------------------------------------------------
  T kappa = T{};
  for (std::size_t i = 0; i < N; ++i) {
    if (beta[i] > kappa) kappa = beta[i];
  }

  std::vector<T> M(N * N);
  for (std::size_t i = 0; i < N; ++i) {
    M[i * N + i] = kappa - beta[i];                  // diagonal
    for (std::size_t j = i + 1; j < N; ++j) {
      const T off = A[i * N + j] + kappa;            // off-diagonal (symmetric)
      M[i * N + j] = off;
      M[j * N + i] = off;
    }
  }

  // -------------------------------------------------------------------------
  // Initialize alpha to the uniform distribution on the gamma-simplex.
  // -------------------------------------------------------------------------
  const T init = gamma / static_cast<T>(N);
  for (std::size_t i = 0; i < N; ++i) result.alpha[i] = init;

  std::vector<T> M_alpha(N, T{});

  // -------------------------------------------------------------------------
  // Replicator iteration.
  //   alpha_i  <-  alpha_i * ((M alpha)_i + b_i) / Z
  //   Z = sum_i alpha_i * ((M alpha)_i + b_i)
  // and rescale to enforce sum alpha = gamma after each step.
  //
  // The Mα loop exploits symmetry of M: each off-diagonal entry M[i,j] is
  // touched once and contributes alpha[j] to M_alpha[i] and alpha[i] to
  // M_alpha[j], cutting work in half.
  // -------------------------------------------------------------------------
  std::size_t iter = 0;
  for (; iter < opts.max_iter; ++iter) {

    // Compute M_alpha exploiting symmetry.
    std::fill(M_alpha.begin(), M_alpha.end(), T{});
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t row = i * N;
      const T alpha_i = result.alpha[i];
      M_alpha[i] += M[row + i] * alpha_i;            // diagonal
      for (std::size_t j = i + 1; j < N; ++j) {
        const T m = M[row + j];
        M_alpha[i] += m * result.alpha[j];
        M_alpha[j] += m * alpha_i;
      }
    }

    // Compute denominator Z = alpha^T (M alpha + b).
    T denom = T{};
    for (std::size_t i = 0; i < N; ++i) {
      denom += result.alpha[i] * (M_alpha[i] + b[i]);
    }
    if (!(denom > T{})) break;   // degenerate; stop and keep current alpha

    // Apply replicator update and track L-infinity change.
    // We update in place. Note: the standard update preserves
    // sum(alpha) = (current sum), so after this loop the sum is still
    // approximately gamma. We rescale explicitly below to remove drift.
    T max_delta = T{};
    T new_sum = T{};
    for (std::size_t i = 0; i < N; ++i) {
      const T a_new = result.alpha[i] * (M_alpha[i] + b[i]) / denom * gamma;
      const T d = std::abs(a_new - result.alpha[i]);
      if (d > max_delta) max_delta = d;
      result.alpha[i] = a_new;
      new_sum += a_new;
    }

    // Rescale to exactly gamma to absorb numerical drift over many iters.
    if (new_sum > T{} && std::abs(new_sum - gamma) > static_cast<T>(0)) {
      const T scale = gamma / new_sum;
      for (std::size_t i = 0; i < N; ++i) result.alpha[i] *= scale;
    }

    if (max_delta < opts.tol) {
      ++iter;
      result.converged = true;
      break;
    }
  }

  result.iterations = iter;
  return result;
}

// Explicit instantiations.
template DominantSetResult<float>  dominant_set<float>(
    const float*, const float*, const float*, std::size_t,
    const DominantSetOptions<float>&);
template DominantSetResult<double> dominant_set<double>(
    const double*, const double*, const double*, std::size_t,
    const DominantSetOptions<double>&);

}  // namespace binaryoptim
