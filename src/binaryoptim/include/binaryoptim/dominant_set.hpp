#pragma once

// ============================================================================
// binaryoptim/dominant_set.hpp
//
// Pavan-Pelillo dominant-set solver via replicator dynamics, generalized to
// the gamma-scaled simplex.
//
// The solver maximizes
//
//     alpha^T A alpha + alpha^T b - alpha^T diag(beta) alpha
//
// subject to the simplex constraint
//
//     alpha_i >= 0,   sum_i alpha_i = gamma
//
// where:
//   A     symmetric N x N matrix, row-major contiguous, A[i,i] handled by beta
//         (the diagonal of A is ignored — the diagonal penalty comes from beta).
//   b     length-N linear term (e.g. external relevance per candidate).
//   beta  length-N per-candidate diagonal regularization, beta[i] >= 0.
//         Higher beta[i] suppresses candidate i from concentrating mass.
//   gamma total simplex mass. gamma = 1 reproduces standard Pavan-Pelillo.
//
// The Pavan-Pelillo shift kappa = max(beta) is used to construct a nonneg
// matrix M = A - diag(beta) + kappa * 11^T whose argmax on the simplex
// matches the original objective. The replicator update
//
//     alpha_i  <-  alpha_i * ((M alpha)_i + b_i) / (alpha^T M alpha + alpha^T b)
//
// preserves the simplex; we then rescale to sum to gamma at each step.
//
// Storage and computation conventions:
//   - All matrices and vectors are caller-allocated contiguous arrays.
//   - The solver is a pure primitive: it does NOT do any pre-filtering
//     of candidates or any post-thresholding of alpha. Compose with the
//     helpers in binaryoptim/thresholds.hpp for those steps.
//   - Symmetry of M is exploited in the inner loop: each off-diagonal
//     entry is read once and contributes to both M*alpha[i] and M*alpha[j].
// ============================================================================

#include <cstddef>
#include <vector>

namespace binaryoptim {

template <typename T = double>
struct DominantSetOptions {
  std::size_t max_iter = 300;
  T           tol      = static_cast<T>(1e-5);
  T           gamma    = static_cast<T>(1);
};

template <typename T = double>
struct DominantSetResult {
  std::vector<T> alpha;        // length N, sums to gamma
  std::size_t    iterations;   // actual replicator iterations executed
  bool           converged;    // true iff the iteration stopped on tol, not max_iter
};

// Solve max alpha^T A alpha + alpha^T b - alpha^T diag(beta) alpha
//       s.t.   alpha_i >= 0,   sum_i alpha_i = gamma
//
// Inputs:
//   A      symmetric N x N matrix, row-major, length N*N. A[i,i] is ignored.
//   b      length N, linear term. May be all zeros.
//   beta   length N, diagonal regularization (>= 0).
//   N      problem size.
//
// On entry alpha is uniform = gamma / N. The replicator runs until either
// max_iter is reached or the L-infinity change in alpha drops below tol.
template <typename T = double>
DominantSetResult<T> dominant_set(
    const T* A,
    const T* b,
    const T* beta,
    std::size_t N,
    const DominantSetOptions<T>& opts = {});

// Convenience overload for std::vector inputs.
template <typename T = double>
inline DominantSetResult<T> dominant_set(
    const std::vector<T>& A,
    const std::vector<T>& b,
    const std::vector<T>& beta,
    std::size_t N,
    const DominantSetOptions<T>& opts = {}) {
  return dominant_set<T>(A.data(), b.data(), beta.data(), N, opts);
}

}  // namespace binaryoptim
