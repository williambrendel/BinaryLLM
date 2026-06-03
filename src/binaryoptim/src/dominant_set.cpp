// ============================================================================
// binaryoptim/src/dominant_set.cpp
//
// Replicator dynamics implementation of the gamma-scaled dominant-set solver.
// See dominant_set.hpp for the math and API tiers.
// ============================================================================

#include "binaryoptim/dominant_set.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace binaryoptim {

namespace {

// ---------------------------------------------------------------------------
// Initialize alpha buffer following the auto-detection rule documented in
// the header:
//   - all-zero (or non-positive sum) -> uniform gamma/N
//   - otherwise                       -> clamp negatives to 0, then rescale
// Optional jitter in [0, epsilon) per entry is added before the final
// rescale to gamma.
// ---------------------------------------------------------------------------
template <typename T>
void init_alpha(T* alpha, std::size_t N, T gamma,
                T init_epsilon, std::uint64_t seed) {
  // 1. Compute sum of non-negative entries; clamp negatives in place.
  T sum_pos = T{};
  for (std::size_t i = 0; i < N; ++i) {
    if (alpha[i] < T{}) alpha[i] = T{};
    sum_pos += alpha[i];
  }

  // 2. If empty / all-zero, fall back to uniform.
  if (!(sum_pos > T{})) {
    const T u = gamma / static_cast<T>(N);
    for (std::size_t i = 0; i < N; ++i) alpha[i] = u;
  }

  // 3. Optional jitter.
  if (init_epsilon > T{}) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(T{}, init_epsilon);
    for (std::size_t i = 0; i < N; ++i) alpha[i] += dist(rng);
  }

  // 4. Final rescale to sum = gamma.
  T total = T{};
  for (std::size_t i = 0; i < N; ++i) total += alpha[i];
  if (total > T{}) {
    const T s = gamma / total;
    for (std::size_t i = 0; i < N; ++i) alpha[i] *= s;
  }
}

// ---------------------------------------------------------------------------
// Build M = A - diag(beta) + kappa * 11^T, where kappa = max(beta).
// Off-diagonal: M[i,j] = A[i,j] + kappa
// Diagonal:     M[i,i] = kappa - beta[i]   (>= 0 by choice of kappa)
// The kappa * 11^T shift is a constant on the gamma-simplex (it adds
// kappa * gamma^2 to the objective) and therefore preserves the argmax.
// ---------------------------------------------------------------------------
template <typename T>
void build_M(const T* A, const T* beta, std::size_t N, T* M) {
  T kappa = T{};
  for (std::size_t i = 0; i < N; ++i) {
    if (beta[i] > kappa) kappa = beta[i];
  }
  for (std::size_t i = 0; i < N; ++i) {
    M[i * N + i] = kappa - beta[i];                  // diagonal
    for (std::size_t j = i + 1; j < N; ++j) {
      const T off = A[i * N + j] + kappa;
      M[i * N + j] = off;
      M[j * N + i] = off;                            // symmetric
    }
  }
}

// ---------------------------------------------------------------------------
// Core replicator loop. alpha is assumed already initialized and on the
// gamma-simplex. M is assumed already built.
// ---------------------------------------------------------------------------
template <typename T>
DominantSetStats replicator_loop(
    const T* M, const T* b,
    std::size_t N,
    T* alpha, T* M_alpha,
    const DominantSetOptions<T>& opts) {

  DominantSetStats stats{};
  const T gamma = opts.gamma;

  std::size_t iter = 0;
  for (; iter < opts.max_iter; ++iter) {

    // M * alpha (exploit symmetry: each off-diagonal entry touched once).
    for (std::size_t i = 0; i < N; ++i) M_alpha[i] = T{};
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t row = i * N;
      const T alpha_i = alpha[i];
      M_alpha[i] += M[row + i] * alpha_i;
      for (std::size_t j = i + 1; j < N; ++j) {
        const T m = M[row + j];
        M_alpha[i] += m * alpha[j];
        M_alpha[j] += m * alpha_i;
      }
    }

    // Denominator Z = alpha^T (M alpha + b).
    T denom = T{};
    for (std::size_t i = 0; i < N; ++i) {
      denom += alpha[i] * (M_alpha[i] + b[i]);
    }
    if (!(denom > T{})) break;  // degenerate; stop with current alpha

    // Replicator update + L-inf change tracking + sum tracking.
    T max_delta = T{};
    T new_sum = T{};
    for (std::size_t i = 0; i < N; ++i) {
      const T a_new = alpha[i] * (M_alpha[i] + b[i]) / denom * gamma;
      const T d = std::abs(a_new - alpha[i]);
      if (d > max_delta) max_delta = d;
      alpha[i] = a_new;
      new_sum += a_new;
    }

    // Rescale to absorb numerical drift back to sum = gamma.
    if (new_sum > T{}) {
      const T scale = gamma / new_sum;
      for (std::size_t i = 0; i < N; ++i) alpha[i] *= scale;
    }

    if (max_delta < opts.tol) {
      ++iter;
      stats.converged = true;
      break;
    }
  }

  stats.iterations = iter;
  return stats;
}

}  // namespace

// ============================================================================
// Pointer-based control form.
// ============================================================================
template <typename T>
DominantSetStats dominant_set(
    const T* A, const T* b, const T* beta,
    std::size_t N,
    T* alpha_inout,
    DominantSetWorkspace<T> ws,
    const DominantSetOptions<T>& opts) {

  if (N == 0 || alpha_inout == nullptr) {
    return DominantSetStats{};
  }

  // Allocate any workspace buffers the caller didn't supply.
  std::vector<T> owned_M;
  std::vector<T> owned_Malpha;
  T* M = ws.M;
  T* M_alpha = ws.M_alpha;
  if (M == nullptr) {
    owned_M.assign(N * N, T{});
    M = owned_M.data();
  }
  if (M_alpha == nullptr) {
    owned_Malpha.assign(N, T{});
    M_alpha = owned_Malpha.data();
  }

  // Initialize alpha (auto-detect uniform vs provided; optional jitter).
  init_alpha<T>(alpha_inout, N, opts.gamma, opts.init_epsilon, opts.seed);

  // Build M from A and beta.
  build_M<T>(A, beta, N, M);

  // Run replicator.
  return replicator_loop<T>(M, b, N, alpha_inout, M_alpha, opts);
}

// ============================================================================
// Span-based convenience form.
// ============================================================================
template <typename T>
DominantSetResult<T> dominant_set(
    std::span<const T> A,
    std::span<const T> b,
    std::span<const T> beta,
    std::size_t N,
    const DominantSetOptions<T>& opts) {

  DominantSetResult<T> result;
  result.alpha.assign(N, T{});
  if (N == 0) return result;

  // Defer to the pointer form; let it own the workspace internally.
  const DominantSetStats stats = dominant_set<T>(
      A.data(), b.data(), beta.data(),
      N,
      result.alpha.data(),
      DominantSetWorkspace<T>{},
      opts);

  result.iterations = stats.iterations;
  result.converged  = stats.converged;
  return result;
}

// ---------------------------------------------------------------------------
// Explicit instantiations (float, double).
// ---------------------------------------------------------------------------
template DominantSetStats dominant_set<float>(
    const float*, const float*, const float*, std::size_t,
    float*, DominantSetWorkspace<float>,
    const DominantSetOptions<float>&);
template DominantSetStats dominant_set<double>(
    const double*, const double*, const double*, std::size_t,
    double*, DominantSetWorkspace<double>,
    const DominantSetOptions<double>&);

template DominantSetResult<float> dominant_set<float>(
    std::span<const float>, std::span<const float>, std::span<const float>,
    std::size_t, const DominantSetOptions<float>&);
template DominantSetResult<double> dominant_set<double>(
    std::span<const double>, std::span<const double>, std::span<const double>,
    std::size_t, const DominantSetOptions<double>&);

}  // namespace binaryoptim
