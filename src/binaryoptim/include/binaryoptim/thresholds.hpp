#pragma once

// ============================================================================
// binaryoptim/thresholds.hpp
//
// Standalone threshold / extraction utilities for distributions on the
// simplex (typically the alpha vector returned by dominant_set).
//
// All functions here are pure: they take an alpha vector and return either
// a scalar threshold value or a set of indices. They do NOT modify the
// input and do not depend on any solver state. Caller composes them with
// dominant_set however it wants.
//
// Two families:
//
//   1. Adaptive thresholds (data-driven cutoffs).
//      - log_ratio_gap_threshold(alpha, min_log_gap)
//          Cut at the largest gap in log(alpha_i / alpha_{i+1}) that is at
//          least min_log_gap. Captures the "elbow" of a sorted distribution.
//      - entropy_threshold(alpha)
//          Keep ceil(exp(H)) candidates, where H is the Shannon entropy of
//          the normalized alpha distribution. Effective support size.
//
//   2. Fixed-k extraction.
//      - top_k_indices(alpha, k)
//          Return the indices of the k largest alpha values, sorted by
//          alpha descending. Useful for the embedder use case where we
//          want a fixed sparsity level independent of distribution shape.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace binaryoptim {

// ---------------------------------------------------------------------------
// log_ratio_gap_threshold
//
// Sort alpha descending, examine consecutive log-ratios log(a_i / a_{i+1}),
// pick the largest gap that exceeds min_log_gap. Return alpha[i+1] as the
// threshold — i.e. keep everything strictly above it.
//
// If no gap exceeds min_log_gap, return 0 (caller falls back to whatever
// static floor they have).
// ---------------------------------------------------------------------------
template <typename T>
T log_ratio_gap_threshold(const T* alpha, std::size_t n, T min_log_gap) {
  if (n == 0) return T{};

  // Copy + sort descending.
  std::vector<T> sorted(alpha, alpha + n);
  std::sort(sorted.begin(), sorted.end(), std::greater<T>());

  // Walk gaps. Skip non-positive entries (log undefined).
  T best_gap = T{};
  std::size_t best_idx = n;  // index of the larger element of the best pair
  for (std::size_t i = 0; i + 1 < n; ++i) {
    const T a = sorted[i];
    const T b = sorted[i + 1];
    if (a <= T{} || b <= T{}) break;
    const T gap = std::log(a) - std::log(b);
    if (gap > best_gap) {
      best_gap = gap;
      best_idx = i;
    }
  }

  if (best_gap < min_log_gap) return T{};
  return sorted[best_idx + 1];
}

template <typename T>
inline T log_ratio_gap_threshold(const std::vector<T>& alpha, T min_log_gap) {
  return log_ratio_gap_threshold<T>(alpha.data(), alpha.size(), min_log_gap);
}

// ---------------------------------------------------------------------------
// entropy_threshold
//
// Compute Shannon entropy H = -sum_i p_i log p_i over the normalized
// distribution p_i = alpha_i / sum(alpha). The "effective support size" is
// exp(H); we keep the top ceil(exp(H)) entries. Return the alpha value of
// the (k+1)-th largest entry as the threshold (i.e. keep entries strictly
// above it).
//
// If sum(alpha) is non-positive or n == 0, return 0.
// ---------------------------------------------------------------------------
template <typename T>
T entropy_threshold(const T* alpha, std::size_t n) {
  if (n == 0) return T{};
  T sum = T{};
  for (std::size_t i = 0; i < n; ++i) {
    if (alpha[i] > T{}) sum += alpha[i];
  }
  if (!(sum > T{})) return T{};

  T H = T{};
  for (std::size_t i = 0; i < n; ++i) {
    if (alpha[i] > T{}) {
      const T p = alpha[i] / sum;
      H -= p * std::log(p);
    }
  }

  std::size_t k = static_cast<std::size_t>(std::ceil(std::exp(H)));
  if (k == 0) k = 1;
  if (k >= n) return T{};

  // Sort descending, return the (k)-th largest (zero-indexed → element k-1
  // is the smallest we keep; element k is the largest we drop).
  std::vector<T> sorted(alpha, alpha + n);
  std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end(),
                   std::greater<T>());
  return sorted[k];
}

template <typename T>
inline T entropy_threshold(const std::vector<T>& alpha) {
  return entropy_threshold<T>(alpha.data(), alpha.size());
}

// ---------------------------------------------------------------------------
// top_k_indices
//
// Return the indices of the k largest entries of alpha, sorted by alpha
// descending. Stable for ties only insofar as std::sort is stable on
// equal keys (i.e. not stable; if you need deterministic tie-breaking,
// pre-perturb your alpha values).
// ---------------------------------------------------------------------------
template <typename T>
std::vector<std::size_t> top_k_indices(const T* alpha, std::size_t n,
                                       std::size_t k) {
  if (k > n) k = n;

  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), std::size_t{0});

  // Partial sort by alpha descending.
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [alpha](std::size_t a, std::size_t b) {
                      return alpha[a] > alpha[b];
                    });

  idx.resize(k);
  return idx;
}

template <typename T>
inline std::vector<std::size_t> top_k_indices(const std::vector<T>& alpha,
                                              std::size_t k) {
  return top_k_indices<T>(alpha.data(), alpha.size(), k);
}

}  // namespace binaryoptim
