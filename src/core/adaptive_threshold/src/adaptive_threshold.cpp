// ============================================================================
// core/adaptive_threshold/src/adaptive_threshold.cpp
//
// Adaptive cut primitives. See header for full documentation.
// ============================================================================

#include "adaptive_threshold.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace core::adaptive_threshold {

// =============================================================================
// Utilities
// =============================================================================

// -----------------------------------------------------------------------------
// entropy_effective_count
// -----------------------------------------------------------------------------
//
// Walk forward from index 0, accumulating sum s and the unnormalized
// entropy term H_un = -sum(v * log(v)). Stop at the first non-positive
// value or at max_cut_index. Convert to Shannon entropy via
//
//   H(p) = log(s) + H_un / s
//
// where p_i = v_i / s. Return ceil(exp(H)) clamped to [1, l].
//
std::size_t entropy_effective_count(
    std::span<const double> sorted_desc,
    std::size_t max_cut_index) noexcept {
  const std::size_t n = sorted_desc.size();
  if (n == 0) return 0;

  const std::size_t cap = std::min<std::size_t>(
      std::max<std::size_t>(max_cut_index, 1), n);

  double s = 0.0;
  double H_un = 0.0;
  std::size_t l = 0;
  for (std::size_t i = 0; i < cap; ++i) {
    const double v = sorted_desc[i];
    if (!(v > 0.0)) break;
    s += v;
    H_un -= v * std::log(v);
    ++l;
  }
  if (l == 0) return 0;

  const double H = std::log(s) + H_un / s;
  const double exp_H = std::exp(H);
  if (!(exp_H > 1.0)) return 1;
  const std::size_t k = static_cast<std::size_t>(std::ceil(exp_H));
  return k > l ? l : k;
}

// -----------------------------------------------------------------------------
// gap_ratio_effective_count
// -----------------------------------------------------------------------------
//
// Walk consecutive pairs tracking the steepest qualifying log-ratio gap.
// On a new max gap that clears min_gap, update cut_idx. Stop at noise
// floor (curr <= eps). Final: max(min(cut_idx, i), 1).
//
std::size_t gap_ratio_effective_count(
    std::span<const double> sorted_desc,
    double min_gap,
    double eps,
    std::size_t max_cut_index) noexcept {
  const std::size_t n = sorted_desc.size();
  if (n == 0) return 0;
  if (n < 2) return n;

  if (!(eps >= 0.0)) eps = 0.0;
  const std::size_t cap = std::min<std::size_t>(
      std::max<std::size_t>(max_cut_index, 1), n);

  std::size_t cut_idx = cap;
  double max_gap = 0.0;
  double ac = sorted_desc[0];
  std::size_t i = 1;
  for (; i < cap; ++i) {
    if (!(ac > eps)) break;
    const double ap = ac;
    ac = sorted_desc[i];
    if (!(ac > 0.0)) {
      // Cliff from positive to <= 0: treat as +infinity gap.
      if (std::numeric_limits<double>::infinity() > max_gap) {
        max_gap = std::numeric_limits<double>::infinity();
        if (max_gap >= min_gap) cut_idx = i;
      }
      ++i;
      break;
    }
    const double g = ap / ac;
    if (g > max_gap) {
      max_gap = g;
      if (g >= min_gap) cut_idx = i;
    }
  }

  std::size_t result = std::min(cut_idx, i);
  if (result < 1) result = 1;
  return result;
}

// -----------------------------------------------------------------------------
// elbow_effective_count
// -----------------------------------------------------------------------------
//
// Trivial guards (n < 3 or flat input) return n. Otherwise compute the
// chord from (0, v[0]) to (n-1, v[n-1]) and find the interior position
// maximizing chord_distance(i) = chord(i) - v[i]. If the maximum
// distance exceeds sig * median(v), return that position; else return n
// to signal "no significant elbow".
//
// Median is read in O(1) thanks to the sorted-descending precondition.
// Using median (not mean) makes the gate robust to peak outliers and
// to noise-only inputs whose y-range collapses to noise magnitude.
//
std::size_t elbow_effective_count(
    std::span<const double> sorted_desc,
    double sig) noexcept {
  const std::size_t n = sorted_desc.size();
  if (n < 3) return n;

  const double y0 = sorted_desc[0];
  const double yN = sorted_desc[n - 1];
  const double y_range = y0 - yN;
  if (!(y_range > 1e-12)) return n;

  double med;
  if (n % 2 == 1) {
    med = sorted_desc[n / 2];
  } else {
    med = 0.5 * (sorted_desc[n / 2 - 1] + sorted_desc[n / 2]);
  }
  const double sig_threshold = sig * med;

  const double xN = static_cast<double>(n - 1);
  double best_diff = 0.0;
  std::size_t best_i = n;
  for (std::size_t i = 1; i < n - 1; ++i) {
    const double t = static_cast<double>(i) / xN;
    const double chord = y0 + t * (yN - y0);
    const double diff = chord - sorted_desc[i];
    if (diff > best_diff) {
      best_diff = diff;
      best_i = i;
    }
  }

  // sig * 0 = 0 is intentional: a peaked distribution whose median sits
  // at zero floor passes any positive elbow distance.
  if (best_diff >= sig_threshold && best_i < n) {
    return best_i;
  }
  return n;
}

// -----------------------------------------------------------------------------
// scaled_mm_effective_count
// -----------------------------------------------------------------------------
//
// Compute median_mass (smallest k where cumulative sum reaches total/2)
// and return min(ceil(factor * mm), n). Returns n on empty or all-zero
// inputs.
//
std::size_t scaled_mm_effective_count(
    std::span<const double> sorted_desc,
    double factor) noexcept {
  const std::size_t n = sorted_desc.size();
  if (n == 0) return 0;

  double total = 0.0;
  for (double v : sorted_desc) total += v;
  if (!(total > 0.0)) return n;

  const double half = total * 0.5;
  double cum = 0.0;
  std::size_t mm = n;
  for (std::size_t i = 0; i < n; ++i) {
    cum += sorted_desc[i];
    if (cum >= half) {
      mm = i + 1;
      break;
    }
  }

  const double scaled = std::ceil(factor * static_cast<double>(mm));
  if (!(scaled > 0.0)) return 1;
  if (scaled >= static_cast<double>(n)) return n;
  return static_cast<std::size_t>(scaled);
}

// =============================================================================
// Composite
// =============================================================================

// -----------------------------------------------------------------------------
// adaptive_prune_count
// -----------------------------------------------------------------------------
//
// Cascade. Try the elbow first; if no significant elbow, fall back to
// the slope-aware scaled-MM. Each utility is responsible for its own
// trivial / flat guards.
//
std::size_t adaptive_prune_count(
    std::span<const double> sorted_desc,
    double sig,
    double factor) noexcept {
  const std::size_t n = sorted_desc.size();
  const std::size_t k_elbow = elbow_effective_count(sorted_desc, sig);
  if (k_elbow < n) return k_elbow;
  return scaled_mm_effective_count(sorted_desc, factor);
}

// -----------------------------------------------------------------------------
// adaptive_prune_threshold
// -----------------------------------------------------------------------------
//
double adaptive_prune_threshold(
    std::span<const double> sorted_desc,
    double sig,
    double factor) noexcept {
  const std::size_t k = adaptive_prune_count(sorted_desc, sig, factor);
  if (k >= sorted_desc.size()) return 0.0;
  return sorted_desc[k];
}

// =============================================================================
// Unsorted-input convenience adapters
// =============================================================================

std::vector<std::size_t> top_k_indices(
    std::span<const double> values,
    std::size_t k) {
  const std::size_t n = values.size();
  if (n == 0 || k == 0) return {};

  std::vector<std::size_t> idx(n);
  for (std::size_t i = 0; i < n; ++i) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(),
      [&](std::size_t a, std::size_t b) {
        return values[a] > values[b];
      });
  if (k > n) k = n;
  idx.resize(k);
  return idx;
}

double entropy_threshold(std::span<const double> values) {
  const std::size_t n = values.size();
  if (n == 0) return 0.0;
  std::vector<double> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end(), std::greater<double>());
  const std::size_t k = entropy_effective_count(sorted);
  if (k >= n) return 0.0;
  return sorted[k];
}

double log_ratio_gap_threshold(
    std::span<const double> values,
    double min_log_gap) noexcept(false) {
  const std::size_t n = values.size();
  if (n == 0) return 0.0;
  std::vector<double> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end(), std::greater<double>());
  const double min_gap = std::exp(min_log_gap);
  const std::size_t k = gap_ratio_effective_count(sorted, min_gap);
  if (k >= n) return 0.0;
  return sorted[k];
}

}  // namespace core::adaptive_threshold
