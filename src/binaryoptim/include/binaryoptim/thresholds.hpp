#pragma once

// ============================================================================
// binaryoptim/thresholds.hpp
//
// Adaptive cut primitives for sorted-descending sequences.
//
// Architecture:
//
//   Four small utilities, each computing one signal:
//
//     entropy_effective_count    — perplexity-based effective count
//     gap_ratio_effective_count  — sharpest qualifying log-ratio cliff
//     elbow_effective_count      — chord-distance argmax with significance
//                                  gate vs the median (robust to noise)
//     scaled_mm_effective_count  — factor times median-mass position,
//                                  saturating at n (slope-aware fallback)
//
//   One composite — the canonical adaptive prune for V-code attention:
//
//     adaptive_prune_count       — cascade: elbow first, scaled_mm fallback
//     adaptive_prune_threshold   — value form of the above
//
// The two old "rolled-into-one" cascades (entropy+ratio for the previous
// adaptive_prune, and elbow+scaled_mm for the short-lived elbow_prune) no
// longer exist as separate functions. Their building blocks are exposed as
// utilities; the canonical adaptive prune is the elbow+scaled_mm cascade.
// Anyone who wants the old entropy+ratio behavior can compose the two
// utilities manually:
//
//     auto cap = entropy_effective_count(v);
//     auto k   = gap_ratio_effective_count(v, 1.5, 1e-10, cap);
//
// Sorted-descending precondition. All functions assume the input is sorted
// descending. The trailing-zero stop in entropy and the noise-floor stop in
// gap_ratio rely on it. Violating the precondition produces mathematically
// valid output but generally meaningless cuts.
//
// Convention. Every count function returns k in [0, n] meaning "keep the
// first k elements". A return of n means "keep all" (no cut). A return of
// 0 means "drop everything" (only happens for empty or all-nonpositive
// input).
//
// Defaults — referenced by name so callers don't sprinkle magic numbers:
//   - kRatioMinGap    = 1.5        (gap_ratio_effective_count)
//   - kRatioEps       = 1e-10      (gap_ratio_effective_count)
//   - kElbowSig       = 0.01       (elbow_effective_count)
//   - kMMFactor       = 2.2        (scaled_mm_effective_count)
//   - kNoMaxCutIndex  = SIZE_MAX   (no cap)
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace binaryoptim {

// Defaults — referenced by name from call sites.
inline constexpr double kRatioMinGap = 1.5;
inline constexpr double kRatioEps = 1e-10;
inline constexpr double kElbowSig = 0.01;
inline constexpr double kMMFactor = 2.2;
inline constexpr std::size_t kNoMaxCutIndex = static_cast<std::size_t>(-1);

// =============================================================================
// Utilities — each computes one signal independently.
// =============================================================================

// -----------------------------------------------------------------------------
// entropy_effective_count
//
// Returns k = ceil(exp(H)) where H is the Shannon entropy of values in
// `sorted_desc[0..min(n, max_cut_index))` interpreted as an unnormalized
// probability mass. Stops at the first non-positive value.
//
// Captures the "broad shape" of the distribution: a delta returns 1, a
// uniform over k items returns k.
//
// Returns 0 only if the input is empty or contains only non-positive values.
// Otherwise returns a value in [1, l] where l is the count of positive
// values examined.
// -----------------------------------------------------------------------------
std::size_t entropy_effective_count(
    std::span<const double> sorted_desc,
    std::size_t max_cut_index = kNoMaxCutIndex) noexcept;

// -----------------------------------------------------------------------------
// gap_ratio_effective_count
//
// Walks consecutive pairs in `sorted_desc[0..min(n, max_cut_index))`,
// finds the steepest ratio prev/curr that clears `min_gap`, and reports
// the index at which that cliff sits. Stops at the first value <= eps
// (noise floor). When no qualifying cliff is found, returns the
// noise-floor stop position.
//
// Strict `>` on the running max-gap means ties resolve in favor of the
// earlier (higher-up) cliff.
//
// Was named `ratio_effective_count` in earlier versions. The new name
// matches the other utilities (X_effective_count) and clarifies that
// "gap" refers to ratios between consecutive values.
//
// Returns:
//   - 0 if the input is empty
//   - n if n < 2 (no pair to compare)
//   - otherwise a value in [1, min(n, max_cut_index)]
// -----------------------------------------------------------------------------
std::size_t gap_ratio_effective_count(
    std::span<const double> sorted_desc,
    double min_gap = kRatioMinGap,
    double eps = kRatioEps,
    std::size_t max_cut_index = kNoMaxCutIndex) noexcept;

// -----------------------------------------------------------------------------
// elbow_effective_count
//
// Geometric kneedle: computes the chord from (0, v[0]) to (n-1, v[n-1])
// and returns the interior position that maximizes the chord-to-curve
// distance, provided the distance is significant.
//
// Significance gate: the elbow distance must exceed `sig * median(v)`.
// Median is taken from the sorted-descending input in O(1). Using
// median (not mean and not y-range) makes the gate robust both to
// peak outliers (which inflate mean) and to noise-only distributions
// (which collapse y-range).
//
// Returns:
//   - n if n < 3 (degenerate)
//   - n if the input is flat (v[0] - v[n-1] <= epsilon)
//   - n if no significant elbow is found (chord curvature insufficient)
//   - the elbow position in [1, n-1] otherwise
//
// The "return n on no elbow" convention matches the other utilities —
// the caller can detect "no cut" by comparing the result to n.
// -----------------------------------------------------------------------------
std::size_t elbow_effective_count(
    std::span<const double> sorted_desc,
    double sig = kElbowSig) noexcept;

// -----------------------------------------------------------------------------
// scaled_mm_effective_count
//
// Returns min(ceil(factor * median_mass(v)), n) where median_mass is
// the smallest k such that the cumulative sum of v[0..k) reaches half
// the total. Saturates at n.
//
// Slope-aware: for steep linear decays, mass concentrates in the head,
// median_mass is small, so the cut is sparse. For shallow decays mass
// spreads out, median_mass approaches n/2, so the result saturates at
// n (no cut). Factor 2.2 puts the saturation boundary at MM = n/2.2,
// which covers all linear slopes <= 25% of the y-range.
//
// Returns:
//   - n if input is empty or all-zero (total mass is zero -> no cut)
//   - otherwise a value in [1, n]
// -----------------------------------------------------------------------------
std::size_t scaled_mm_effective_count(
    std::span<const double> sorted_desc,
    double factor = kMMFactor) noexcept;

// =============================================================================
// Composite — the canonical adaptive prune.
// =============================================================================

// -----------------------------------------------------------------------------
// adaptive_prune_count
//
// Cascade:
//   1. k = elbow_effective_count(v, sig);
//      If k < n (significant elbow), return k.
//   2. Else, return scaled_mm_effective_count(v, factor).
//
// This is the recommended cut for V-code attention substrates and any
// setting where a single data-derived break is desired without external
// caps. Catches:
//   - sharp cliffs at any position (elbow path)
//   - power-law / exponential natural elbows (elbow path)
//   - linear decays via slope-aware MM fallback
//   - uniform / near-uniform via saturation to n
// -----------------------------------------------------------------------------
std::size_t adaptive_prune_count(
    std::span<const double> sorted_desc,
    double sig = kElbowSig,
    double factor = kMMFactor) noexcept;

// -----------------------------------------------------------------------------
// adaptive_prune_threshold
//
// Value form: returns the boundary value at position k =
// adaptive_prune_count(...). If k == n (no cut), returns 0.0; otherwise
// returns sorted_desc[k] (the first dropped value).
// -----------------------------------------------------------------------------
double adaptive_prune_threshold(
    std::span<const double> sorted_desc,
    double sig = kElbowSig,
    double factor = kMMFactor) noexcept;

// =============================================================================
// Unsorted-input convenience adapters. These sort internally before
// delegating. Use the sorted primitives above for performance-critical
// paths.
// =============================================================================

// Returns indices into `values` for the top-k largest values, in
// descending order. Stable on ties. If k > n, returns all n indices.
// If k == 0 or `values` is empty, returns empty.
std::vector<std::size_t> top_k_indices(
    std::span<const double> values,
    std::size_t k);

// Value form of entropy_effective_count on unsorted input.
double entropy_threshold(std::span<const double> values);

// Value form of gap_ratio_effective_count on unsorted input. Input is
// the natural log of the minimum gap, i.e. min_gap = exp(min_log_gap).
double log_ratio_gap_threshold(
    std::span<const double> values,
    double min_log_gap) noexcept(false);

}  // namespace binaryoptim
