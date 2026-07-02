#pragma once

// ============================================================================
// binarycore::sparse::info_jaccard.hpp
//
// Surprisal-weighted set similarity over sparse binary vectors — the
// pipeline's native metric (spec §7.5.0). Operands are SORTED-ASCENDING,
// DEDUPED lists of set-bit positions (`const std::uint32_t*` + length);
// `weights` is a per-position weight table indexed by bit position.
//
// Templated on the weight type W so one implementation serves both paths:
//   - W = float   → training / table-construction / instrumentation
//   - W = integer → quantized-multiplicity inference path (§7.2): weighted
//                   matching is then an integer popcount, no float.
//
//   info_content(A)         = Σ_{e∈A} w_e
//   weighted_dot(A,B)       = Σ_{e∈A∩B} w_e            (two-pointer merge)
//   info_weighted_jaccard   = wdot / (I_A + I_B − wdot); 0 if wdot == 0
//   surprisal(p)            = −log p
//   rescale_weights_to_max  = scale so max weight → 1, ratios preserved
//
// Reduces to plain set Jaccard when weights are uniform (a unit test).
// For token-vs-aggregate matching (codeword mask, pooled bag) prefer the
// asymmetric weighted-containment form weighted_dot(x,φ)/info_content(φ)
// rather than the symmetric jaccard — see spec §4.3 / §7.5.0.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace binarycore::sparse {

// Σ w_e over the set positions in `a`. Empty / null → 0.
template <class W>
W info_content(const std::uint32_t* a, std::size_t na, const W* weights) {
  W acc = W{0};
  for (std::size_t i = 0; i < na; ++i) acc += weights[a[i]];
  return acc;
}

// Σ w_e over the intersection of sorted position lists `a` and `b`.
template <class W>
W weighted_dot(const std::uint32_t* a, std::size_t na,
               const std::uint32_t* b, std::size_t nb, const W* weights) {
  W acc = W{0};
  std::size_t i = 0, j = 0;
  while (i < na && j < nb) {
    if (a[i] < b[j]) {
      ++i;
    } else if (b[j] < a[i]) {
      ++j;
    } else {
      acc += weights[a[i]];
      ++i;
      ++j;
    }
  }
  return acc;
}

// Cached form: caller supplies I_a = info_content(a), I_b = info_content(b).
// Short-circuits to 0 on empty intersection regardless of I_a / I_b.
template <class W>
double info_weighted_jaccard(const std::uint32_t* a, std::size_t na, W ia,
                             const std::uint32_t* b, std::size_t nb, W ib,
                             const W* weights) {
  const W wdot = weighted_dot(a, na, b, nb, weights);
  if (wdot == W{0}) return 0.0;
  const W denom = ia + ib - wdot;  // ≥ wdot > 0 since ia,ib ≥ wdot
  return static_cast<double>(wdot) / static_cast<double>(denom);
}

// Uncached form: computes I_a and I_b internally.
template <class W>
double info_weighted_jaccard(const std::uint32_t* a, std::size_t na,
                             const std::uint32_t* b, std::size_t nb,
                             const W* weights) {
  const W wdot = weighted_dot(a, na, b, nb, weights);
  if (wdot == W{0}) return 0.0;
  const W ia = info_content(a, na, weights);
  const W ib = info_content(b, nb, weights);
  const W denom = ia + ib - wdot;
  return static_cast<double>(wdot) / static_cast<double>(denom);
}

// Shannon surprisal −log p. p ≤ 0 (or non-finite) → +∞ (defensive);
// p == 1 → 0. Natural log (bits are a caller-side rescale by 1/log 2).
template <class T>
T surprisal(T p) {
  if (!(p > T{0})) return std::numeric_limits<T>::infinity();
  return -std::log(p);
}

// Scale a weight container so its max element becomes 1, preserving all
// ratios. All-zero (or empty, or non-positive-max) container: no-op.
// Idempotent. Jaccard is invariant under this (uniform rescale).
template <class Container>
void rescale_weights_to_max(Container& weights) {
  using V = typename Container::value_type;
  V mx = V{0};
  for (const V& w : weights)
    if (w > mx) mx = w;
  if (!(mx > V{0})) return;
  for (V& w : weights) w /= mx;
}

}  // namespace binarycore::sparse
