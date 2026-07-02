#pragma once

// ============================================================================
// core/codebook/pursuit.hpp
//
// α-threshold matching pursuit (spec §4.3): encode a signature into the SET of
// codewords it fires. Energy descent in an SDM — fire every codeword whose
// surprisal-weighted containment match ≥ α·(best match), subtract their
// coverage from the residual, repeat until the residual is covered or a budget
// is hit.
//
//   match(residual, φ_k) = weighted_dot(residual, φ_k) / info_content(φ_k)
//
// α is a rational alpha_num/alpha_den so the threshold test stays INTEGER
// (cross-multiplied) — no float, no exp on the forward path (spec §7.2).
// α→1 fires one codeword (hard/argmax); α<1 fires a graded set.
//
// Candidate atoms are found via the codebook's inverted index (only atoms
// sharing a bit with the residual), so this is sublinear in K — never an
// all-K scan (spec §7.3).
// ============================================================================

#include "codebook.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace core::codebook {

struct PursuitConfig {
  std::uint32_t alpha_num = 9;   // α = alpha_num / alpha_den (default 0.9)
  std::uint32_t alpha_den = 10;
  std::size_t max_fired = 32;    // budget on |fired|
};

// weights: per-3F-bit surprisal multiplicities (WGT1), size == codebook.dim().
// signature: sorted-unique 3F set positions. Returns fired atom ids, sorted.
std::vector<std::uint32_t> pursuit_encode(
    const Codebook& cb,
    const std::vector<std::uint32_t>& signature,
    const std::vector<std::uint16_t>& weights,
    const PursuitConfig& cfg = {});

}  // namespace core::codebook
