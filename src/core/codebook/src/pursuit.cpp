// ============================================================================
// core/codebook/src/pursuit.cpp
// ============================================================================

#include "pursuit.hpp"

#include "info_jaccard.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace core::codebook {

std::vector<std::uint32_t> pursuit_encode(
    const Codebook& cb,
    const std::vector<std::uint32_t>& signature,
    const std::vector<std::uint32_t>& weights,
    const PursuitConfig& cfg) {
  std::vector<std::uint32_t> fired;
  if (signature.empty() || cb.size() == 0) return fired;

  const std::uint32_t* w = weights.data();

  // info_content(φ_k), computed lazily and reused across iterations.
  std::vector<std::int64_t> ic_cache(cb.size(), -1);
  auto ic_of = [&](std::uint32_t k) -> std::uint64_t {
    if (ic_cache[k] < 0) {
      const Atom& a = cb.atom(k);
      ic_cache[k] = static_cast<std::int64_t>(
          binarycore::sparse::info_content<std::uint32_t>(a.data(), a.size(), w));
    }
    return static_cast<std::uint64_t>(ic_cache[k]);
  };

  const std::uint64_t num = cfg.alpha_num;
  const std::uint64_t den = cfg.alpha_den;

  std::vector<char> is_fired(cb.size(), 0);
  std::vector<std::uint32_t> residual = signature;

  while (fired.size() < cfg.max_fired && !residual.empty()) {
    // Candidate atoms: any atom sharing a bit with the residual (inverted
    // index) that hasn't already fired.
    std::vector<std::uint32_t> cand;
    for (std::uint32_t e : residual) {
      const auto& lst = cb.atoms_with_bit(e);
      cand.insert(cand.end(), lst.begin(), lst.end());
    }
    std::sort(cand.begin(), cand.end());
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

    // Score candidates by weighted containment; track the best ratio.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> scored;  // (k, wdot)
    scored.reserve(cand.size());
    std::uint64_t best_wdot = 0, best_ic = 1;
    bool have_best = false;
    for (std::uint32_t k : cand) {
      if (is_fired[k]) continue;
      const Atom& a = cb.atom(k);
      const std::uint64_t wdot =
          binarycore::sparse::weighted_dot<std::uint32_t>(
              residual.data(), residual.size(), a.data(), a.size(), w);
      if (wdot == 0) continue;
      const std::uint64_t ic = ic_of(k);
      if (ic == 0) continue;
      scored.emplace_back(k, wdot);
      if (!have_best || wdot * best_ic > best_wdot * ic) {
        best_wdot = wdot;
        best_ic = ic;
        have_best = true;
      }
    }
    if (!have_best) break;

    // Fire all k with wdot/ic ≥ (num/den)·(best_wdot/best_ic), cross-multiplied.
    std::size_t newly = 0;
    for (auto [k, wdot] : scored) {
      const std::uint64_t ic = static_cast<std::uint64_t>(ic_cache[k]);
      if (wdot * den * best_ic >= num * best_wdot * ic) {
        is_fired[k] = 1;
        fired.push_back(k);
        ++newly;
        if (fired.size() >= cfg.max_fired) break;
      }
    }
    if (newly == 0) break;

    // Subtract coverage: residual = signature bits not covered by any fired
    // atom's mask.
    const std::vector<std::uint32_t> covered = cb.decode(fired);
    std::vector<std::uint32_t> next;
    std::set_difference(signature.begin(), signature.end(), covered.begin(),
                        covered.end(), std::back_inserter(next));
    residual.swap(next);
  }

  std::sort(fired.begin(), fired.end());
  return fired;
}

}  // namespace core::codebook
