#pragma once

// ============================================================================
// core/pass1/pass1.hpp
//
// Pass-1 dictionary learning (spec §4): Online Dictionary Learning that turns
// a signature stream into K binary codewords via seed → stream → batch-refresh,
// then freeze. This stage covers seeding + accumulation + refresh + the D cap;
// the incoherence penalty and hub-stripping/re-seed arrive in later stages.
//
// SPEC AUGMENTATION (shadow-float = accumulator). The spec's "shadow-float
// weights, hard binary forward, freeze" is realized here without a dense
// K×3F float matrix (~3 GB, infeasible). Each atom's soft weight is the
// surprisal-weighted, decayed accumulator b_k[e] = Σ_{members} w_e over the
// bits of the signatures that fired it; the hard binary forward is φ_k =
// top-D of b_k. This is streaming-friendly and keeps the learning-to-hash
// "soft evidence → hard freeze" pattern.
//
//   seed:      ridge-leverage sampling of warmup signatures off the FD factor
//              (spec §4.4). Each seed atom = the top-D surprisal-weighted bits
//              of the sampled signature.
//   observe:   α-threshold pursuit → for each fired atom, add the signature's
//              surprisal-weighted bits to b_k; bump n_k. Auto-refreshes every
//              refresh_every signatures.
//   refresh:   φ_k ← top-D bits of b_k (D cap by construction); decay b_k.
//   finalize:  a last refresh; the codebook is then the frozen dictionary.
//
// Everything is float-at-build (accumulators, leverage). Inference-time pursuit
// stays integer.
// ============================================================================

#include "codebook.hpp"
#include "frequent_directions.hpp"
#include "pursuit.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace core::pass1 {

// A signature as a sorted-unique list of 3F set positions.
using Signature = std::vector<std::uint32_t>;

// Flatten a BinaryVecDynamic signature to a sorted global position list.
Signature to_positions(
    const binarycore::binary_vec::BigSparseBinaryVecDynamic& v);

class Pass1Learner {
 public:
  struct Config {
    std::size_t K = 8192;             // dictionary size
    std::uint32_t D = 256;            // max codeword width
    std::size_t refresh_every = 4096; // batch atom-refresh cadence
    double decay = 0.5;               // accumulator decay per refresh
    // Incoherence (separation) weight. Realizes the spec's squared-overlap
    // penalty as a per-bit congestion cost at refresh: a candidate bit's
    // support must exceed λ·(number of other atoms already holding it), else
    // it is dropped — spreading shared bits thin (low-degree overlap graph).
    // Tied to α (co-tune); 0 disables separation. See spec §4.1/§7.1.
    double lambda = 0.5;
    // C-band (identity) emphasis. Per signature the OR-pooled L/R bands carry
    // hundreds of bits vs C's ~tens, so the surprisal weights alone let context
    // dominate and atoms cluster by resemblance, not identity — measured BER_C
    // precision collapses. Scaling the C band's weights (C = [F,2F)) makes
    // atoms align to token identity (the spec's optional explicit block weight,
    // §7.5.2). ~20 restores BER_C precision to ~0.95 on Alice; 1 = off.
    double c_band_boost = 1.0;
    // Cap on the reservoir of under-covered signatures used to re-seed dead
    // codewords (spec §4.4 periodic re-seed).
    std::size_t reseed_pool_cap = 4096;
    core::codebook::PursuitConfig pursuit{};
    std::uint64_t rng_seed = 0x9E3779B97F4A7C15ULL;
  };

  // weights: per-3F-bit surprisal multiplicities widened to uint32 (from the
  // uint16 WGT1 table) so full-signature info sums don't overflow.
  Pass1Learner(std::size_t dim, std::vector<std::uint32_t> weights, Config cfg);

  // Hub-stripping by promiscuity (spec §4.2): mark the `fraction` most-
  // promiscuous 3F bits (whose FD energy concentrates in the dominant "common"
  // direction — diffuse fillers) so they are excluded from codewords. Read off
  // the FD factor, O(dim·rank) — no F×F. Call before seed(); default: none.
  void strip_hubs(const core::sketch::FrequentDirections& fd, double fraction);

  // Seed K atoms via ridge-leverage sampling of `warmup` off the FD factor.
  void seed(const core::sketch::FrequentDirections& fd,
            const std::vector<Signature>& warmup);

  // Encode one signature (pursuit) and accumulate into the fired atoms;
  // auto-refreshes on the configured cadence.
  void observe(const Signature& sig);

  void refresh();    // recompute every atom from its accumulator now
  void finalize();   // final refresh; the codebook is frozen after this

  const core::codebook::Codebook& codebook() const noexcept { return cb_; }

  // Fraction of atoms that have fired at least once.
  double utilization() const;

  // Mean surprisal-weighted reconstruction recall over `sigs`:
  // weighted_dot(decode(pursuit(x)), x) / info_content(x), averaged.
  double mean_recall(const std::vector<Signature>& sigs) const;

 private:
  // Recompute every non-dead atom from its accumulator, penalizing bit
  // congestion (incoherence), then re-seed dead atoms from the reservoir.
  // do_decay applies exponential forgetting after.
  void apply_refresh_(bool do_decay);
  void refresh_atom_(std::size_t k, const std::vector<std::uint32_t>& bit_degree);
  void reseed_dead_();

  std::size_t dim_;
  std::vector<std::uint32_t> weights_;
  Config cfg_;
  core::codebook::Codebook cb_;

  std::vector<std::unordered_map<std::uint32_t, double>> acc_;  // b_k
  std::vector<double> nk_;                                      // firing mass
  std::vector<char> fired_ever_;
  std::vector<char> stripped_;                 // hub bits excluded from atoms
  std::vector<Signature> reseed_pool_;         // under-covered signatures (ring)
  std::size_t reseed_head_ = 0;
  std::size_t seen_ = 0;
};

}  // namespace core::pass1
