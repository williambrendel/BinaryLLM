#pragma once

// ============================================================================
// core/signatures/weight_table.hpp
//
// Per-typed-3F-bit surprisal weight table (WGT1) — the frozen integer
// multiplicities the inference path matches against (spec §7.2, §7.5.0).
//
//   w_e = log(N / df_e) = −log p_e        (IDF / surprisal of bit position e)
//
// where N = number of signatures streamed and df_e = how many of them have
// bit e set. Keyed per TYPED 3F bit position (0..3F), NOT per untyped piece:
// because L/R are OR-pooled over neighbors, a piece fires far more often in
// its L/R slots than in its C slot, so per-typed-bit counting makes
// w_C ≫ w_L,w_R emerge on its own (§7.5.2) — no explicit block weight.
//
// Surprisal is linearly scaled into integer multiplicities [1, cap] and
// frozen (df_e == 0 → 0). Downstream matching (weighted_dot / info_content
// in binarycore::sparse) then stays integer popcount, no float on the hot
// path.
// ============================================================================

#include "binary_vec.hpp"

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace core::signatures {

using binarycore::binary_vec::BinaryVecDynamic;

inline constexpr char kWgtMagic[4] = {'W', 'G', 'T', '1'};
inline constexpr std::uint32_t kWgtVersion = 1;
inline constexpr std::uint8_t kDefaultQuantCap = 8;

// Frozen table: one integer multiplicity per 3F bit position. Index the
// metric's weight argument by 3F bit position with weights.data().
struct SurprisalTable {
  std::uint32_t dim = 0;                // = 3F
  std::uint8_t quant_cap = kDefaultQuantCap;
  std::vector<std::uint16_t> weights;   // size == dim
};

// Streaming document-frequency accumulator over a signature stream. Bounded
// memory (dim counters) — no N², no F×F.
class SurprisalCounter {
 public:
  explicit SurprisalCounter(std::size_t dim) : df_(dim, 0) {}

  // Increment df for every set bit of `sig`; count one signature.
  void add(const BinaryVecDynamic& sig);

  std::size_t dim() const noexcept { return df_.size(); }
  std::uint64_t token_count() const noexcept { return n_; }
  const std::vector<std::uint64_t>& df() const noexcept { return df_; }

  // Freeze: w_e = log(N/df_e), linearly scaled into [1, cap]; df_e == 0 → 0.
  SurprisalTable finalize(std::uint8_t quant_cap = kDefaultQuantCap) const;

 private:
  std::vector<std::uint64_t> df_;
  std::uint64_t n_ = 0;
};

// WGT1 binary format (little-endian):
//   magic "WGT1" | version u32 | dim u32 | cap u8 | pad u8[3] | weights u16[dim]
void save_surprisal(const SurprisalTable& table, std::ostream& out);
SurprisalTable load_surprisal(std::istream& in);

}  // namespace core::signatures
