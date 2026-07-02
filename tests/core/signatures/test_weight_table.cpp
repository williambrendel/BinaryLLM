// ============================================================================
// tests/core/signatures/test_weight_table.cpp
//
// Contracts for the WGT1 surprisal weight table (spec §7.2, §7.5.0):
// document-frequency counting, surprisal quantization, WGT1 round-trip, and
// the per-typed-bit C-emphasis property (rarer bit → higher multiplicity).
// ============================================================================

#include "weight_table.hpp"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

#include "doctest.h"

using binarycore::binary_vec::BinaryVecDynamic;
using core::signatures::load_surprisal;
using core::signatures::save_surprisal;
using core::signatures::SurprisalCounter;
using core::signatures::SurprisalTable;

namespace {

// Build a signature of dimension `dim` with the given global set positions.
BinaryVecDynamic sig(std::size_t dim, std::vector<std::uint32_t> positions) {
  BinaryVecDynamic v(dim);
  constexpr std::size_t cs = BinaryVecDynamic::chunk_size;
  for (std::uint32_t g : positions) {
    v.chunks[g / cs].data.push_back(static_cast<std::uint16_t>(g % cs));
  }
  for (auto& c : v.chunks) std::sort(c.data.begin(), c.data.end());
  return v;
}

}  // namespace

TEST_CASE("SurprisalCounter: df and token counting") {
  SurprisalCounter c(10);
  c.add(sig(10, {0, 1, 2}));
  c.add(sig(10, {0, 1}));
  c.add(sig(10, {0}));
  CHECK(c.token_count() == 3);
  CHECK(c.df()[0] == 3);
  CHECK(c.df()[1] == 2);
  CHECK(c.df()[2] == 1);
  CHECK(c.df()[3] == 0);
}

TEST_CASE("SurprisalCounter::finalize quantization") {
  SurprisalCounter c(4);
  // bit 0 fires in all 4 tokens (common), bit 1 in one (rare), bit 3 never.
  c.add(sig(4, {0, 1}));
  c.add(sig(4, {0}));
  c.add(sig(4, {0}));
  c.add(sig(4, {0}));
  const SurprisalTable t = c.finalize(8);

  CHECK(t.dim == 4);
  CHECK(t.quant_cap == 8);
  CHECK(t.weights[0] == 1);         // common → floor multiplicity
  CHECK(t.weights[1] == t.quant_cap);  // rarest observed → cap
  CHECK(t.weights[1] > t.weights[0]);  // monotone in surprisal
  CHECK(t.weights[3] == 0);         // never fires → 0
}

TEST_CASE("SurprisalCounter::finalize: uniform corpus → all multiplicity 1") {
  SurprisalCounter c(3);
  // every bit fires in every token → w_e = 0 everywhere.
  c.add(sig(3, {0, 1, 2}));
  c.add(sig(3, {0, 1, 2}));
  const SurprisalTable t = c.finalize(8);
  CHECK(t.weights[0] == 1);
  CHECK(t.weights[1] == 1);
  CHECK(t.weights[2] == 1);
}

TEST_CASE("SurprisalCounter::finalize: empty stream → all zero") {
  SurprisalCounter c(5);
  const SurprisalTable t = c.finalize(8);
  CHECK(t.dim == 5);
  for (std::uint16_t w : t.weights) CHECK(w == 0);
}

TEST_CASE("per-typed-bit C-emphasis: OR-pooled context bit outweighs identity") {
  // Model the layout property: an L/R bit (fires often, via pooling) gets a
  // LOWER weight than a C bit that fires rarely. Positions here stand in for
  // (band, piece): pos 0 = a common pooled context bit, pos 1 = a rare C bit.
  SurprisalCounter c(2);
  for (int i = 0; i < 10; ++i) c.add(sig(2, {0}));  // context bit: df=10
  c.add(sig(2, {1}));                                // identity bit: df=1
  const SurprisalTable t = c.finalize(8);
  CHECK(t.weights[1] > t.weights[0]);
}

TEST_CASE("WGT1 round-trip") {
  SurprisalTable t;
  t.dim = 6;
  t.quant_cap = 5;
  t.weights = {0, 1, 2, 3, 4, 5};

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  save_surprisal(t, ss);

  // Real-header check: first four bytes are the WGT1 magic.
  const std::string bytes = ss.str();
  REQUIRE(bytes.size() >= 4);
  CHECK(bytes[0] == 'W');
  CHECK(bytes[1] == 'G');
  CHECK(bytes[2] == 'T');
  CHECK(bytes[3] == '1');

  const SurprisalTable r = load_surprisal(ss);
  CHECK(r.dim == t.dim);
  CHECK(r.quant_cap == t.quant_cap);
  CHECK(r.weights == t.weights);
}
