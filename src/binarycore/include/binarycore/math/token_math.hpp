#pragma once

// ============================================================================
// binarycore/math/token_math.hpp
//
// Type-aware math operations on TokenVec values produced by the token encoder.
//
// TokenVec inherits from BinaryVec64, so all generic math primitives on
// BinaryVec64 already work on TokenVec via the implicit upcast. These
// overloads add ONE piece of extra semantics: a cross-type comparison
// (a word-token vs a symbol-token) is treated as maximally distant —
// the two live in disjoint subspaces and should never register as similar.
//
//   hamming(word, symbol)     -> 64
//   dot(word, symbol)         -> 0
//   similarity(word, symbol)  -> 0
//
// Same-type Tokens fall through to the regular bit-XOR / bit-AND math.
//
// If you want the type guard rail bypassed (e.g. you're feeding a TokenVec
// into generic vector code that does not care which is word vs symbol),
// either rely on the implicit upcast to BinaryVec64 or call to_vec()
// explicitly to make the intent visible.
// ============================================================================

#include "binarycore/encoding/token_encoder.hpp"
#include "binarycore/math/popcount.hpp"

#include <cstdint>

namespace binarycore {

// Cross-type tokens are defined to be maximally distant (64 bits).
// Same-type tokens fall through to plain BinaryVec64 Hamming.
inline int hamming(const TokenVec& a, const TokenVec& b) noexcept {
  if (is_symbol(a) != is_symbol(b)) return 64;
  return __builtin_popcountll(a.data[0] ^ b.data[0]);
}

// Cross-type tokens have zero overlap by definition.
inline int dot(const TokenVec& a, const TokenVec& b) noexcept {
  if (is_symbol(a) != is_symbol(b)) return 0;
  return __builtin_popcountll(a.data[0] & b.data[0]);
}

// Jaccard-style similarity in [0, 1]. Returns 0 across types or when
// either operand is the empty token.
inline double similarity(const TokenVec& a, const TokenVec& b) noexcept {
  if (is_symbol(a) != is_symbol(b)) return 0.0;
  const int pa = __builtin_popcountll(a.data[0]);
  const int pb = __builtin_popcountll(b.data[0]);
  if (pa == 0 || pb == 0) return 0.0;
  const int inter = __builtin_popcountll(a.data[0] & b.data[0]);
  const int uni = pa + pb - inter;
  return static_cast<double>(inter) / static_cast<double>(uni);
}

}  // namespace binarycore
