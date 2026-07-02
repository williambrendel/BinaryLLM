// ============================================================================
// core/signatures/src/word_encoder.cpp
// ============================================================================

#include "word_encoder.hpp"

#include "decomposer.hpp"

#include <algorithm>

namespace core::signatures {

std::vector<std::uint32_t> encode_word(
    const core::parts::PartDictionary& dict,
    std::string_view word) {
  auto ids = core::parts::decompose_word(dict, word);
  // Canonicalize: sort + dedup. Option B set semantics.
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

}  // namespace core::signatures
