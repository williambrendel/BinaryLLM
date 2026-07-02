#pragma once

// ============================================================================
// core/signatures/word_encoder.hpp
//
// Encode a word into its Option B bag: the sorted-unique set of part
// IDs produced by decomposing the word against the dictionary.
//
// One algorithm — choose decomposition behavior by REORDERING THE
// DICT (see core::parts::reorder_dict_*). The encoder is strategy-
// agnostic.
//
// `word` is assumed to be lowercased ASCII. Callers reading raw text
// should lowercase first (the tokenizer does this at training time).
// ============================================================================

#include "dictionary.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace core::signatures {

std::vector<std::uint32_t> encode_word(
    const core::parts::PartDictionary& dict,
    std::string_view word);

}  // namespace core::signatures
