#pragma once

#include "token_encoder.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace binarycore {

// ============================================================================
// Token
// ----------------------------------------------------------------------------
// A token carries both the encoded 64-bit vector and the original source text
// (for debugging, collision analysis, round-tripping). In production paths
// only `encoded` is used.
// ============================================================================
struct Token {
    uint64_t    encoded;    // 64-bit binary encoding
    std::string text;       // original source text (empty for synthetic symbols
                            // like paragraph_break)

    bool is_word() const noexcept { return binarycore::is_word(encoded); }
    bool is_symbol() const noexcept { return binarycore::is_symbol(encoded); }
};

// ============================================================================
// Tokenization rules:
//   - Single space  → skip (word separator only)
//   - 2+ spaces or tab → emit Indent symbol
//   - 1 newline (\n, \r\n, \r) → emit Newline symbol
//   - 2+ newlines  → emit ParagraphBreak symbol
//   - Symbol char  → emit corresponding symbol
//   - Letters/digits → read run, emit Word token
//   - Other characters → emit Unknown symbol
// ============================================================================
std::vector<Token> tokenize(std::string_view input);

} // namespace binarycore
