#pragma once

#include "binarycore/encoding/token_encoder.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace binarycore {

// ============================================================================
// Token
// ----------------------------------------------------------------------------
// A Token is the scanner's output record. It carries both the encoded binary
// vector and the original source text (used for debugging, collision
// analysis, and round-tripping). Production paths only consume `encoded`.
//
// Note the distinction from the underlying encoder type:
//   - TokenVec is the 64-bit bit-vector representation produced by the
//     encoder (a subclass of BinaryVec64 with word/symbol type semantics).
//   - Token (this struct) is a tokenizer-level record bundling that
//     TokenVec with the source text it came from.
// ============================================================================
struct Token {
  TokenVec encoded;   // 64-bit binary encoding (subclass of BinaryVec64)
  std::string text;   // original source text (empty for synthetic symbols
                      // like ParagraphBreak)

  bool is_word() const noexcept { return binarycore::is_word(encoded); }
  bool is_symbol() const noexcept { return binarycore::is_symbol(encoded); }
};

// ============================================================================
// Tokenization rules:
//   - Single space   -> skip (word separator only)
//   - 2+ spaces or tab -> emit Indent symbol
//   - 1 newline (\n, \r\n, \r) -> emit Newline symbol
//   - 2+ newlines    -> emit ParagraphBreak symbol
//   - Symbol char    -> emit corresponding symbol
//   - Letters/digits -> read run, emit Word token
//   - Other characters -> emit Unknown symbol
// ============================================================================
std::vector<Token> tokenize(std::string_view input);

}  // namespace binarycore
