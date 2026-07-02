// ============================================================================
// core/signatures/src/paragraph_split.cpp
// ============================================================================

#include "paragraph_split.hpp"

namespace core::signatures {

using core::parts::StreamToken;

bool delimiter_ends_paragraph(std::string_view delim_value) noexcept {
  // The tokenizer normalizes 2+ newlines (with optional intervening
  // whitespace) to exactly "\n\n", so equality is sufficient.
  return delim_value == "\n\n";
}

std::vector<ParagraphRange> split_paragraphs(
    std::span<const StreamToken> tokens) {
  return split_paragraphs(tokens, ParagraphSplitConfig{});
}

std::vector<ParagraphRange> split_paragraphs(
    std::span<const StreamToken> tokens,
    const ParagraphSplitConfig& cfg) {
  std::vector<ParagraphRange> out;
  if (tokens.empty()) return out;

  std::size_t start = 0;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto& tok = tokens[i];
    const bool is_break =
        tok.type == StreamToken::Type::Delimiter &&
        delimiter_ends_paragraph(tok.value);

    // running paragraph length, including the current token at index i
    const std::size_t running = i + 1 - start;
    const bool force_split = running >= cfg.max_tokens;

    if (is_break || force_split) {
      out.push_back(ParagraphRange{start, i + 1});
      start = i + 1;
    }
  }
  // Trailing tokens (after last boundary, or all tokens if none)
  // form a final unterminated paragraph.
  if (start < tokens.size()) {
    out.push_back(ParagraphRange{start, tokens.size()});
  }
  return out;
}

}  // namespace core::signatures
