#pragma once

// ============================================================================
// core/signatures/paragraph_split.hpp
//
// Splits a stream of StreamTokens into paragraph-shaped windows.
//
// A paragraph break is signaled by a delimiter token whose value is
// exactly "\n\n". The tokenizer (parts::tokenize_stream) normalizes any
// run of 2+ newlines (with optional intervening whitespace) to this
// canonical value, so equality comparison is sufficient.
//
// If a paragraph runs longer than cfg.max_tokens without an explicit
// "\n\n" delimiter, the splitter forces a split at that boundary. This
// caps per-paragraph work for input that lacks blank-line structure
// (e.g., one-line files, machine-generated text, single-block dumps).
//
// As with sentence_split, ranges are half-open [start, end). The
// terminator delimiter ("\n\n", when present) is included at the END of
// the preceding paragraph's range — matching the sentence_split
// convention where the sentence-terminator delimiter is part of the
// sentence it ends.
// ============================================================================

#include "tokenize.hpp"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace core::signatures {

// Half-open token-index range identifying one paragraph.
struct ParagraphRange {
  std::size_t start;  // inclusive index into the token stream
  std::size_t end;    // exclusive index; range covers tokens [start, end)
};

// Configuration controlling paragraph-splitting heuristics.
struct ParagraphSplitConfig {
  // Maximum tokens per paragraph. When a running paragraph reaches this
  // length without seeing a "\n\n" delimiter, a forced split occurs. The
  // forced split's boundary token is included in the preceding
  // paragraph's range (same convention as an explicit "\n\n" boundary).
  //
  // Default chosen to accommodate typical long-form text (articles,
  // sections) while keeping per-paragraph processing bounded.
  std::size_t max_tokens = 4096;
};

// Context-free: does the delimiter value mark a paragraph break? Returns
// true iff the value is exactly "\n\n" (the tokenizer's canonical form
// for 2+ newlines). Exposed for testing the primitive layer.
bool delimiter_ends_paragraph(std::string_view delim_value) noexcept;

// Identify paragraph boundaries using default configuration.
std::vector<ParagraphRange> split_paragraphs(
    std::span<const core::parts::StreamToken> tokens);

// Identify paragraph boundaries with custom configuration. Allows
// tuning the max-token cap for unusual corpora.
std::vector<ParagraphRange> split_paragraphs(
    std::span<const core::parts::StreamToken> tokens,
    const ParagraphSplitConfig& cfg);

}  // namespace core::signatures
