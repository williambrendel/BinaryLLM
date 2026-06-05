#pragma once

// ============================================================================
// binarytrain/signatures/sentence_split.hpp
//
// Splits a stream of StreamTokens into sentence-shaped windows.
//
// The basic primitive `delimiter_ends_sentence(value)` answers a
// context-free question: does this delimiter's string contain '.',
// '!', or '?'? It is exposed for testing.
//
// The full detector `is_sentence_end(tokens, i, cfg)` examines the
// delimiter together with neighboring tokens and applies several
// heuristics:
//
//   1. The delim must contain a terminator ('.', '!', '?').
//   2. If ',' or ';' appears AFTER the first terminator in the delim
//      (e.g., ".,", ".;"), it's mid-sentence punctuation, not a
//      boundary (handles "i.e.,").
//   3. If the IMMEDIATELY preceding token is a Word in
//      cfg.abbreviations (e.g., "mr", "dr"), the period is part of
//      the abbreviation, not a sentence end.
//   4. If the immediately preceding token is a Word containing '.'
//      (acronym shape, e.g., "u.s.a"), the trailing period is part of
//      the acronym UNLESS followed by end-of-stream or a delimiter
//      carrying a newline (paragraph break).
//   5. If we recently saw '@' or '://' since the last whitespace AND
//      the NEXT token is a Word with no whitespace between, this
//      period is intra-URL/email ("foo@bar.com").
//
// `split_sentences` uses `is_sentence_end` with default config. A
// custom-config overload is provided for extending the abbreviation
// list. Ranges are half-open [start, end) and include the terminator.
// ============================================================================

#include "binarytrain/parts/tokenize.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace binarytrain::signatures {

// Half-open token-index range identifying one sentence.
struct SentenceRange {
  std::size_t start;  // inclusive index into the token stream
  std::size_t end;    // exclusive index; range covers tokens [start, end)
};

// Configuration controlling sentence-splitting heuristics.
struct SentenceSplitConfig {
  // Lowercase word values treated as abbreviations: a sentence-
  // terminator delimiter immediately following one of these is part of
  // the abbreviation rather than a sentence end. The default covers
  // common English titles, months, days, and short forms. Caller may
  // extend or replace.
  std::unordered_set<std::string> abbreviations = {
    // Titles and honorifics
    "mr", "mrs", "ms", "miss", "dr", "prof",
    "sr", "jr", "st", "sen", "rep", "gov", "pres",
    "gen", "rev", "hon", "capt", "cpl", "sgt",
    "lt", "col", "maj", "adm",
    // Months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
    // Days
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs",
    "fri", "sat", "sun",
    // Misc abbreviations
    "vs", "etc", "ave", "blvd", "rd", "mt", "ft",
    "inc", "ltd", "co", "corp", "dept", "no", "vol",
    "ch", "pg", "pp", "ed", "eds",
  };
};

// Context-free: does the delimiter value contain a sentence terminator
// (one of '.', '!', '?')? Exposed for testing the primitive layer.
bool delimiter_ends_sentence(std::string_view delim_value) noexcept;

// Context-aware sentence-end detection. See file header comment for the
// full rule set. Returns true iff token[i] is a real sentence boundary
// given the surrounding tokens and configuration.
bool is_sentence_end(
    std::span<const binarytrain::parts::StreamToken> tokens,
    std::size_t i,
    const SentenceSplitConfig& cfg) noexcept;

// Identify sentence boundaries using default configuration.
std::vector<SentenceRange> split_sentences(
    std::span<const binarytrain::parts::StreamToken> tokens);

// Identify sentence boundaries with custom configuration. Allows
// extending the abbreviation set for domain-specific corpora.
std::vector<SentenceRange> split_sentences(
    std::span<const binarytrain::parts::StreamToken> tokens,
    const SentenceSplitConfig& cfg);

}  // namespace binarytrain::signatures
