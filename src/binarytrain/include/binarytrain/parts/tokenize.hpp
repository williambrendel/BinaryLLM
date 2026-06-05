#pragma once

// ============================================================================
// binarytrain/parts/tokenize.hpp
//
// Stream tokenizer. Splits raw text into a sequence of words and
// delimiter tokens.
//
// Token categories:
//
//   word        — maximal run of letters/digits (lowercased)
//   delimiter   — whitespace or punctuation, with normalization:
//                   single space               → " "
//                   2+ spaces, tab, mixed indent → "\t" (indent class)
//                   single newline             → "\n"
//                   2+ newlines (with optional intervening whitespace)
//                                              → "\n\n" (paragraph class)
//                   "..." or longer dot run    → "..." (ellipsis class)
//                   2+ dashes                  → "--" (em-dash class)
//                   other punctuation runs     → kept literally
//
// The tokenizer makes no judgments about which tokens earn dictionary
// slots — that's the extractor's job. It produces every word and every
// delimiter the input contains, in stream order.
// ============================================================================

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace binarytrain::parts {

// One emitted token from tokenize_stream.
struct StreamToken {
  enum class Type : std::uint8_t {
    Word,       // value is the lowercased word
    Delimiter,  // value is the normalized delimiter
  };
  Type type;
  std::string value;
};

// Tokenize a raw UTF-8 string (ASCII letters lowercased; non-ASCII
// passed through as bytes — they're not classified as letters and
// therefore land in delimiter tokens or are silently dropped if they
// appear inside what should be a word, depending on the byte's class).
//
// Non-ASCII handling note: ASCII letters [A-Za-z], digits [0-9], spaces,
// tabs, newlines, and the standard ASCII punctuation set get full
// support. Anything outside is treated as "other punctuation" — it
// participates in delimiter runs but doesn't get any normalization.
// For French/Spanish corpora with accented characters, callers should
// pre-normalize (e.g., NFD-strip diacritics) before passing to this
// function.
std::vector<StreamToken> tokenize_stream(std::string_view raw);

// Lowercase ASCII letters in-place, others unchanged.
std::string ascii_lowercase(std::string_view s);

}  // namespace binarytrain::parts
