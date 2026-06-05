// ============================================================================
// binarytrain/src/signatures/sentence_split.cpp
// ============================================================================

#include "binarytrain/signatures/sentence_split.hpp"

namespace binarytrain::signatures {

namespace {

using binarytrain::parts::StreamToken;

// Find the index of the first terminator char ('.', '!', '?'), or
// std::string_view::npos if none.
std::size_t find_first_terminator(std::string_view v) noexcept {
  for (std::size_t k = 0; k < v.size(); ++k) {
    if (v[k] == '.' || v[k] == '!' || v[k] == '?') return k;
  }
  return std::string_view::npos;
}

bool word_contains_dot(std::string_view v) noexcept {
  for (char c : v) {
    if (c == '.') return true;
  }
  return false;
}

bool delim_contains_newline(std::string_view v) noexcept {
  for (char c : v) {
    if (c == '\n') return true;
  }
  return false;
}

bool delim_contains_whitespace(std::string_view v) noexcept {
  for (char c : v) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return true;
  }
  return false;
}

// '@' as a standalone delim, or '://' embedded anywhere in the value.
bool delim_carries_url_marker(std::string_view v) noexcept {
  if (v == "@") return true;
  return v.find("://") != std::string_view::npos;
}

// Walk back from index i toward 0. Stop at the first whitespace-
// bearing delimiter (which marks the end of the current "word group")
// or the start of the stream. Returns true if any delimiter along the
// way carries a URL/email marker ('@' or '://').
bool in_url_or_email_span(
    std::span<const StreamToken> tokens, std::size_t i) noexcept {
  for (std::size_t j = i; j-- > 0; ) {
    if (tokens[j].type == StreamToken::Type::Delimiter) {
      if (delim_contains_whitespace(tokens[j].value)) return false;
      if (delim_carries_url_marker(tokens[j].value)) return true;
    }
    // Word tokens: keep walking back.
  }
  return false;
}

}  // namespace

bool delimiter_ends_sentence(std::string_view delim_value) noexcept {
  return find_first_terminator(delim_value) != std::string_view::npos;
}

bool is_sentence_end(
    std::span<const StreamToken> tokens,
    std::size_t i,
    const SentenceSplitConfig& cfg) noexcept {
  if (i >= tokens.size()) return false;
  const auto& tok = tokens[i];
  if (tok.type != StreamToken::Type::Delimiter) return false;

  // Rule 1: delimiter value must contain a terminator.
  const std::size_t first_term = find_first_terminator(tok.value);
  if (first_term == std::string_view::npos) return false;

  // Rule 2: continuing punctuation (',' or ';') after the first
  // terminator means mid-sentence (e.g., ".,", ".;").
  for (std::size_t k = first_term + 1; k < tok.value.size(); ++k) {
    if (tok.value[k] == ',' || tok.value[k] == ';') return false;
  }

  // Rules 3 and 4 require the IMMEDIATELY preceding token to be a
  // Word. If there's a delim (whitespace or other) between the word
  // and the period, neither rule applies.
  if (i > 0 && tokens[i - 1].type == StreamToken::Type::Word) {
    const std::string& prev = tokens[i - 1].value;

    // Rule 3: abbreviation. The period is part of the abbreviation.
    if (cfg.abbreviations.find(prev) != cfg.abbreviations.end()) {
      return false;
    }

    // Rule 4: acronym-shaped word (contains '.'). The trailing period
    // is part of the acronym unless we're at end-of-stream or the
    // next delim carries a newline (paragraph or line break).
    if (word_contains_dot(prev)) {
      if (i + 1 >= tokens.size()) return true;  // end of stream
      const auto& next = tokens[i + 1];
      if (next.type == StreamToken::Type::Delimiter &&
          delim_contains_newline(next.value)) {
        return true;
      }
      return false;
    }
  }

  // Rule 5: URL/email intra-domain period. Both conditions hold:
  //   (a) URL/email context — lookback to most recent whitespace
  //       saw '@' or '://'.
  //   (b) Next token is a Word with no whitespace separator.
  if (i + 1 < tokens.size() &&
      tokens[i + 1].type == StreamToken::Type::Word &&
      in_url_or_email_span(tokens, i)) {
    return false;
  }

  return true;
}

std::vector<SentenceRange> split_sentences(
    std::span<const StreamToken> tokens) {
  return split_sentences(tokens, SentenceSplitConfig{});
}

std::vector<SentenceRange> split_sentences(
    std::span<const StreamToken> tokens,
    const SentenceSplitConfig& cfg) {
  std::vector<SentenceRange> out;
  if (tokens.empty()) return out;

  std::size_t start = 0;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (is_sentence_end(tokens, i, cfg)) {
      out.push_back(SentenceRange{start, i + 1});
      start = i + 1;
    }
  }
  // Trailing tokens (after last boundary, or all tokens if none)
  // form a final unterminated sentence.
  if (start < tokens.size()) {
    out.push_back(SentenceRange{start, tokens.size()});
  }
  return out;
}

}  // namespace binarytrain::signatures
