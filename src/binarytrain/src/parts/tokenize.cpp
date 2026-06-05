// ============================================================================
// binarytrain/src/parts/tokenize.cpp
// ============================================================================

#include "binarytrain/parts/tokenize.hpp"

#include <cctype>

namespace binarytrain::parts {

namespace {

// Character classification helpers. ASCII-only by design.
constexpr bool is_letter(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
constexpr bool is_digit(unsigned char c) {
  return c >= '0' && c <= '9';
}
constexpr bool is_word_char(unsigned char c) {
  return is_letter(c) || is_digit(c);
}
constexpr bool is_space(unsigned char c) {
  return c == ' ';
}
constexpr bool is_tab(unsigned char c) {
  return c == '\t';
}
constexpr bool is_newline(unsigned char c) {
  return c == '\n' || c == '\r';
}
constexpr bool is_whitespace(unsigned char c) {
  return is_space(c) || is_tab(c) || is_newline(c);
}

inline char to_lower_ascii(unsigned char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A'))
                                : static_cast<char>(c);
}

// Returns the number of "prefix" characters at position p that form
// a leading sign and/or currency symbol for a numeric literal. Zero
// means "not a number prefix at this position".
//
// Recognized prefixes (each must be followed by a digit):
//   [+-]      sign only         → -5,  +5
//   $         currency only     → $5
//   [+-]$     sign + currency   → -$5, +$5
//
// The character at raw[p-1] (if p > 0) must be non-alphanumeric for
// the prefix to fire. This keeps `5+5` tokenized as [5, +, 5] (math
// operator preserved) instead of absorbing the `+` into a number.
std::size_t number_prefix_len(std::string_view raw, std::size_t p) {
  if (p >= raw.size()) return 0;
  if (p > 0 &&
      is_word_char(static_cast<unsigned char>(raw[p - 1]))) {
    return 0;
  }
  std::size_t k = 0;
  // Optional sign.
  if (p + k < raw.size()) {
    unsigned char c = static_cast<unsigned char>(raw[p + k]);
    if (c == '-' || c == '+') ++k;
  }
  // Optional currency.
  if (p + k < raw.size() && raw[p + k] == '$') ++k;
  // Must be followed by a digit.
  if (k > 0 && p + k < raw.size() &&
      is_digit(static_cast<unsigned char>(raw[p + k]))) {
    return k;
  }
  return 0;
}

// Classify a whitespace run into a normalized token value.
std::string classify_whitespace_run(std::string_view run) {
  if (run.empty()) return {};

  std::size_t newlines = 0;
  std::size_t other_ws = 0;
  for (unsigned char c : run) {
    if (c == '\n') ++newlines;
    else if (c == '\r') { /* ignored */ }
    else if (is_space(c) || is_tab(c)) ++other_ws;
  }

  if (newlines >= 2) return "\n\n";
  if (newlines == 1) return "\n";
  if (other_ws == 1 && run.size() == 1 && run[0] == ' ') return " ";
  return "\t";
}

// Classify a punctuation run. Special cases:
//   "..." or longer dot run → "..."  (ellipsis)
//   "--" or longer dash run → "--"   (em-dash class)
//   anything else           → kept literally
std::string classify_punctuation_run(std::string_view run) {
  if (run.empty()) return {};

  if (run.size() >= 3) {
    bool all_dots = true;
    for (char c : run) if (c != '.') { all_dots = false; break; }
    if (all_dots) return "...";
  }

  if (run.size() >= 2) {
    bool all_dashes = true;
    for (char c : run) if (c != '-') { all_dashes = false; break; }
    if (all_dashes) return "--";
  }

  return std::string(run);
}

enum class Category { Word, Whitespace, Punctuation };

Category classify(unsigned char c) {
  if (is_word_char(c)) return Category::Word;
  if (is_whitespace(c)) return Category::Whitespace;
  return Category::Punctuation;
}

}  // namespace

std::string ascii_lowercase(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) out.push_back(to_lower_ascii(c));
  return out;
}

// Tokenize the input. Maintains a single piece of cross-token state:
// `in_quote`, toggled by each apostrophe character in punctuation
// runs. Used by the trailing-apostrophe absorption rule.
//
// Per-character connector rules (applied inside a Word run):
//
//   `-` and `&`:
//     internal iff alphanumeric character on both sides.
//
//   `'` (apostrophe):
//     internal iff alphanumeric on right (Alice's, don't, rock'n'roll);
//     trailing-absorb iff alphanumeric on left, non-alnum on right,
//     AND !in_quote (mothers', boys', students').
//
//   `,` (comma):
//     internal iff digit on both sides (1,000; 2,4-d; 2,4,5-t;
//     European 3,14 decimals).
//
//   `.` (period, decimal + acronym rules):
//     (a) Decimal: internal iff digit on both sides (3.14, 0.5, 1.0).
//     (b) Acronym: internal iff right is alphanumeric AND left is a
//         letter that is itself "isolated" — preceded by start-of-
//         word-run or another `.` (U.S.A, i.e., A.B.A).
//
// Number-prefix rule (applied at the outer loop, before classify):
//
//   A `[+-]?\$?` sequence immediately followed by a digit — and not
//   itself preceded by an alphanumeric character — starts a word run
//   that includes the entire prefix. This absorbs signs and currency
//   into a single number token: -5, +5, $5, -$5, +$1,234.50, etc.
//   The boundary condition (raw[i-1] non-alnum) keeps `5+5` as
//   [5, +, 5] (math operator preserved as delimiter), and keeps
//   `mother$Earth` decomposed naturally.
//
//   Inside a punctuation run, we also break early when the next
//   position would match a number prefix — except when doing so
//   would split an all-dashes sequence (em-dash priority preserved:
//   `--5` stays as ["--", "5"], not as ["-", "-5"]).
std::vector<StreamToken> tokenize_stream(std::string_view raw) {
  std::vector<StreamToken> out;
  if (raw.empty()) return out;

  const std::size_t n = raw.size();
  std::size_t i = 0;
  bool in_quote = false;

  while (i < n) {
    const unsigned char start_c = static_cast<unsigned char>(raw[i]);
    const std::size_t num_prefix = number_prefix_len(raw, i);

    if (num_prefix > 0 || classify(start_c) == Category::Word) {
      // Word run, possibly leading with a number prefix.
      std::size_t j = (num_prefix > 0) ? (i + num_prefix) : (i + 1);
      while (j < n) {
        const unsigned char c = static_cast<unsigned char>(raw[j]);
        const unsigned char left = static_cast<unsigned char>(raw[j - 1]);
        const bool has_right = (j + 1 < n);
        const unsigned char right = has_right
            ? static_cast<unsigned char>(raw[j + 1])
            : static_cast<unsigned char>(0);

        if (is_word_char(c)) {
          ++j;
          continue;
        }
        if (c == '-' || c == '&') {
          if (has_right && is_word_char(right)) { ++j; continue; }
          break;
        }
        if (c == '\'') {
          if (has_right && is_word_char(right)) { ++j; continue; }
          if (!in_quote) { ++j; break; }
          break;
        }
        if (c == ',') {
          if (is_digit(left) && has_right && is_digit(right)) {
            ++j;
            continue;
          }
          break;
        }
        if (c == '.') {
          // Decimal: digit on both sides.
          if (is_digit(left) && has_right && is_digit(right)) {
            ++j;
            continue;
          }
          // Acronym: right alnum, left isolated letter.
          if (has_right && is_word_char(right) && is_letter(left)) {
            bool left_isolated;
            if (j == i + 1) {
              left_isolated = true;
            } else {
              left_isolated = (raw[j - 2] == '.');
            }
            if (left_isolated) {
              ++j;
              continue;
            }
          }
          break;
        }
        break;
      }
      std::string_view run(raw.data() + i, j - i);
      out.push_back({StreamToken::Type::Word, ascii_lowercase(run)});
      i = j;
    } else {
      // Whitespace or Punctuation: maximal same-category run.
      const Category cat = classify(start_c);
      std::size_t j = i + 1;
      while (j < n &&
             classify(static_cast<unsigned char>(raw[j])) == cat) {
        // In a punctuation run, break early if we're at the start of
        // a number prefix — but preserve em-dash priority: don't
        // break if every char from `i` to `j` (inclusive) is a dash,
        // because the punct classifier wants the full run for the
        // `--` em-dash output.
        if (cat == Category::Punctuation &&
            number_prefix_len(raw, j) > 0) {
          bool all_dashes_so_far = (raw[j] == '-');
          if (all_dashes_so_far) {
            for (std::size_t k = i; k < j; ++k) {
              if (raw[k] != '-') {
                all_dashes_so_far = false;
                break;
              }
            }
          }
          if (!all_dashes_so_far) break;
        }
        ++j;
      }
      std::string_view run(raw.data() + i, j - i);
      if (cat == Category::Whitespace) {
        out.push_back({StreamToken::Type::Delimiter,
                       classify_whitespace_run(run)});
      } else {
        out.push_back({StreamToken::Type::Delimiter,
                       classify_punctuation_run(run)});
        for (unsigned char c : run) {
          if (c == '\'') in_quote = !in_quote;
        }
      }
      i = j;
    }
  }

  return out;
}

}  // namespace binarytrain::parts
