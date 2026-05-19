#include "binarycore/encoding/token_encoder.hpp"

#include <cctype>

namespace binarycore {

namespace {

constexpr bool is_ascii_letter(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
constexpr bool is_ascii_upper(char c) noexcept { return c >= 'A' && c <= 'Z'; }
constexpr bool is_ascii_lower(char c) noexcept { return c >= 'a' && c <= 'z'; }
constexpr bool is_ascii_digit(char c) noexcept { return c >= '0' && c <= '9'; }

constexpr char to_lower(char c) noexcept {
  return is_ascii_upper(c) ? static_cast<char>(c - 'A' + 'a') : c;
}

// Index of a vowel double (aa,ee,ii,oo,uu) → 0..4, or -1.
constexpr int vowel_double_index(char c) noexcept {
  switch (c) {
    case 'a': return 0;
    case 'e': return 1;
    case 'i': return 2;
    case 'o': return 3;
    case 'u': return 4;
    default: return -1;
  }
}

// Index of a consonant double → 0..10, or -1.
// Order matches the ConsonantDoublesStart layout: ll rr ss tt nn mm pp ff gg cc dd
constexpr int consonant_double_index(char c) noexcept {
  switch (c) {
    case 'l': return 0;
    case 'r': return 1;
    case 's': return 2;
    case 't': return 3;
    case 'n': return 4;
    case 'm': return 5;
    case 'p': return 6;
    case 'f': return 7;
    case 'g': return 8;
    case 'c': return 9;
    case 'd': return 10;
    default: return -1;
  }
}

constexpr uint64_t length_bucket(std::size_t len) noexcept {
  return len > 15 ? 15ULL : static_cast<uint64_t>(len);
}

bits::Caps classify_caps(std::string_view word) noexcept {
  bool any_letter = false;
  bool first_upper = false;
  bool any_upper = false;
  bool any_lower = false;
  bool first_seen = false;

  for (char c : word) {
    if (!is_ascii_letter(c)) continue;
    if (!first_seen) {
      first_upper = is_ascii_upper(c);
      first_seen = true;
    }
    any_letter = true;
    if (is_ascii_upper(c)) any_upper = true;
    if (is_ascii_lower(c)) any_lower = true;
  }

  if (!any_letter) return bits::Caps::None;
  if (any_upper && !any_lower) return bits::Caps::AllUpper;
  if (first_upper && !any_lower) return bits::Caps::AllUpper;
  if (any_upper && any_lower) {
    return first_upper ? bits::Caps::FirstUpper : bits::Caps::Mixed;
  }
  return bits::Caps::None;
}

}  // namespace

uint64_t encode_word(std::string_view word) noexcept {
  uint64_t v = 0;

  bool first_letter_set = false;
  char prev_lower = 0;

  for (char c : word) {
    if (is_ascii_letter(c)) {
      char lc = to_lower(c);
      int letter_idx = lc - 'a';

      v |= (1ULL << (bits::LetterPresenceStart + letter_idx));

      if (!first_letter_set) {
        v |= (static_cast<uint64_t>(letter_idx) << bits::FirstLetterStart);
        first_letter_set = true;
      }

      if (lc == prev_lower) {
        int vd = vowel_double_index(lc);
        if (vd >= 0) {
          v |= (1ULL << (bits::VowelDoublesStart + vd));
        } else {
          int cd = consonant_double_index(lc);
          if (cd >= 0) {
            v |= (1ULL << (bits::ConsonantDoublesStart + cd));
          }
        }
      }
      prev_lower = lc;
    } else if (is_ascii_digit(c)) {
      int digit = c - '0';
      v |= (1ULL << (bits::DigitPresenceStart + digit));
      prev_lower = 0;
    } else {
      prev_lower = 0;
    }
  }

  v |= (static_cast<uint64_t>(classify_caps(word)) << bits::CapsStart);
  v |= (length_bucket(word.size()) << bits::LengthBucketStart);
  return v;
}

uint64_t encode_symbol(bits::Symbol s) noexcept {
  uint64_t v = 0;
  v |= (static_cast<uint64_t>(s) << bits::SymbolIdStart);
  v |= (1ULL << bits::Type);
  return v;
}

uint64_t encode_symbol_char(char c) noexcept {
  switch (c) {
    case '.': return encode_symbol(bits::Symbol::Period);
    case ',': return encode_symbol(bits::Symbol::Comma);
    case '!': return encode_symbol(bits::Symbol::Exclamation);
    case '?': return encode_symbol(bits::Symbol::Question);
    case ';': return encode_symbol(bits::Symbol::Semicolon);
    case ':': return encode_symbol(bits::Symbol::Colon);
    case '\'': return encode_symbol(bits::Symbol::Apostrophe);
    case '"': return encode_symbol(bits::Symbol::Quote);
    case '(': return encode_symbol(bits::Symbol::LParen);
    case ')': return encode_symbol(bits::Symbol::RParen);
    case '[': return encode_symbol(bits::Symbol::LBracket);
    case ']': return encode_symbol(bits::Symbol::RBracket);
    case '{': return encode_symbol(bits::Symbol::LBrace);
    case '}': return encode_symbol(bits::Symbol::RBrace);
    case '/': return encode_symbol(bits::Symbol::Slash);
    case '\\': return encode_symbol(bits::Symbol::Backslash);
    case '-': return encode_symbol(bits::Symbol::Hyphen);
    case '_': return encode_symbol(bits::Symbol::Underscore);
    case '@': return encode_symbol(bits::Symbol::At);
    case '$': return encode_symbol(bits::Symbol::Dollar);
    case '&': return encode_symbol(bits::Symbol::Ampersand);
    case '#': return encode_symbol(bits::Symbol::Hash);
    case '+': return encode_symbol(bits::Symbol::Plus);
    case '*': return encode_symbol(bits::Symbol::Asterisk);
    case '~': return encode_symbol(bits::Symbol::Tilde);
    case '>': return encode_symbol(bits::Symbol::Greater);
    case '<': return encode_symbol(bits::Symbol::Less);
    case '=': return encode_symbol(bits::Symbol::Equals);
    case '%': return encode_symbol(bits::Symbol::Percent);
    case '|': return encode_symbol(bits::Symbol::Pipe);
    default: return encode_symbol(bits::Symbol::Unknown);
  }
}

}  // namespace binarycore
