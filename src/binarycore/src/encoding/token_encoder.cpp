#include "binarycore/encoding/token_encoder.hpp"

#include <cctype>
#include <cstdint>

namespace binarycore {

namespace {

constexpr bool is_ascii_letter(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
constexpr bool is_ascii_upper(char c) noexcept {
  return c >= 'A' && c <= 'Z';
}
constexpr bool is_ascii_lower(char c) noexcept {
  return c >= 'a' && c <= 'z';
}
constexpr bool is_ascii_digit(char c) noexcept {
  return c >= '0' && c <= '9';
}

constexpr char to_lower(char c) noexcept {
  return is_ascii_upper(c) ? static_cast<char>(c - 'A' + 'a') : c;
}

constexpr int vowel_double_index(char c) noexcept {
  switch (c) {
  case 'a':
    return 0;
  case 'e':
    return 1;
  case 'i':
    return 2;
  case 'o':
    return 3;
  default:
    return -1;
  }
}

constexpr int consonant_double_index(char c) noexcept {
  switch (c) {
  case 'l':
    return 0;
  case 'r':
    return 1;
  case 's':
    return 2;
  case 't':
    return 3;
  case 'n':
    return 4;
  case 'm':
    return 5;
  case 'p':
    return 6;
  case 'f':
    return 7;
  case 'g':
    return 8;
  case 'c':
    return 9;
  case 'd':
    return 10;
  case 'b':
    return 11;
  default:
    return -1;
  }
}

constexpr uint64_t length_bucket_gray(std::size_t len) noexcept {
  const uint64_t clamped = (len > 15) ? 15ULL : static_cast<uint64_t>(len);
  return clamped ^ (clamped >> 1);
}

bits::Caps classify_caps(std::string_view word) noexcept {
  bool any_letter = false;
  bool first_upper = false;
  bool any_upper = false;
  bool any_lower = false;
  bool first_seen = false;

  for (char c : word) {
    if (!is_ascii_letter(c))
      continue;
    if (!first_seen) {
      first_upper = is_ascii_upper(c);
      first_seen = true;
    }
    any_letter = true;
    if (is_ascii_upper(c))
      any_upper = true;
    if (is_ascii_lower(c))
      any_lower = true;
  }

  if (!any_letter)
    return bits::Caps::None;
  if (any_upper && !any_lower)
    return bits::Caps::AllUpper;
  if (first_upper && !any_lower)
    return bits::Caps::AllUpper;
  if (any_upper && any_lower) {
    return first_upper ? bits::Caps::FirstUpper : bits::Caps::Mixed;
  }
  return bits::Caps::None;
}

// ----------------------------------------------------------------------------
// Ordering checksum (3 bits) — FNV-1a-based.
//
// Why FNV-1a and not the previous rotate-XOR:
//   The original 3-bit rotate-XOR collapsed the 26-letter alphabet into 8
//   equivalence classes BEFORE mixing — letters whose (index+1) & 7 matched
//   contributed identically. That created systematic bias on real anagram
//   pairs like 'perverse/preserve' where the underlying letter sequence,
//   though different, shared the same equivalence-class profile.
//
// FNV-1a uses:
//   - a 32-bit accumulator (full mixing range)
//   - alphabet values 1..26 for letters, 27..36 for digits (no overlap)
//   - XOR + multiply-by-prime per character (cascades single-bit flips
//     across all 32 internal bits within a few rounds)
//   - mask to 3 bits only at the end
//
// Constants are the canonical 32-bit FNV-1a values:
//   offset basis  = 2166136261  (0x811C9DC5)
//   FNV prime     = 16777619    (0x01000193)
//
// We treat digits as part of the same alphanumeric stream. This means
// '132' and '213' produce different checksums (digit ORDER matters now),
// resolving the v2 digit-permutation collisions.
// ----------------------------------------------------------------------------
constexpr uint64_t ordering_checksum(std::string_view word) noexcept {
  uint32_t h = 2166136261u; // FNV-1a offset basis
  bool any = false;
  for (char c : word) {
    uint32_t v = 0;
    if (c >= 'a' && c <= 'z') {
      v = static_cast<uint32_t>(c - 'a') + 1u; // 1..26
    } else if (c >= 'A' && c <= 'Z') {
      v = static_cast<uint32_t>(c - 'A') + 1u; // 1..26 (case-insensitive)
    } else if (c >= '0' && c <= '9') {
      v = static_cast<uint32_t>(c - '0') + 27u; // 27..36 (digits)
    } else {
      continue; // skip non-alphanumeric characters
    }
    h ^= v;
    h *= 16777619u; // FNV-1a prime
    any = true;
  }
  if (!any)
    return 0; // preserve all-zeros for empty input
  return static_cast<uint64_t>(h >> 29) & 0x7ULL;
}

} // namespace

uint64_t encode_word(std::string_view word) noexcept {
  uint64_t v = 0;

  bool first_letter_set = false;
  char prev_lower = 0;

  for (char c : word) {
    if (is_ascii_letter(c)) {
      const char lc = to_lower(c);
      const int letter_idx = lc - 'a';
      const int bit_pos = bits::letter_bit_position(letter_idx);

      v |= (1ULL << (bits::LetterPresenceStart + bit_pos));

      if (!first_letter_set) {
        v |= (static_cast<uint64_t>(letter_idx) << bits::FirstLetterStart);
        first_letter_set = true;
      }

      if (lc == prev_lower) {
        const int vd = vowel_double_index(lc);
        if (vd >= 0) {
          v |= (1ULL << (bits::VowelDoublesStart + vd));
        } else {
          const int cd = consonant_double_index(lc);
          if (cd >= 0) {
            v |= (1ULL << (bits::ConsonantDoublesStart + cd));
          }
        }
      }
      prev_lower = lc;
    } else if (is_ascii_digit(c)) {
      const int digit = c - '0';
      v |= (1ULL << (bits::DigitPresenceStart + digit));
      prev_lower = 0;
    } else {
      prev_lower = 0;
    }
  }

  v |= (static_cast<uint64_t>(classify_caps(word)) << bits::CapsStart);
  v |= (length_bucket_gray(word.size()) << bits::LengthBucketStart);
  v |= (ordering_checksum(word) << bits::OrderingChecksumStart);

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
  case '.':
    return encode_symbol(bits::Symbol::Period);
  case ',':
    return encode_symbol(bits::Symbol::Comma);
  case '!':
    return encode_symbol(bits::Symbol::Exclamation);
  case '?':
    return encode_symbol(bits::Symbol::Question);
  case ';':
    return encode_symbol(bits::Symbol::Semicolon);
  case ':':
    return encode_symbol(bits::Symbol::Colon);
  case '\'':
    return encode_symbol(bits::Symbol::Apostrophe);
  case '"':
    return encode_symbol(bits::Symbol::Quote);
  case '(':
    return encode_symbol(bits::Symbol::LParen);
  case ')':
    return encode_symbol(bits::Symbol::RParen);
  case '[':
    return encode_symbol(bits::Symbol::LBracket);
  case ']':
    return encode_symbol(bits::Symbol::RBracket);
  case '{':
    return encode_symbol(bits::Symbol::LBrace);
  case '}':
    return encode_symbol(bits::Symbol::RBrace);
  case '/':
    return encode_symbol(bits::Symbol::Slash);
  case '\\':
    return encode_symbol(bits::Symbol::Backslash);
  case '-':
    return encode_symbol(bits::Symbol::Hyphen);
  case '_':
    return encode_symbol(bits::Symbol::Underscore);
  case '@':
    return encode_symbol(bits::Symbol::At);
  case '$':
    return encode_symbol(bits::Symbol::Dollar);
  case '&':
    return encode_symbol(bits::Symbol::Ampersand);
  case '#':
    return encode_symbol(bits::Symbol::Hash);
  case '+':
    return encode_symbol(bits::Symbol::Plus);
  case '*':
    return encode_symbol(bits::Symbol::Asterisk);
  case '~':
    return encode_symbol(bits::Symbol::Tilde);
  case '>':
    return encode_symbol(bits::Symbol::Greater);
  case '<':
    return encode_symbol(bits::Symbol::Less);
  case '=':
    return encode_symbol(bits::Symbol::Equals);
  case '%':
    return encode_symbol(bits::Symbol::Percent);
  case '|':
    return encode_symbol(bits::Symbol::Pipe);
  default:
    return encode_symbol(bits::Symbol::Unknown);
  }
}

} // namespace binarycore
