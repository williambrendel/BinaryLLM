// ============================================================================
// binarycore/src/encoding/token_encoder.cpp
//
// Implementation of the v7.4 token encoder declared in token_encoder.hpp.
// See the header for the full bit-layout documentation.
// ============================================================================

#include "binarycore/encoding/token_encoder.hpp"

#include <array>
#include <cstdint>

namespace binarycore {

namespace {

// ---------------------------------------------------------------------------
// Character classification primitives.
// ---------------------------------------------------------------------------

constexpr bool is_ascii_letter(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
constexpr bool is_ascii_upper(char c) noexcept { return c >= 'A' && c <= 'Z'; }
constexpr bool is_ascii_digit(char c) noexcept { return c >= '0' && c <= '9'; }

constexpr char to_lower(char c) noexcept {
  return is_ascii_upper(c) ? static_cast<char>(c - 'A' + 'a') : c;
}

// ---------------------------------------------------------------------------
// Length bucket: clamp at 15, Gray-code so adjacent lengths differ in
// exactly one bit. An insertion or deletion will therefore flip a single
// bit in this field (most of the time).
// ---------------------------------------------------------------------------
constexpr uint64_t length_bucket_gray(std::size_t len) noexcept {
  const uint64_t clamped = (len > 15) ? 15ULL : static_cast<uint64_t>(len);
  return clamped ^ (clamped >> 1);
}

// ---------------------------------------------------------------------------
// Capitalization classifier. Returns one of the bits::Caps values.
//   - Words with no letters (e.g. "42") return None.
//   - A single-letter uppercase word ("A") is classified as AllUpper.
// ---------------------------------------------------------------------------
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
    else                   any_lower = true;
  }

  if (!any_letter)                                return bits::Caps::None;
  if (any_upper && !any_lower)                    return bits::Caps::AllUpper;
  if (first_upper && !any_lower)                  return bits::Caps::AllUpper;
  if (any_upper && any_lower) {
    return first_upper ? bits::Caps::FirstUpper : bits::Caps::Mixed;
  }
  return bits::Caps::None;
}

// ---------------------------------------------------------------------------
// First / last letter discriminator bits.
//
// These two bits each carry a single empirically-chosen rule. The letter
// sets were picked by enumerating the most-common collision pairs at the
// first or last position in the corpora and selecting the set that
// maximizes weighted graph cut (i.e. that separates the most pairs).
//
// The bits are not arbitrary hash bits — they encode structural information
// the encoder explicitly cares about, and they survive intact under any
// typo not touching the first or last letter respectively.
// ---------------------------------------------------------------------------
constexpr bool last_letter_bit_value(char lc) noexcept {
  switch (lc) {
    case 'e': case 'o': case 'n': case 't': case 'z':
      return true;
    default:
      return false;
  }
}

constexpr bool first_letter_bit_value(char lc) noexcept {
  switch (lc) {
    case 's': case 't': case 'e': case 'w': case 'r':
      return true;
    default:
      return false;
  }
}

}  // namespace

// ===========================================================================
// encode_word
//
// Single-pass scan of the input. Letters and digits both contribute to the
// 10-bit positional sum and to their respective presence/count fields;
// other characters are ignored.
// ===========================================================================
TokenVec encode_word(std::string_view word) noexcept {
  uint64_t v = 0;

  // Per-letter occurrence counts for the 12 count-bearing letters. The
  // 2-bit slot saturates at 3, so a uint8_t with an explicit cap is plenty.
  std::array<uint8_t, 12> counts{};

  // First / last letter trackers — only the first and last alphabetic chars.
  char first_letter_lower = 0;
  char last_letter_lower = 0;
  bool first_seen = false;

  // Positional sum state. The sum runs over both letters and digits with
  // index ranges 1..26 and 27..36 respectively, so a digit at the start of
  // the string and a letter at the same position cannot cancel each other.
  // any_token preserves the empty-string -> all-zeros invariant.
  uint32_t sum = 0;
  uint32_t pos = 1;
  bool any_token = false;

  for (char c : word) {
    if (is_ascii_letter(c)) {
      const char lc = to_lower(c);

      if (!first_seen) {
        first_letter_lower = lc;
        first_seen = true;
      }

      // Tier 1: count-bearing letters increment a 2-bit slot (cap at 3).
      // Tier 2: rare letters set a presence bit (possibly shared with a
      // partner in the same pair group).
      const int cs = bits::count_slot(lc);
      if (cs >= 0) {
        if (counts[cs] < 3) counts[cs]++;
      } else {
        const int pb = bits::presence_bit(lc);
        if (pb >= 0) {
          v |= (1ULL << (bits::PresenceStart + pb));
        }
      }

      last_letter_lower = lc;

      // Letter contribution to positional sum: token-index 1..26.
      sum += (static_cast<uint32_t>(lc - 'a') + 1u) * pos;
      pos++;
      any_token = true;

    } else if (is_ascii_digit(c)) {
      // Digit presence bit (one bit per digit 0..9).
      v |= (1ULL << (bits::DigitPresenceStart + (c - '0')));

      // Digit contribution to positional sum: token-index 27..36.
      sum += (static_cast<uint32_t>(c - '0') + 27u) * pos;
      pos++;
      any_token = true;
    }
    // All other characters are silently dropped.
  }

  // Pack accumulated 2-bit letter counts.
  for (int slot = 0; slot < 12; ++slot) {
    v |= (static_cast<uint64_t>(counts[slot])
          << (bits::LetterCountStart + slot * 2));
  }

  v |= (length_bucket_gray(word.size()) << bits::LengthBucketStart);
  v |= (static_cast<uint64_t>(classify_caps(word)) << bits::CapsStart);

  // Optional discriminator bits (only set when the rule fires).
  if (last_letter_lower != 0 && last_letter_bit_value(last_letter_lower)) {
    v |= (1ULL << bits::LastLetterBit);
  }
  if (first_letter_lower != 0 && first_letter_bit_value(first_letter_lower)) {
    v |= (1ULL << bits::FirstLetterBit);
  }

  // 10-bit Gray-coded positional sum. Truncate to 10 bits, then Gray-encode
  // so that small numerical shifts in the sum (the kind a typo causes)
  // translate to small Hamming-distance shifts in the encoded field.
  if (any_token) {
    const uint32_t s = sum & 0x3FFu;
    v |= (static_cast<uint64_t>(s ^ (s >> 1)) << bits::PositionalSumStart);
  }

  return TokenVec{v};
}

// ===========================================================================
// encode_symbol
//
// Symbols live in a disjoint bit subspace (bit 63 = 1). Their identity is a
// small enum packed into bits 0..5.
// ===========================================================================
TokenVec encode_symbol(bits::Symbol s) noexcept {
  uint64_t v = 0;
  v |= (static_cast<uint64_t>(s) << bits::SymbolIdStart);
  v |= (1ULL << bits::Type);
  return TokenVec{v};
}

// ===========================================================================
// encode_symbol_char
//
// Map a single character to a known symbol enum, falling back to Unknown.
// ===========================================================================
TokenVec encode_symbol_char(char c) noexcept {
  switch (c) {
    case '.':  return encode_symbol(bits::Symbol::Period);
    case ',':  return encode_symbol(bits::Symbol::Comma);
    case '!':  return encode_symbol(bits::Symbol::Exclamation);
    case '?':  return encode_symbol(bits::Symbol::Question);
    case ';':  return encode_symbol(bits::Symbol::Semicolon);
    case ':':  return encode_symbol(bits::Symbol::Colon);
    case '\'': return encode_symbol(bits::Symbol::Apostrophe);
    case '"':  return encode_symbol(bits::Symbol::Quote);
    case '(':  return encode_symbol(bits::Symbol::LParen);
    case ')':  return encode_symbol(bits::Symbol::RParen);
    case '[':  return encode_symbol(bits::Symbol::LBracket);
    case ']':  return encode_symbol(bits::Symbol::RBracket);
    case '{':  return encode_symbol(bits::Symbol::LBrace);
    case '}':  return encode_symbol(bits::Symbol::RBrace);
    case '/':  return encode_symbol(bits::Symbol::Slash);
    case '\\': return encode_symbol(bits::Symbol::Backslash);
    case '-':  return encode_symbol(bits::Symbol::Hyphen);
    case '_':  return encode_symbol(bits::Symbol::Underscore);
    case '@':  return encode_symbol(bits::Symbol::At);
    case '$':  return encode_symbol(bits::Symbol::Dollar);
    case '&':  return encode_symbol(bits::Symbol::Ampersand);
    case '#':  return encode_symbol(bits::Symbol::Hash);
    case '+':  return encode_symbol(bits::Symbol::Plus);
    case '*':  return encode_symbol(bits::Symbol::Asterisk);
    case '~':  return encode_symbol(bits::Symbol::Tilde);
    case '>':  return encode_symbol(bits::Symbol::Greater);
    case '<':  return encode_symbol(bits::Symbol::Less);
    case '=':  return encode_symbol(bits::Symbol::Equals);
    case '%':  return encode_symbol(bits::Symbol::Percent);
    case '|':  return encode_symbol(bits::Symbol::Pipe);
    default:   return encode_symbol(bits::Symbol::Unknown);
  }
}

}  // namespace binarycore
