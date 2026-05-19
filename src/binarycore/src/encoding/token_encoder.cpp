#include "binarycore/encoding/token_encoder.hpp"

#include <array>
#include <cstdint>

namespace binarycore {

namespace {

constexpr bool is_ascii_letter(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
constexpr bool is_ascii_upper(char c) noexcept { return c >= 'A' && c <= 'Z'; }
constexpr bool is_ascii_digit(char c) noexcept { return c >= '0' && c <= '9'; }

constexpr char to_lower(char c) noexcept {
  return is_ascii_upper(c) ? static_cast<char>(c - 'A' + 'a') : c;
}

constexpr uint64_t length_bucket_gray(std::size_t len) noexcept {
  const uint64_t clamped = (len > 15) ? 15ULL : static_cast<uint64_t>(len);
  return clamped ^ (clamped >> 1);
}

bits::Caps classify_caps(std::string_view word) noexcept {
  bool any_letter = false, first_upper = false, any_upper = false, any_lower = false;
  bool first_seen = false;
  for (char c : word) {
    if (!is_ascii_letter(c)) continue;
    if (!first_seen) { first_upper = is_ascii_upper(c); first_seen = true; }
    any_letter = true;
    if (is_ascii_upper(c)) any_upper = true;
    else any_lower = true;
  }
  if (!any_letter) return bits::Caps::None;
  if (any_upper && !any_lower) return bits::Caps::AllUpper;
  if (first_upper && !any_lower) return bits::Caps::AllUpper;
  if (any_upper && any_lower) return first_upper ? bits::Caps::FirstUpper : bits::Caps::Mixed;
  return bits::Caps::None;
}

constexpr bool last_letter_bit_value(char lc) noexcept {
  switch (lc) {
    case 'e': case 'o': case 'n': case 't': case 'z': return true;
    default: return false;
  }
}

constexpr bool first_letter_bit_value(char lc) noexcept {
  switch (lc) {
    case 's': case 't': case 'e': case 'w': case 'r': return true;
    default: return false;
  }
}

}  // namespace

uint64_t encode_word(std::string_view word) noexcept {
  uint64_t v = 0;

  std::array<uint8_t, 12> counts{};
  char last_letter_lower = 0;
  char first_letter_lower = 0;
  bool first_seen = false;
  bool any_token = false;

  uint32_t sum = 0;
  uint32_t pos = 1;

  for (char c : word) {
    if (is_ascii_letter(c)) {
      const char lc = to_lower(c);
      if (!first_seen) { first_letter_lower = lc; first_seen = true; }
      any_token = true;
      const int cs = bits::count_slot(lc);
      if (cs >= 0) {
        if (counts[cs] < 3) counts[cs]++;
      } else {
        const int pb = bits::presence_bit(lc);
        if (pb >= 0) v |= (1ULL << (bits::PresenceStart + pb));
      }
      last_letter_lower = lc;

      // Positional sum index for letters: 1..26
      sum += (static_cast<uint32_t>(lc - 'a') + 1u) * pos;
      pos++;
    } else if (is_ascii_digit(c)) {
      any_token = true;
      v |= (1ULL << (bits::DigitPresenceStart + (c - '0')));

      // Positional sum index for digits: 27..36 (offset past letters)
      sum += (static_cast<uint32_t>(c - '0') + 27u) * pos;
      pos++;
    }
  }

  for (int slot = 0; slot < 12; ++slot) {
    v |= (static_cast<uint64_t>(counts[slot]) << (bits::LetterCountStart + slot * 2));
  }

  v |= (length_bucket_gray(word.size()) << bits::LengthBucketStart);
  v |= (static_cast<uint64_t>(classify_caps(word)) << bits::CapsStart);

  if (last_letter_lower != 0 && last_letter_bit_value(last_letter_lower)) {
    v |= (1ULL << bits::LastLetterBit);
  }
  if (first_letter_lower != 0 && first_letter_bit_value(first_letter_lower)) {
    v |= (1ULL << bits::FirstLetterBit);
  }

  if (any_token) {
    const uint32_t s = sum & 0x3FFu;
    v |= (static_cast<uint64_t>(s ^ (s >> 1)) << bits::PositionalSumStart);
  }

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
