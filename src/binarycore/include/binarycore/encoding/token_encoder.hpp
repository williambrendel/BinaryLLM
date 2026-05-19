#pragma once

// v7.4: identical layout to v7.2, but the positional sum now folds in digit
// characters as well (with index 27 + (c - '0')). This breaks digit anagrams
// like 132/213/321 that were colliding in v7.2.

#include "binarycore/containers/binary_vec.hpp"

#include <cstdint>
#include <string_view>

namespace binarycore {

namespace bits {

constexpr int Type = 63;

constexpr int PresenceStart = 0;
constexpr int LetterCountStart = 11;
constexpr int DigitPresenceStart = 35;
constexpr int LengthBucketStart = 45;
constexpr int CapsStart = 49;
constexpr int LastLetterBit = 51;
constexpr int FirstLetterBit = 52;
constexpr int PositionalSumStart = 53;

constexpr int count_slot(char lc) noexcept {
  switch (lc) {
    case 'a': return 0;
    case 'e': return 1;
    case 'i': return 2;
    case 'o': return 3;
    case 'l': return 4;
    case 'n': return 5;
    case 'r': return 6;
    case 's': return 7;
    case 't': return 8;
    case 'm': return 9;
    case 'p': return 10;
    case 'c': return 11;
    default:  return -1;
  }
}

constexpr int presence_bit(char lc) noexcept {
  switch (lc) {
    case 'b': return 0;
    case 'd': return 1;
    case 'f': return 2;
    case 'g': return 3;
    case 'h': return 4;
    case 'u': return 5;
    case 'v': return 6;
    case 'y': return 7;
    case 'x': case 'z': return 8;
    case 'q': case 'j': return 9;
    case 'k': case 'w': return 10;
    default:  return -1;
  }
}

enum class Caps : uint64_t {
  None = 0b00, FirstUpper = 0b01, Mixed = 0b10, AllUpper = 0b11,
};

constexpr int SymbolIdStart = 0;

enum class Symbol : uint64_t {
  Period = 0,         Comma = 1,         Exclamation = 2,   Question = 3,
  Semicolon = 4,      Colon = 5,         Apostrophe = 6,    Quote = 7,
  LParen = 8,         RParen = 9,        LBracket = 10,     RBracket = 11,
  LBrace = 12,        RBrace = 13,       Newline = 14,      Indent = 15,
  Slash = 16,         Backslash = 17,    Hyphen = 18,       Underscore = 19,
  At = 20,            Dollar = 21,       Ampersand = 22,    Hash = 23,
  Plus = 24,          Asterisk = 25,     Tilde = 26,        Greater = 27,
  Less = 28,          Equals = 29,       Percent = 30,      Pipe = 31,
  ParagraphBreak = 32,
  Unknown = 63,
};

}  // namespace bits

uint64_t encode_word(std::string_view word) noexcept;
uint64_t encode_symbol(bits::Symbol s) noexcept;
uint64_t encode_symbol_char(char c) noexcept;

constexpr bool is_symbol(uint64_t token) noexcept {
  return (token >> bits::Type) & 1ULL;
}
constexpr bool is_word(uint64_t token) noexcept { return !is_symbol(token); }

inline BinaryVec64 to_vec(uint64_t token) noexcept {
  return BinaryVec64{{token}};
}

}  // namespace binarycore
