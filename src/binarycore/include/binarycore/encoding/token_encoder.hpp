#pragma once

// ============================================================================
// binarycore/encoding/token_encoder.hpp  (v2)
// ----------------------------------------------------------------------------
// Bit 63 is the type flag:
//   - 0 → word token
//   - 1 → symbol token
//
// Word layout (bit 63 = 0):
//
//   bits 0-23   letter presence (24 bits)
//                 22 dedicated bits for: a b c d e f g h i k l m n o p r s t u v w y
//                 1  shared bit:        {x, z}
//                 1  shared bit:        {q, j}
//
//   bits 24-28  first letter (5 bits, value 0..25; full alphabet preserved
//                 because first-letter discrimination still distinguishes
//                 x from z and q from j even though their presence bits
//                 are shared)
//
//   bits 29-30  capitalization (2 bits)
//
//   bits 31-34  word length bucket, GRAY-CODED (4 bits)
//                 Adjacent lengths differ in exactly one bit, so morphological
//                 cousins (cat/cats, run/runs) have small Hamming distance in
//                 the length field. Non-adjacent lengths can still be far
//                 apart, which is the desired behavior (cat vs extraordinarily
//                 should NOT look similar by length).
//
//   bits 35-44  digit presence (10 bits, one per 0..9)
//
//   bits 45-48  vowel doubles (4 bits: aa ee ii oo) — dropped uu (essentially
//                 nonexistent in Latin-script languages)
//
//   bits 49-59  consonant doubles (11 bits: ll rr ss tt nn mm pp ff gg cc dd bb)
//
//   bits 60-62  ordering checksum (3 bits)
//                 Rotate-XOR hash over the lowercase letters in the order they
//                 appear. Distinguishes anagram-style collisions (same letter
//                 set, same first letter, same length, same doubles, different
//                 ordering — e.g. pares/peras/pesar/presa).
//
//   bit 63      type flag (0 for word)
//
// Symbol layout (bit 63 = 1):  unchanged from v1.
// ============================================================================

#include "binarycore/containers/binary_vec.hpp"

#include <cstdint>
#include <string_view>

namespace binarycore {

namespace bits {

// Type flag.
constexpr int Type = 63;

// Word layout offsets.
constexpr int LetterPresenceStart = 0;        // bits 0..23 (24 bits)
constexpr int FirstLetterStart = 24;          // bits 24..28 (5 bits)
constexpr int CapsStart = 29;                 // bits 29..30 (2 bits)
constexpr int LengthBucketStart = 31;         // bits 31..34 (4 bits, Gray code)
constexpr int DigitPresenceStart = 35;        // bits 35..44 (10 bits)
constexpr int VowelDoublesStart = 45;         // bits 45..48 (4 bits)
constexpr int ConsonantDoublesStart = 49;     // bits 49..59 (11 bits)
constexpr int OrderingChecksumStart = 60;     // bits 60..62 (3 bits)

// ----------------------------------------------------------------------------
// Letter-presence bit positions.
// ----------------------------------------------------------------------------
// 26 letters, 24 bits. 22 letters get dedicated bits; the rare-letter pairs
// {x, z} and {q, j} share one bit each.
//
// LetterBit[letter_index] returns the bit POSITION (0..23) within the
// letter-presence field where this letter contributes. Shared-pair letters
// return the same position, which is correct behavior — both light the same
// bit. The first-letter field disambiguates which one actually starts the
// word.
//
// Letter indices: 'a' = 0, 'b' = 1, ..., 'z' = 25.
// ----------------------------------------------------------------------------
constexpr int letter_bit_position(int letter_index) noexcept {
  // Layout: dedicated bits 0..21 for the 22 common letters, then 22 = {x,z}
  // shared bit, 23 = {q,j} shared bit. Letters are assigned in alphabetical
  // order over the dedicated portion, with the four rare letters routed to
  // the two shared bits.
  //
  // Mapping by letter index:
  //   a(0)→0   b(1)→1   c(2)→2   d(3)→3   e(4)→4   f(5)→5
  //   g(6)→6   h(7)→7   i(8)→8   j(9)→23  k(10)→9  l(11)→10
  //   m(12)→11 n(13)→12 o(14)→13 p(15)→14 q(16)→23 r(17)→15
  //   s(18)→16 t(19)→17 u(20)→18 v(21)→19 w(22)→20 x(23)→22
  //   y(24)→21 z(25)→22
  //
  // The lookup table makes this an O(1) check at compile time.
  switch (letter_index) {
    case 0:  return 0;   // a
    case 1:  return 1;   // b
    case 2:  return 2;   // c
    case 3:  return 3;   // d
    case 4:  return 4;   // e
    case 5:  return 5;   // f
    case 6:  return 6;   // g
    case 7:  return 7;   // h
    case 8:  return 8;   // i
    case 9:  return 23;  // j → shared {q,j}
    case 10: return 9;   // k
    case 11: return 10;  // l
    case 12: return 11;  // m
    case 13: return 12;  // n
    case 14: return 13;  // o
    case 15: return 14;  // p
    case 16: return 23;  // q → shared {q,j}
    case 17: return 15;  // r
    case 18: return 16;  // s
    case 19: return 17;  // t
    case 20: return 18;  // u
    case 21: return 19;  // v
    case 22: return 20;  // w
    case 23: return 22;  // x → shared {x,z}
    case 24: return 21;  // y
    case 25: return 22;  // z → shared {x,z}
    default: return 0;
  }
}

// Capitalization values (stored in CapsStart..CapsStart+1).
enum class Caps : uint64_t {
  None = 0b00,        // hello
  FirstUpper = 0b01,  // Hello
  Mixed = 0b10,       // hELLo, iPhone
  AllUpper = 0b11,    // HELLO
};

// Symbol layout offsets (unchanged from v1).
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

// ============================================================================
// Encoder API (unchanged signatures)
// ============================================================================

uint64_t encode_word(std::string_view word) noexcept;
uint64_t encode_symbol(bits::Symbol s) noexcept;
uint64_t encode_symbol_char(char c) noexcept;

constexpr bool is_symbol(uint64_t token) noexcept {
  return (token >> bits::Type) & 1ULL;
}
constexpr bool is_word(uint64_t token) noexcept {
  return !is_symbol(token);
}

inline BinaryVec64 to_vec(uint64_t token) noexcept {
  return BinaryVec64{{token}};
}

}  // namespace binarycore
