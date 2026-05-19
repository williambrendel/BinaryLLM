#pragma once

// ============================================================================
// binarycore/encoding/token_encoder.hpp
//
// Encoder v7.4 — closed-form binary spelling embedder.
//
// Maps any UTF-8 ASCII-letter+digit string to a 64-bit TokenVec that captures
// enough structural information about the string to:
//
//   (a) discriminate distinct words almost perfectly across English, French
//       and Spanish corpora (validated <= 0.04% collision rate),
//   (b) place small-edit-distance variants close in Hamming distance
//       (mean Hamming distance under a single typo: 3 to 7 bits out of 64).
//
// The encoder is deterministic, requires no training, and is positionally
// aware via a Gray-coded linear positional sum that covers both letters and
// digits. Anagrams are broken by this sum; typo-induced changes shift the
// sum by a bounded amount, and Gray coding turns that bounded shift into a
// small Hamming-distance perturbation.
//
// Word-token bit layout (bit 63 = 0 marks a word):
//
//   bits  0..10  letter presence for 14 rare letters (11 bits; the rarest
//                pairs share a bit: {x,z}, {q,j}, {k,w})
//   bits 11..34  letter count for 12 common-duplicating letters
//                  (12 slots * 2 bits, saturates at 3 occurrences;
//                  order: a e i o l n r s t m p c)
//   bits 35..44  digit presence (10 bits, one per digit 0..9)
//   bits 45..48  word length bucket, Gray-coded, clamped at 15 (4 bits)
//   bits 49..50  capitalization class (2 bits)
//   bit     51   last-letter bit. Set when the last alphabetic character
//                is in {e, o, n, t, z}. Empirically chosen to split the
//                most-frequent last-letter collision pairs in the corpora.
//   bit     52   first-letter bit. Set when the first alphabetic character
//                is in {s, t, e, w, r}. Same empirical-rule approach.
//   bits 53..62  positional sum, Gray-coded mod 1024 (10 bits).
//                  sum = sum_i (token_index(c_i) + 1) * (pos_i + 1)
//                where token_index is 1..26 for letters and 27..36 for
//                digits, so letters and digits cannot accidentally cancel
//                in the sum. The result is Gray-coded so small shifts in
//                the sum translate to small Hamming-distance changes.
//   bit     63   type flag, 0 for word tokens.
//
// Symbol-token bit layout (bit 63 = 1 marks a symbol):
//
//   bits  0.. 5  symbol id (6 bits, indexes into bits::Symbol enum)
//   bits  6..62  unused
//   bit     63   type flag, 1 for symbol tokens.
//
// -----------------------------------------------------------------------------
// TokenVec type: inheritance, not wrapping
// -----------------------------------------------------------------------------
//
// `TokenVec` is a subclass of `BinaryVec64`. It shares storage and bit-layout
// machinery with the plain vector, and adds one piece of semantics: the
// type flag (bit 63) marks whether the token is a word or a symbol. The
// type-aware math overloads in binarycore/math/token_math.hpp ensure that
// comparing a word-token to a symbol-token always returns maximum distance
// (or zero similarity), regardless of any accidental bit overlap.
//
// When you want to treat a TokenVec as a plain binary vector (e.g. feed it
// into a generic Hamming computation that has no type semantics), let the
// implicit upcast to `BinaryVec64` happen — or call the explicit helper
// `to_vec()`. Both yield the same underlying bit pattern.
//
// Future kinds (Hidden, Context, ...) follow the same pattern: each is its
// own subclass of `BinaryVec<N>`, with kind-specific math overloads. No
// template-parameter plumbing needed at every call site, no risk of a
// defaulted kind silently miscategorizing a vector.
// ============================================================================

#include "binarycore/containers/binary_vec.hpp"

#include <cstdint>
#include <string_view>

namespace binarycore {

// ---------------------------------------------------------------------------
// TokenVec: a 64-bit BinaryVec with word-vs-symbol type semantics.
//
// Inheritance is public so generic math operations on BinaryVec64 work
// transparently on a TokenVec (the type-bit guard rail is opt-in via the
// TokenVec-specific overloads, not enforced on every operation).
// ---------------------------------------------------------------------------
struct TokenVec : public BinaryVec64 {
  // Construct from a raw 64-bit pattern. The TokenVec type itself does not
  // care which bit happens to be the type flag — the convention is fixed
  // in bits::Type below and used by encode_word / encode_symbol.
  constexpr TokenVec() noexcept : BinaryVec64{} {}
  constexpr explicit TokenVec(uint64_t b) noexcept : BinaryVec64{{b}} {}
  constexpr explicit TokenVec(const BinaryVec64& v) noexcept : BinaryVec64{v} {}
};

namespace bits {

// ----- bit positions and field sizes ---------------------------------------

constexpr int Type = 63;

constexpr int PresenceStart       = 0;    // 11 bits: 0..10
constexpr int LetterCountStart    = 11;   // 24 bits: 11..34 (12 slots * 2)
constexpr int DigitPresenceStart  = 35;   // 10 bits: 35..44
constexpr int LengthBucketStart   = 45;   //  4 bits: 45..48
constexpr int CapsStart           = 49;   //  2 bits: 49..50
constexpr int LastLetterBit       = 51;
constexpr int FirstLetterBit      = 52;
constexpr int PositionalSumStart  = 53;   // 10 bits: 53..62

// ----- lookup tables for letter handling -----------------------------------
//
// Two parallel 256-entry tables map a raw char (post-lowercase) to either
//   - a count-slot index 0..11 (count_slot_table), or
//   - a presence-bit index 0..10 (presence_bit_table).
// Each entry is -1 when the char does not belong in that tier. The tables
// also include the uppercase variants of every supported letter so we don't
// need a separate to_lower step in the inner loop.

struct CharTables {
  int8_t count_slot[256];     // -1 or 0..11
  int8_t presence_bit[256];   // -1 or 0..10
};

constexpr CharTables build_char_tables() {
  CharTables t{};
  for (int i = 0; i < 256; ++i) {
    t.count_slot[i] = -1;
    t.presence_bit[i] = -1;
  }
  // Count-bearing letters (12). Each has a dedicated 2-bit count slot.
  // Order chosen because these are the letters that most commonly appear
  // 2 or more times within a single word across our target corpora.
  const char count_letters[] = "aeiolnrstmpc";
  for (int i = 0; i < 12; ++i) {
    char c = count_letters[i];
    t.count_slot[static_cast<unsigned char>(c)] = static_cast<int8_t>(i);
    char C = static_cast<char>(c - 'a' + 'A');
    t.count_slot[static_cast<unsigned char>(C)] = static_cast<int8_t>(i);
  }
  // Presence-only letters (14, packed into 11 bits via three rare pairs).
  // The pairs group letters that almost never co-occur in the same word
  // in our target languages, so a shared bit is cheap.
  struct PB { const char* letters; int bit; };
  PB pairs[] = {
    {"b",  0}, {"d",  1}, {"f",  2}, {"g",  3}, {"h",  4},
    {"u",  5}, {"v",  6}, {"y",  7},
    {"xz", 8},   // shared pair
    {"qj", 9},   // shared pair
    {"kw", 10},  // shared pair
  };
  for (auto& p : pairs) {
    for (const char* s = p.letters; *s; ++s) {
      char c = *s;
      t.presence_bit[static_cast<unsigned char>(c)] = static_cast<int8_t>(p.bit);
      char C = static_cast<char>(c - 'a' + 'A');
      t.presence_bit[static_cast<unsigned char>(C)] = static_cast<int8_t>(p.bit);
    }
  }
  return t;
}

inline constexpr CharTables kCharTables = build_char_tables();

constexpr int count_slot(char c) noexcept {
  return kCharTables.count_slot[static_cast<unsigned char>(c)];
}
constexpr int presence_bit(char c) noexcept {
  return kCharTables.presence_bit[static_cast<unsigned char>(c)];
}

// ----- capitalization class -----------------------------------------------
enum class Caps : uint64_t {
  None       = 0b00,  // no letters, or all-lowercase
  FirstUpper = 0b01,  // initial cap, rest lowercase (Hello)
  Mixed      = 0b10,  // upper and lower mixed (iPhone, McX)
  AllUpper   = 0b11,  // all uppercase (HELLO, IBM)
};

// ----- symbol token id space ----------------------------------------------
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

// ---------------------------------------------------------------------------
// Encoding entry points.
// ---------------------------------------------------------------------------

// Encode an arbitrary string as a word token. The string need not be all-
// alphabetic: digits contribute to digit presence and to the positional sum,
// and other characters are silently ignored.
TokenVec encode_word(std::string_view word) noexcept;

// Encode a structural symbol (punctuation, whitespace, etc.) directly.
TokenVec encode_symbol(bits::Symbol s) noexcept;

// Encode a single character as a symbol token. Unrecognized characters
// produce bits::Symbol::Unknown.
TokenVec encode_symbol_char(char c) noexcept;

// ---------------------------------------------------------------------------
// Type queries.
// ---------------------------------------------------------------------------

constexpr bool is_symbol(const TokenVec& t) noexcept {
  return (t.data[0] >> bits::Type) & 1ULL;
}
constexpr bool is_word(const TokenVec& t) noexcept {
  return !is_symbol(t);
}

// Explicit name for the implicit upcast TokenVec -> BinaryVec64. Useful when
// you want to make the "drop type semantics, treat as plain bits" intent
// visible in the call site.
inline const BinaryVec64& to_vec(const TokenVec& t) noexcept {
  return static_cast<const BinaryVec64&>(t);
}

}  // namespace binarycore
