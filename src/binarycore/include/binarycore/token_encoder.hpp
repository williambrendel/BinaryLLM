#pragma once

#include "binary_vec.hpp"

#include <cstdint>
#include <string_view>

namespace binarycore {

// ============================================================================
// Token encoding (64 bits per token).
// ----------------------------------------------------------------------------
// Bit 63 (the high bit) is the *type flag*:
//   - 0 → word token   (bits 0..62 use word layout)
//   - 1 → symbol token (bits 0..62 use symbol layout)
//
// The high bit guarantees that any word/symbol pair has Hamming distance ≥ 1
// just from the type flag, so they live in disjoint regions of the space.
// ============================================================================

namespace bits {

// --- Type flag ---
constexpr int Type = 63;

// --- Word layout (bit 63 = 0) ---
//
// dims 0-25:   letter presence (a..z)
// dims 26-30:  first letter (0..25 packed as 5-bit unsigned)
// dims 31-32:  capitalization (2 bits)
// dims 33-36:  word length bucket (4 bits, 0..15)
// dims 37-46:  digit presence (0..9)
// dims 47-51:  vowel doubles (aa, ee, ii, oo, uu)
// dims 52-62:  consonant doubles (ll, rr, ss, tt, nn, mm, pp, ff, gg, cc, dd)
//              (11 bits — we use the most common doubles in English/Latin)

constexpr int LetterPresenceStart = 0;       // bits 0..25
constexpr int FirstLetterStart    = 26;      // bits 26..30 (5 bits)
constexpr int CapsStart           = 31;      // bits 31..32 (2 bits)
constexpr int LengthBucketStart   = 33;      // bits 33..36 (4 bits)
constexpr int DigitPresenceStart  = 37;      // bits 37..46
constexpr int VowelDoublesStart   = 47;      // bits 47..51
constexpr int ConsonantDoublesStart = 52;    // bits 52..62

// Capitalization values stored in CapsStart..CapsStart+1.
enum class Caps : uint64_t {
    None        = 0b00, // hello
    FirstUpper  = 0b01, // Hello
    Mixed       = 0b10, // hELLo, iPhone
    AllUpper    = 0b11, // HELLO
};

// --- Symbol layout (bit 63 = 1) ---
//
// dims 0-5:    symbol identity (0..63 indexing into the symbol table below)
// dims 6-62:   reserved
//
// The symbol identity is a 6-bit unsigned integer indexing into:

constexpr int SymbolIdStart = 0;             // bits 0..5

enum class Symbol : uint64_t {
    Period             = 0,
    Comma              = 1,
    Exclamation        = 2,
    Question           = 3,
    Semicolon          = 4,
    Colon              = 5,
    Apostrophe         = 6,
    Quote              = 7,
    LParen             = 8,
    RParen             = 9,
    LBracket           = 10,
    RBracket           = 11,
    LBrace             = 12,
    RBrace             = 13,
    Newline            = 14,
    Indent             = 15, // tab OR 2+ consecutive spaces
    Slash              = 16,
    Backslash          = 17,
    Hyphen             = 18,
    Underscore         = 19,
    At                 = 20,
    Dollar             = 21,
    Ampersand          = 22,
    Hash               = 23,
    Plus               = 24,
    Asterisk           = 25,
    Tilde              = 26,
    Greater            = 27,
    Less               = 28,
    Equals             = 29,
    Percent            = 30,
    Pipe               = 31,
    ParagraphBreak     = 32, // \n\n+ (or platform variants)
    Unknown            = 63, // catch-all for unmapped characters

    // 33..62: reserved
};

} // namespace bits

// ============================================================================
// Encoder API.
// ----------------------------------------------------------------------------
// Encoding is a pure function of the input string. No state, no learning, no
// floats. The same string always produces the same 64-bit vector.
// ============================================================================

// Encode a word (ASCII letters and digits, mixed case OK).
// Empty strings, or strings with no encodable content, return an all-zeros
// word vector (still valid: bit 63 = 0).
uint64_t encode_word(std::string_view word) noexcept;

// Encode a symbol given its identity from the Symbol enum.
uint64_t encode_symbol(bits::Symbol s) noexcept;

// Encode a symbol given its character (for single-char symbols like '.').
// Returns encode_symbol(Symbol::Unknown) for characters not in the table.
uint64_t encode_symbol_char(char c) noexcept;

// Type query.
constexpr bool is_symbol(uint64_t token) noexcept {
    return (token >> bits::Type) & 1ULL;
}
constexpr bool is_word(uint64_t token) noexcept {
    return !is_symbol(token);
}

// Wrap a raw 64-bit token in a BinaryVec64 for similarity operations.
inline BinaryVec64 to_vec(uint64_t token) noexcept {
    return BinaryVec64({token});
}

} // namespace binarycore
