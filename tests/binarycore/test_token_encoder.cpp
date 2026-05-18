#include "binarycore/token_encoder.hpp"
#include "doctest.h"

using namespace binarycore;

// Helper to extract bitfield value from encoded token.
static uint64_t get_field(uint64_t token, int start, int width) {
    uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
    return (token >> start) & mask;
}
static bool get_bit(uint64_t token, int pos) {
    return (token >> pos) & 1ULL;
}

TEST_CASE("type flag distinguishes words from symbols") {
    uint64_t w = encode_word("hello");
    uint64_t s = encode_symbol(bits::Symbol::Period);
    CHECK(is_word(w));
    CHECK_FALSE(is_symbol(w));
    CHECK(is_symbol(s));
    CHECK_FALSE(is_word(s));

    // Type flag forces large hamming between any word and any symbol.
    BinaryVec64 wv = to_vec(w);
    BinaryVec64 sv = to_vec(s);
    CHECK(wv.hamming(sv) >= 1);
}

TEST_CASE("letter presence: 'cat' sets c, a, t and no others") {
    uint64_t t = encode_word("cat");
    // c=2, a=0, t=19
    CHECK(get_bit(t, bits::LetterPresenceStart + 0));  // a
    CHECK(get_bit(t, bits::LetterPresenceStart + 2));  // c
    CHECK(get_bit(t, bits::LetterPresenceStart + 19)); // t
    // Other letters not set
    for (int i = 0; i < 26; ++i) {
        if (i == 0 || i == 2 || i == 19) continue;
        CHECK_FALSE(get_bit(t, bits::LetterPresenceStart + i));
    }
}

TEST_CASE("letter presence is case-insensitive") {
    CHECK(get_field(encode_word("CAT"), bits::LetterPresenceStart, 26) ==
          get_field(encode_word("cat"), bits::LetterPresenceStart, 26));
}

TEST_CASE("first letter field captures first alphabetic character") {
    CHECK(get_field(encode_word("cat"), bits::FirstLetterStart, 5) == 2); // c
    CHECK(get_field(encode_word("apple"), bits::FirstLetterStart, 5) == 0); // a
    CHECK(get_field(encode_word("Zebra"), bits::FirstLetterStart, 5) == 25); // z
}

TEST_CASE("capitalization classification") {
    CHECK(get_field(encode_word("hello"), bits::CapsStart, 2) ==
          static_cast<uint64_t>(bits::Caps::None));
    CHECK(get_field(encode_word("Hello"), bits::CapsStart, 2) ==
          static_cast<uint64_t>(bits::Caps::FirstUpper));
    CHECK(get_field(encode_word("HELLO"), bits::CapsStart, 2) ==
          static_cast<uint64_t>(bits::Caps::AllUpper));
    CHECK(get_field(encode_word("iPhone"), bits::CapsStart, 2) ==
          static_cast<uint64_t>(bits::Caps::Mixed));
}

TEST_CASE("length bucket") {
    CHECK(get_field(encode_word("a"), bits::LengthBucketStart, 4) == 1);
    CHECK(get_field(encode_word("hello"), bits::LengthBucketStart, 4) == 5);
    // 16+ chars all clamp to 15
    CHECK(get_field(encode_word("aaaaaaaaaaaaaaaa"), bits::LengthBucketStart, 4) == 15);
    CHECK(get_field(encode_word("aaaaaaaaaaaaaaaaaaaaa"), bits::LengthBucketStart, 4) == 15);
}

TEST_CASE("digit presence") {
    uint64_t t = encode_word("abc123");
    CHECK(get_bit(t, bits::DigitPresenceStart + 1));
    CHECK(get_bit(t, bits::DigitPresenceStart + 2));
    CHECK(get_bit(t, bits::DigitPresenceStart + 3));
    CHECK_FALSE(get_bit(t, bits::DigitPresenceStart + 0));
}

TEST_CASE("vowel doubles") {
    CHECK(get_bit(encode_word("book"), bits::VowelDoublesStart + 3));     // oo
    CHECK(get_bit(encode_word("feel"), bits::VowelDoublesStart + 1));     // ee
    CHECK(get_bit(encode_word("aardvark"), bits::VowelDoublesStart + 0)); // aa
    CHECK_FALSE(get_bit(encode_word("cat"), bits::VowelDoublesStart + 0));
}

TEST_CASE("consonant doubles") {
    // 'll' is index 0 in the consonant-double table
    CHECK(get_bit(encode_word("hello"), bits::ConsonantDoublesStart + 0)); // ll
    CHECK(get_bit(encode_word("better"), bits::ConsonantDoublesStart + 3)); // tt
    CHECK_FALSE(get_bit(encode_word("cat"), bits::ConsonantDoublesStart + 0));
}

TEST_CASE("doubles work across case ('LL' = 'll')") {
    CHECK(get_bit(encode_word("HELLO"), bits::ConsonantDoublesStart + 0));
    CHECK(get_bit(encode_word("hELLo"), bits::ConsonantDoublesStart + 0));
}

TEST_CASE("encoding is deterministic (same string → same token)") {
    CHECK(encode_word("hello") == encode_word("hello"));
    CHECK(encode_word("World") == encode_word("World"));
    CHECK(encode_word("hello") != encode_word("World"));
}

TEST_CASE("morphological variants are close in Hamming distance") {
    // The encoding was designed so plurals, conjugations etc. have small distance.
    auto h = [](const char* a, const char* b) {
        return to_vec(encode_word(a)).hamming(to_vec(encode_word(b)));
    };
    CHECK(h("book", "books") <= 4);
    CHECK(h("run", "running") <= 6);
    CHECK(h("walk", "walked") <= 6);
    // Semantically distant words → larger distance
    CHECK(h("cat", "philosophy") >= 10);
}

TEST_CASE("symbols all have type flag set") {
    for (int i = 0; i < 64; ++i) {
        uint64_t s = encode_symbol(static_cast<bits::Symbol>(i));
        CHECK(is_symbol(s));
    }
}

TEST_CASE("encode_symbol_char maps common characters correctly") {
    CHECK(encode_symbol_char('.') == encode_symbol(bits::Symbol::Period));
    CHECK(encode_symbol_char(',') == encode_symbol(bits::Symbol::Comma));
    CHECK(encode_symbol_char('@') == encode_symbol(bits::Symbol::At));
    CHECK(encode_symbol_char('$') == encode_symbol(bits::Symbol::Dollar));
    // Unmapped character → Unknown
    CHECK(encode_symbol_char('\x01') == encode_symbol(bits::Symbol::Unknown));
}

TEST_CASE("empty word encodes to all-zeros word token") {
    uint64_t t = encode_word("");
    CHECK(is_word(t));
    CHECK(t == 0);
}
