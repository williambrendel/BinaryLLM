#include "binarycore/encoding/token_encoder.hpp"
#include "binarycore/math/hamming.hpp"
#include "doctest.h"

using namespace binarycore;

static uint64_t get_field(uint64_t token, int start, int width) {
  uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
  return (token >> start) & mask;
}
static bool get_bit(uint64_t token, int pos) {
  return (token >> pos) & 1ULL;
}

// ============================================================================
// Type flag
// ============================================================================

TEST_CASE("type flag distinguishes words from symbols") {
  uint64_t w = encode_word("hello");
  uint64_t s = encode_symbol(bits::Symbol::Period);
  CHECK(is_word(w));
  CHECK_FALSE(is_symbol(w));
  CHECK(is_symbol(s));
  CHECK_FALSE(is_word(s));
  CHECK(hamming(to_vec(w), to_vec(s)) >= 1);
}

// ============================================================================
// Letter presence
// ============================================================================

TEST_CASE("letter presence: 'cat' sets bits for c, a, t") {
  uint64_t t = encode_word("cat");
  // Look up the bit positions for each letter via the mapping.
  CHECK(get_bit(t, bits::LetterPresenceStart + bits::letter_bit_position(0)));  // a
  CHECK(get_bit(t, bits::LetterPresenceStart + bits::letter_bit_position(2)));  // c
  CHECK(get_bit(t, bits::LetterPresenceStart + bits::letter_bit_position(19))); // t
}

TEST_CASE("letter presence is case-insensitive") {
  // The letter-presence field should be identical regardless of case.
  uint64_t a = encode_word("CAT");
  uint64_t b = encode_word("cat");
  CHECK(get_field(a, bits::LetterPresenceStart, 24) ==
        get_field(b, bits::LetterPresenceStart, 24));
}

TEST_CASE("paired letters {x,z} share their presence bit") {
  // 'x' and 'z' should set the same presence bit.
  uint64_t with_x = encode_word("axe");
  uint64_t with_z = encode_word("aze");  // contrived but valid
  // The {x,z} shared bit position is letter_bit_position(23) == letter_bit_position(25).
  CHECK(bits::letter_bit_position(23) == bits::letter_bit_position(25));
  // Both should have that bit lit.
  const int shared_pos = bits::LetterPresenceStart + bits::letter_bit_position(23);
  CHECK(get_bit(with_x, shared_pos));
  CHECK(get_bit(with_z, shared_pos));
}

TEST_CASE("paired letters {q,j} share their presence bit") {
  uint64_t with_q = encode_word("aqua");
  uint64_t with_j = encode_word("ajar");
  CHECK(bits::letter_bit_position(9) == bits::letter_bit_position(16));
  const int shared_pos = bits::LetterPresenceStart + bits::letter_bit_position(9);
  CHECK(get_bit(with_q, shared_pos));
  CHECK(get_bit(with_j, shared_pos));
}

TEST_CASE("first-letter field still distinguishes paired letters") {
  // 'x' starts → first_letter == 23. 'z' starts → first_letter == 25.
  // So 'xerox' and 'zebra' have different first-letter fields even though
  // both light the same letter-presence bit.
  CHECK(get_field(encode_word("xerox"), bits::FirstLetterStart, 5) == 23);
  CHECK(get_field(encode_word("zebra"), bits::FirstLetterStart, 5) == 25);
  CHECK(get_field(encode_word("queen"), bits::FirstLetterStart, 5) == 16);
  CHECK(get_field(encode_word("jump"),  bits::FirstLetterStart, 5) == 9);
}

// ============================================================================
// Capitalization
// ============================================================================

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

// ============================================================================
// Length bucket (Gray-coded)
// ============================================================================

// Compute expected Gray code: x ^ (x >> 1).
static uint64_t gray(uint64_t x) { return x ^ (x >> 1); }

TEST_CASE("length bucket uses Gray code") {
  CHECK(get_field(encode_word("a"), bits::LengthBucketStart, 4) == gray(1));
  CHECK(get_field(encode_word("hello"), bits::LengthBucketStart, 4) == gray(5));
  CHECK(get_field(encode_word("aaaaaaaaaaaaaaaa"), bits::LengthBucketStart, 4) == gray(15));
  CHECK(get_field(encode_word("aaaaaaaaaaaaaaaaaaaaa"), bits::LengthBucketStart, 4) == gray(15));
}

TEST_CASE("adjacent lengths differ in exactly one bit in the length field") {
  // The key property of Gray code: |length_i - length_j| == 1 → Hamming 1.
  for (int len = 1; len < 15; ++len) {
    const uint64_t a = gray(len);
    const uint64_t b = gray(len + 1);
    int hd = 0;
    for (int k = 0; k < 4; ++k) {
      if (((a >> k) & 1) != ((b >> k) & 1)) ++hd;
    }
    CHECK(hd == 1);
  }
}

// ============================================================================
// Digit presence
// ============================================================================

TEST_CASE("digit presence") {
  uint64_t t = encode_word("abc123");
  CHECK(get_bit(t, bits::DigitPresenceStart + 1));
  CHECK(get_bit(t, bits::DigitPresenceStart + 2));
  CHECK(get_bit(t, bits::DigitPresenceStart + 3));
  CHECK_FALSE(get_bit(t, bits::DigitPresenceStart + 0));
}

// ============================================================================
// Vowel and consonant doubles
// ============================================================================

TEST_CASE("vowel doubles") {
  CHECK(get_bit(encode_word("book"), bits::VowelDoublesStart + 3));      // oo
  CHECK(get_bit(encode_word("feel"), bits::VowelDoublesStart + 1));      // ee
  CHECK(get_bit(encode_word("aardvark"), bits::VowelDoublesStart + 0));  // aa
  CHECK_FALSE(get_bit(encode_word("cat"), bits::VowelDoublesStart + 0));
}

TEST_CASE("uu vowel double was dropped — no bit lit for 'vacuum'") {
  // vacuum has uu. In v1 this would have set a bit. In v2, that bit
  // doesn't exist — the field is only 4 bits wide.
  // We just confirm no vowel-double bit is incorrectly set for 'vacuum'.
  uint64_t t = encode_word("vacuum");
  // aa, ee, ii, oo should all be unset.
  for (int i = 0; i < 4; ++i) {
    CHECK_FALSE(get_bit(t, bits::VowelDoublesStart + i));
  }
}

TEST_CASE("consonant doubles") {
  CHECK(get_bit(encode_word("hello"), bits::ConsonantDoublesStart + 0));   // ll
  CHECK(get_bit(encode_word("better"), bits::ConsonantDoublesStart + 3));  // tt
  CHECK_FALSE(get_bit(encode_word("cat"), bits::ConsonantDoublesStart + 0));
}

TEST_CASE("doubles work across case") {
  CHECK(get_bit(encode_word("HELLO"), bits::ConsonantDoublesStart + 0));
  CHECK(get_bit(encode_word("hELLo"), bits::ConsonantDoublesStart + 0));
}

// ============================================================================
// Ordering checksum
// ============================================================================

TEST_CASE("ordering checksum differs for anagrams (most cases)") {
  // The point of the checksum: anagram pairs that previously collided should
  // now differ in the checksum field, separating them in encoding space.
  //
  // We can't guarantee EVERY anagram pair differs (3-bit hash has collisions),
  // but typical pairs should.
  auto cksum = [](const char* s) {
    return get_field(encode_word(s), bits::OrderingChecksumStart, 3);
  };

  // Several anagram pairs from our corpus collision analysis:
  // We expect at least most of these to differ.
  int differ = 0;
  if (cksum("tide") != cksum("tied")) ++differ;
  if (cksum("sign") != cksum("sing")) ++differ;
  if (cksum("dare") != cksum("dear")) ++differ;
  if (cksum("form") != cksum("from")) ++differ;
  if (cksum("blow") != cksum("bowl")) ++differ;
  // At least 3 of 5 should differ for the hash to be doing useful work.
  CHECK(differ >= 3);
}

TEST_CASE("ordering checksum is deterministic") {
  CHECK(encode_word("hello") == encode_word("hello"));
  // The checksum field specifically should match for repeated inputs.
  CHECK(get_field(encode_word("hello"), bits::OrderingChecksumStart, 3) ==
        get_field(encode_word("hello"), bits::OrderingChecksumStart, 3));
}

TEST_CASE("ordering checksum is order-sensitive") {
  // Anagrams MUST differ in their letter ordering, so the checksum should
  // usually (not always) reflect that. We test that the checksum is at least
  // not collapsing — different inputs produce different checksums sometimes.
  auto cksum = [](const char* s) {
    return get_field(encode_word(s), bits::OrderingChecksumStart, 3);
  };
  // 'abc' vs 'cba' vs 'bac' — at least two of three should differ.
  int distinct = 0;
  if (cksum("abc") != cksum("cba")) ++distinct;
  if (cksum("abc") != cksum("bac")) ++distinct;
  if (cksum("cba") != cksum("bac")) ++distinct;
  CHECK(distinct >= 1);
}

// ============================================================================
// Determinism and morphological-distance properties
// ============================================================================

TEST_CASE("encoding is deterministic") {
  CHECK(encode_word("hello") == encode_word("hello"));
  CHECK(encode_word("World") == encode_word("World"));
  CHECK(encode_word("hello") != encode_word("World"));
}

TEST_CASE("morphological variants are close in Hamming distance") {
  // With Gray-coded length, single-character morphological changes should now
  // have SMALLER Hamming distance than under v1 standard binary length.
  auto h = [](const char* a, const char* b) {
    return hamming(to_vec(encode_word(a)), to_vec(encode_word(b)));
  };
  CHECK(h("book", "books") <= 4);
  CHECK(h("run", "running") <= 8);
  CHECK(h("walk", "walked") <= 8);
  CHECK(h("cat", "philosophy") >= 10);
}

// ============================================================================
// Symbols (unchanged)
// ============================================================================

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
  CHECK(encode_symbol_char('\x01') == encode_symbol(bits::Symbol::Unknown));
}

TEST_CASE("empty word encodes to all-zeros word token") {
  uint64_t t = encode_word("");
  CHECK(is_word(t));
  CHECK(t == 0);
}
