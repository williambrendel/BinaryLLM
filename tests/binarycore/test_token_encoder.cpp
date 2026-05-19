#include "binarycore/encoding/token_encoder.hpp"
#include "binarycore/math/hamming.hpp"
#include "binarycore/math/token_math.hpp"
#include "doctest.h"

using namespace binarycore;

// Helpers for inspecting fields of a TokenVec.
static uint64_t get_field(const TokenVec& t, int start, int width) {
  uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
  return (t.data[0] >> start) & mask;
}
static bool get_bit(const TokenVec& t, int pos) {
  return (t.data[0] >> pos) & 1ULL;
}
static uint64_t slot_count(const TokenVec& t, char lc) {
  const int slot = bits::count_slot(lc);
  if (slot < 0) return 0;
  return (t.data[0] >> (bits::LetterCountStart + slot * 2)) & 0x3ULL;
}
static uint64_t gray(uint64_t x) { return x ^ (x >> 1); }

// -----------------------------------------------------------------------------
// Type bit
// -----------------------------------------------------------------------------

TEST_CASE("type flag distinguishes words from symbols") {
  CHECK(is_word(encode_word("hello")));
  CHECK(is_symbol(encode_symbol(bits::Symbol::Period)));
}

TEST_CASE("empty word encodes to all-zeros word token") {
  TokenVec t = encode_word("");
  CHECK(is_word(t));
  CHECK(t.data[0] == 0);
}

// -----------------------------------------------------------------------------
// Letter count for the 12 tracked common letters
// -----------------------------------------------------------------------------

TEST_CASE("letter count: cat has count=1 for a, c, t") {
  TokenVec t = encode_word("cat");
  CHECK(slot_count(t, 'a') == 1);
  CHECK(slot_count(t, 'c') == 1);
  CHECK(slot_count(t, 't') == 1);
  CHECK(slot_count(t, 'e') == 0);
}

TEST_CASE("letter count: hello has 1 e, 2 l, 1 o") {
  TokenVec t = encode_word("hello");
  CHECK(slot_count(t, 'e') == 1);
  CHECK(slot_count(t, 'l') == 2);
  CHECK(slot_count(t, 'o') == 1);
}

TEST_CASE("letter count saturates at 3") {
  CHECK(slot_count(encode_word("aaaaa"), 'a') == 3);
}

TEST_CASE("count is case-insensitive in the count field") {
  TokenVec lower = encode_word("cat");
  TokenVec upper = encode_word("CAT");
  uint64_t lower_counts = get_field(lower, bits::LetterCountStart, 24);
  uint64_t upper_counts = get_field(upper, bits::LetterCountStart, 24);
  CHECK(lower_counts == upper_counts);
}

// -----------------------------------------------------------------------------
// Presence for the 14 rare letters
// -----------------------------------------------------------------------------

TEST_CASE("presence for rare letters: h, b") {
  TokenVec t = encode_word("habit");
  CHECK(get_bit(t, bits::PresenceStart + bits::presence_bit('h')));
  CHECK(get_bit(t, bits::PresenceStart + bits::presence_bit('b')));
}

TEST_CASE("paired letters share their presence bit") {
  CHECK(bits::presence_bit('x') == bits::presence_bit('z'));
  CHECK(bits::presence_bit('q') == bits::presence_bit('j'));
  CHECK(bits::presence_bit('k') == bits::presence_bit('w'));
}

// -----------------------------------------------------------------------------
// Capitalization
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Length bucket
// -----------------------------------------------------------------------------

TEST_CASE("length bucket uses Gray code") {
  CHECK(get_field(encode_word("a"),     bits::LengthBucketStart, 4) == gray(1));
  CHECK(get_field(encode_word("hello"), bits::LengthBucketStart, 4) == gray(5));
  CHECK(get_field(encode_word("aaaaaaaaaaaaaaaa"),
                  bits::LengthBucketStart, 4) == gray(15));
}

TEST_CASE("Gray-coded adjacent lengths differ in exactly one bit") {
  for (int len = 1; len < 15; ++len) {
    uint64_t a = gray(len);
    uint64_t b = gray(len + 1);
    int hd = __builtin_popcountll(a ^ b);
    CHECK(hd == 1);
  }
}

// -----------------------------------------------------------------------------
// Digit presence
// -----------------------------------------------------------------------------

TEST_CASE("digit presence") {
  TokenVec t = encode_word("abc123");
  CHECK(get_bit(t, bits::DigitPresenceStart + 1));
  CHECK(get_bit(t, bits::DigitPresenceStart + 2));
  CHECK(get_bit(t, bits::DigitPresenceStart + 3));
  CHECK_FALSE(get_bit(t, bits::DigitPresenceStart + 0));
}

// -----------------------------------------------------------------------------
// First / last letter discriminator bits
// -----------------------------------------------------------------------------

TEST_CASE("last-letter bit set for {e, o, n, t, z}") {
  CHECK(get_bit(encode_word("home"), bits::LastLetterBit));   // e
  CHECK(get_bit(encode_word("hello"), bits::LastLetterBit));  // o
  CHECK(get_bit(encode_word("run"), bits::LastLetterBit));    // n
  CHECK(get_bit(encode_word("cat"), bits::LastLetterBit));    // t
  CHECK_FALSE(get_bit(encode_word("car"), bits::LastLetterBit));  // r
  CHECK_FALSE(get_bit(encode_word("data"), bits::LastLetterBit)); // a
}

TEST_CASE("first-letter bit set for {s, t, e, w, r}") {
  CHECK(get_bit(encode_word("super"), bits::FirstLetterBit));  // s
  CHECK(get_bit(encode_word("tap"), bits::FirstLetterBit));    // t
  CHECK_FALSE(get_bit(encode_word("apple"), bits::FirstLetterBit));  // a
  CHECK_FALSE(get_bit(encode_word("dog"), bits::FirstLetterBit));    // d
}

// -----------------------------------------------------------------------------
// Determinism / symbol encoding
// -----------------------------------------------------------------------------

TEST_CASE("encoding is deterministic") {
  CHECK(encode_word("hello").data[0] == encode_word("hello").data[0]);
  CHECK(encode_word("hello").data[0] != encode_word("world").data[0]);
}

TEST_CASE("symbols all have type flag set") {
  for (int i = 0; i < 64; ++i) {
    CHECK(is_symbol(encode_symbol(static_cast<bits::Symbol>(i))));
  }
}

TEST_CASE("encode_symbol_char maps known characters") {
  CHECK(encode_symbol_char('.').data[0] == encode_symbol(bits::Symbol::Period).data[0]);
  CHECK(encode_symbol_char(',').data[0] == encode_symbol(bits::Symbol::Comma).data[0]);
  CHECK(encode_symbol_char('@').data[0] == encode_symbol(bits::Symbol::At).data[0]);
  CHECK(encode_symbol_char('\x01').data[0] == encode_symbol(bits::Symbol::Unknown).data[0]);
}

// -----------------------------------------------------------------------------
// TokenVec math: cross-type guard rail
// -----------------------------------------------------------------------------

TEST_CASE("TokenVec hamming on different types returns 64") {
  TokenVec w = encode_word("hello");
  TokenVec s = encode_symbol(bits::Symbol::Period);
  CHECK(hamming(w, s) == 64);
  CHECK(hamming(s, w) == 64);
}

TEST_CASE("TokenVec hamming on same type returns popcount of XOR") {
  TokenVec a = encode_word("hello");
  TokenVec b = encode_word("world");
  int expected = __builtin_popcountll(a.data[0] ^ b.data[0]);
  CHECK(hamming(a, b) == expected);
}

TEST_CASE("TokenVec dot on different types returns 0") {
  TokenVec w = encode_word("hello");
  TokenVec s = encode_symbol(bits::Symbol::Period);
  CHECK(dot(w, s) == 0);
}

TEST_CASE("TokenVec similarity on cross-type tokens is 0") {
  TokenVec w = encode_word("hello");
  TokenVec s = encode_symbol(bits::Symbol::Period);
  CHECK(similarity(w, s) == 0.0);
}

TEST_CASE("TokenVec similarity of identical tokens is 1") {
  TokenVec a = encode_word("hello");
  CHECK(similarity(a, a) == doctest::Approx(1.0));
}

// -----------------------------------------------------------------------------
// TokenVec-as-BinaryVec64: generic vector math sees through inheritance
// -----------------------------------------------------------------------------

TEST_CASE("TokenVec can be used as a BinaryVec64 via implicit upcast") {
  TokenVec w = encode_word("hello");
  TokenVec s = encode_symbol(bits::Symbol::Period);
  // Generic hamming on BinaryVec64 has no type guard rail.
  const BinaryVec64& vw = to_vec(w);
  const BinaryVec64& vs = to_vec(s);
  int gh = hamming(vw, vs);
  CHECK(gh >= 0);
  CHECK(gh <= 64);
}
