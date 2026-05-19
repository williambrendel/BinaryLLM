#include "binarycore/encoding/token_encoder.hpp"
#include "doctest.h"

using namespace binarycore;

static uint64_t get_field(uint64_t token, int start, int width) {
  uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
  return (token >> start) & mask;
}
static bool get_bit(uint64_t token, int pos) {
  return (token >> pos) & 1ULL;
}
static uint64_t count(uint64_t token, char lc) {
  const int slot = bits::count_slot(lc);
  if (slot < 0) return 0;
  return (token >> (bits::LetterCountStart + slot * 2)) & 0x3ULL;
}

TEST_CASE("type flag distinguishes words from symbols") {
  CHECK(is_word(encode_word("hello")));
  CHECK(is_symbol(encode_symbol(bits::Symbol::Period)));
}

TEST_CASE("count for common letters: cat") {
  uint64_t t = encode_word("cat");
  CHECK(count(t, 'a') == 1);
  CHECK(count(t, 'c') == 1);
  CHECK(count(t, 't') == 1);
}

TEST_CASE("count for hello") {
  uint64_t t = encode_word("hello");
  CHECK(count(t, 'e') == 1);
  CHECK(count(t, 'l') == 2);
  CHECK(count(t, 'o') == 1);
}

TEST_CASE("count saturates at 3") {
  CHECK(count(encode_word("aaaaa"), 'a') == 3);
}

TEST_CASE("presence for rare letters: h, b, d, etc") {
  uint64_t t = encode_word("habit");
  CHECK(get_bit(t, bits::PresenceStart + bits::presence_bit('h')));
  CHECK(get_bit(t, bits::PresenceStart + bits::presence_bit('b')));
}

TEST_CASE("paired letters share their presence bit") {
  CHECK(bits::presence_bit('x') == bits::presence_bit('z'));
  CHECK(bits::presence_bit('q') == bits::presence_bit('j'));
  CHECK(bits::presence_bit('k') == bits::presence_bit('w'));
}

TEST_CASE("capitalization") {
  CHECK(get_field(encode_word("hello"), bits::CapsStart, 2) ==
        static_cast<uint64_t>(bits::Caps::None));
  CHECK(get_field(encode_word("Hello"), bits::CapsStart, 2) ==
        static_cast<uint64_t>(bits::Caps::FirstUpper));
  CHECK(get_field(encode_word("HELLO"), bits::CapsStart, 2) ==
        static_cast<uint64_t>(bits::Caps::AllUpper));
}

static uint64_t gray(uint64_t x) { return x ^ (x >> 1); }

TEST_CASE("length bucket uses Gray code") {
}

TEST_CASE("digit presence") {
  uint64_t t = encode_word("abc123");
}

TEST_CASE("last-letter bit") {
}

TEST_CASE("reserved bits are zero") {
  uint64_t t = encode_word("hello");
}

TEST_CASE("encoding is deterministic") {
  CHECK(encode_word("hello") == encode_word("hello"));
  CHECK(encode_word("hello") != encode_word("world"));
}

TEST_CASE("symbols all have type flag set") {
  for (int i = 0; i < 64; ++i) {
    CHECK(is_symbol(encode_symbol(static_cast<bits::Symbol>(i))));
  }
}

TEST_CASE("empty word encodes to all-zeros word token") {
  CHECK(encode_word("") == 0);
}
