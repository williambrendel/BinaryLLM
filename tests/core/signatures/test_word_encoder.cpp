// ============================================================================
// tests/core/signatures/test_word_encoder.cpp
// ============================================================================

#include "word_encoder.hpp"
#include "decomposer.hpp"
#include "dictionary.hpp"
#include "doctest.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using core::parts::Kind;
using core::parts::PartDictionary;
using core::signatures::encode_word;

namespace {

// Build a small dict covering enough of English to exercise the
// encoder without dragging in the full extractor. Letter atoms use
// the ## position encoding.
PartDictionary make_test_dict() {
  PartDictionary dict;
  // Whole atoms
  dict.add(Kind::Whole, "the");
  // Starts
  dict.add(Kind::Start, "un");
  dict.add(Kind::Start, "run");
  // Ends
  dict.add(Kind::End, "ing");
  dict.add(Kind::End, "ed");
  // Mids
  dict.add(Kind::Mid, "port");
  // Letters (positional, ## encoding)
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
  for (char c = '0'; c <= '9'; ++c) {
    const std::string s(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
  return dict;
}

bool is_sorted_unique(const std::vector<std::uint32_t>& v) {
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i - 1] >= v[i]) return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Empty input
// ---------------------------------------------------------------------------

TEST_CASE("encode_word: empty input → empty bag") {
  auto dict = make_test_dict();
  auto bag = encode_word(dict, "");
  CHECK(bag.empty());
}

// ---------------------------------------------------------------------------
// Whole-word path
// ---------------------------------------------------------------------------

TEST_CASE("encode_word: known whole-word → one-element bag") {
  auto dict = make_test_dict();
  auto bag = encode_word(dict, "the");
  REQUIRE(bag.size() == 1);
  CHECK(bag[0] == dict.lookup(Kind::Whole, "the"));
}

// ---------------------------------------------------------------------------
// Composite words: same parts as decompose, but sorted-unique
// ---------------------------------------------------------------------------

TEST_CASE("encode_word: 'running' has same set as decompose_word") {
  auto dict = make_test_dict();
  auto bag = encode_word(dict, "running");

  auto stream = core::parts::decompose_word(dict, "running");
  std::sort(stream.begin(), stream.end());
  stream.erase(std::unique(stream.begin(), stream.end()), stream.end());

  CHECK(bag == stream);
}

TEST_CASE("encode_word: 'unported' has same set as decompose_word") {
  auto dict = make_test_dict();
  auto bag = encode_word(dict, "unported");

  auto stream = core::parts::decompose_word(dict, "unported");
  std::sort(stream.begin(), stream.end());
  stream.erase(std::unique(stream.begin(), stream.end()), stream.end());

  CHECK(bag == stream);
}

// ---------------------------------------------------------------------------
// Sorted-ascending unique invariant
// ---------------------------------------------------------------------------

TEST_CASE("encode_word: result is sorted ascending and unique") {
  auto dict = make_test_dict();
  for (std::string_view w : {"running", "unported", "xyz", "100", "1st",
                              "the", "a"}) {
    auto bag = encode_word(dict, w);
    CAPTURE(w);
    CHECK(is_sorted_unique(bag));
  }
}

TEST_CASE("encode_word: same-position repeated singletons collapse") {
  // Build a dict where the SAME id can fire at multiple positions of
  // the same word — by using a Mid atom that matches a repeating
  // pattern. "ababab" with Mid="ab" → ab (mid at pos 0), ab (mid at
  // pos 2), ab (mid at pos 4): three claims of the same Mid id.
  PartDictionary dict;
  dict.add(Kind::Mid, "ab");
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
  auto stream = core::parts::decompose_word(dict, "ababab");
  // stream contains the Mid 'ab' id multiple times.
  auto bag = encode_word(dict, "ababab");
  // bag has the Mid id exactly once.
  CHECK(bag.size() < stream.size());
  CHECK(is_sorted_unique(bag));
}

// ---------------------------------------------------------------------------
// Option B uniqueness sanity (smoke-level)
// ---------------------------------------------------------------------------

TEST_CASE("encode_word: distinct words produce distinct bags (sanity)") {
  auto dict = make_test_dict();
  auto b_running  = encode_word(dict, "running");
  auto b_unported = encode_word(dict, "unported");
  auto b_xyz      = encode_word(dict, "xyz");
  auto b_the      = encode_word(dict, "the");
  auto b_a        = encode_word(dict, "a");

  // No two of these five distinct words should produce the same bag
  // under the small test dictionary.
  CHECK(b_running  != b_unported);
  CHECK(b_running  != b_xyz);
  CHECK(b_running  != b_the);
  CHECK(b_running  != b_a);
  CHECK(b_unported != b_xyz);
  CHECK(b_unported != b_the);
  CHECK(b_unported != b_a);
  CHECK(b_xyz      != b_the);
  CHECK(b_xyz      != b_a);
  CHECK(b_the      != b_a);
}

TEST_CASE("encode_word: non-empty input always yields non-empty bag (Letter backstop)") {
  auto dict = make_test_dict();
  for (std::string_view w : {"a", "ab", "abc", "1", "23", "running",
                              "xyz", "100"}) {
    CAPTURE(w);
    auto bag = encode_word(dict, w);
    CHECK_FALSE(bag.empty());
  }
}
