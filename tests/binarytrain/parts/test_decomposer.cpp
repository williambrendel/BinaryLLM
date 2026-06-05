// ============================================================================
// tests/binarytrain/parts/test_decomposer.cpp
// ============================================================================

#include "binarytrain/parts/decomposer.hpp"
#include "binarytrain/parts/dictionary.hpp"
#include "doctest.h"

#include <vector>

using binarytrain::parts::decompose_word;
using binarytrain::parts::decompose_delimiter;
using binarytrain::parts::Kind;
using binarytrain::parts::kInvalidPartId;
using binarytrain::parts::PartDictionary;

namespace {

// Build a small dictionary covering enough parts to verify the
// algorithm. Letter atoms use the "##" WordPiece-style position
// marker.
PartDictionary make_test_dict() {
  PartDictionary dict;
  // Whole-word atom.
  dict.add(Kind::Whole, "the");
  // Start parts.
  dict.add(Kind::Start, "un");
  dict.add(Kind::Start, "run");
  // End parts.
  dict.add(Kind::Start, "ing");  // also as Start, distinct from End-ing
  dict.add(Kind::End, "ing");
  dict.add(Kind::End, "ed");
  // Mid parts.
  dict.add(Kind::Mid, "port");
  // Letter singletons (alphabetic).
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s = std::string(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
  // Digit singletons.
  for (char c = '0'; c <= '9'; ++c) {
    const std::string s = std::string(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
  // Delimiters.
  dict.add(Kind::Delimiter, " ");
  dict.add(Kind::Delimiter, ".");
  return dict;
}

}  // namespace

TEST_CASE("decompose: whole-word fast path") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "the");
  REQUIRE(ids.size() == 1);
  CHECK(ids[0] == dict.lookup(Kind::Whole, "the"));
}

TEST_CASE("decompose: 'running' → run + n + ing") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "running");
  // Greedy: Start='run', End='ing', remaining 'n' at position 3 → Mid
  // scan finds no L=3+ match → letter singleton "##n##".
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "run"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##n##"));
  CHECK(ids[2] == dict.lookup(Kind::End, "ing"));
}

TEST_CASE("decompose: 'unported' → un + port + ed") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "unported");
  // Start='un', End='ed', Mid scan finds 'port' at position 2.
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "un"));
  CHECK(ids[1] == dict.lookup(Kind::Mid, "port"));
  CHECK(ids[2] == dict.lookup(Kind::End, "ed"));
}

TEST_CASE("decompose: 'xyz' → all letter singletons (no parts match)") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "xyz");
  // No Start, no End, no Mid. All chars → letter singletons.
  // Position 0 with no Start → "x##"
  // Position 1 middle → "##y##"
  // Position 2 with no End → "##z"
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Letter, "x##"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##y##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##z"));
}

TEST_CASE("decompose: empty word produces no parts") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "");
  CHECK(ids.empty());
}

TEST_CASE("decompose: longest-match wins (Start tries 4 → 3 → 2)") {
  PartDictionary dict;
  dict.add(Kind::Start, "un");      // length 2
  dict.add(Kind::Start, "unp");     // length 3
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s = std::string(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }

  // "unpack" — Starts available: "unp"(3), "un"(2). Longest: "unp".
  auto ids = decompose_word(dict, "unpack");
  // Expect: unp at position 0 (Start), then "##a##", "##c##", "##k"
  REQUIRE(ids.size() == 4);
  CHECK(ids[0] == dict.lookup(Kind::Start, "unp"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##a##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##c##"));
  CHECK(ids[3] == dict.lookup(Kind::Letter, "##k"));
}

TEST_CASE("decompose: delimiter known → one ID") {
  auto dict = make_test_dict();
  auto ids = decompose_delimiter(dict, " ");
  REQUIRE(ids.size() == 1);
  CHECK(ids[0] == dict.lookup(Kind::Delimiter, " "));
}

TEST_CASE("decompose: delimiter unknown → char fallback") {
  auto dict = make_test_dict();
  // "?!" is not in the dict, but neither is '?' or '!' as single
  // chars in our test dict. Result: empty.
  auto ids = decompose_delimiter(dict, "?!");
  CHECK(ids.empty());
}

TEST_CASE("decompose: 1-char word → start-position singleton") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "a");
  REQUIRE(ids.size() == 1);
  // Single-char word: convention is start-position.
  CHECK(ids[0] == dict.lookup(Kind::Letter, "a##"));
}

TEST_CASE("decompose: 1-char Whole atom takes precedence over Letter singleton") {
  PartDictionary dict;
  dict.add(Kind::Whole, "a");
  dict.add(Kind::Letter, "a##");
  auto ids = decompose_word(dict, "a");
  REQUIRE(ids.size() == 1);
  CHECK(ids[0] == dict.lookup(Kind::Whole, "a"));
}

TEST_CASE("decompose: digits become singletons by position") {
  auto dict = make_test_dict();
  // "100" — no Start, no End, no Mid match.
  auto ids = decompose_word(dict, "100");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Letter, "1##"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##0##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##0"));
}

TEST_CASE("decompose: mixed alphanumeric like '1st'") {
  auto dict = make_test_dict();
  // '1st' has no matching Start/End/Mid in our test dict.
  // Position 0 → "1##", position 1 → "##s##", position 2 → "##t".
  auto ids = decompose_word(dict, "1st");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Letter, "1##"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##s##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##t"));
}
