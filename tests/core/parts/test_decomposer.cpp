// ============================================================================
// tests/core/parts/test_decomposer.cpp
//
// Tests for the (single) decompose_word algorithm: Whole fast path
// → dict-ID-order peel (Start ≤1, End ≤1, Mid multi-fire) → letter
// fill.
//
// Priority is determined by dict ID order. To test "longest wins" or
// "kind precedence", reorder the dict via reorder_dict_* and re-test
// in test_dict_reorder.cpp.
// ============================================================================

#include "decomposer.hpp"
#include "dictionary.hpp"
#include "doctest.h"

#include <vector>

using core::parts::decompose_word;
using core::parts::decompose_delimiter;
using core::parts::Kind;
using core::parts::kInvalidPartId;
using core::parts::PartDictionary;

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
  // Start "ing" (distinct kind from End "ing").
  dict.add(Kind::Start, "ing");
  // End parts.
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
  // Iteration: Start "un" (no), Start "run" (yes, claim [0,3)),
  // Start "ing" (start claimed, skip), End "ing" (yes, claim [4,7)),
  // End "ed" (end claimed, skip), Mid "port" (no).
  // Letter fill: pos 3 = 'n' → "##n##".
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "run"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##n##"));
  CHECK(ids[2] == dict.lookup(Kind::End, "ing"));
}

TEST_CASE("decompose: 'unported' → un + port + ed") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "unported");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "un"));
  CHECK(ids[1] == dict.lookup(Kind::Mid, "port"));
  CHECK(ids[2] == dict.lookup(Kind::End, "ed"));
}

TEST_CASE("decompose: 'xyz' → all letter singletons") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "xyz");
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

TEST_CASE("decompose: dict ID order determines Start priority") {
  // Lower ID wins regardless of length. With the new algorithm, the
  // order parts were added to the dict determines who fires first.
  // The "longest wins" property is achieved by REORDERING the dict
  // (see test_dict_reorder.cpp).
  PartDictionary dict;
  dict.add(Kind::Start, "un");   // id 0
  dict.add(Kind::Start, "unp");  // id 1
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s = std::string(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }

  // "unpack" — Start "un" added first, claims [0,2). Start "unp"
  // then blocked. Letters fill p, a, c, k.
  auto ids = decompose_word(dict, "unpack");
  REQUIRE(ids.size() == 5);
  CHECK(ids[0] == dict.lookup(Kind::Start, "un"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##p##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##a##"));
  CHECK(ids[3] == dict.lookup(Kind::Letter, "##c##"));
  CHECK(ids[4] == dict.lookup(Kind::Letter, "##k"));
}

TEST_CASE("decompose: Mid fires multiple times (multi-Mid peel)") {
  // "abcabc" with single Mid "abc". The Mid should claim BOTH
  // non-overlapping occurrences.
  PartDictionary dict;
  dict.add(Kind::Mid, "abc");
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s = std::string(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }

  auto ids = decompose_word(dict, "abcabc");
  REQUIRE(ids.size() == 2);
  CHECK(ids[0] == dict.lookup(Kind::Mid, "abc"));
  CHECK(ids[1] == dict.lookup(Kind::Mid, "abc"));
}

TEST_CASE("decompose: Start fires at most once") {
  PartDictionary dict;
  dict.add(Kind::Start, "ab");
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s = std::string(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
  // "abab" — Start "ab" claims [0,2). Cannot fire again at [2,4).
  // Letters fill a, b.
  auto ids = decompose_word(dict, "abab");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "ab"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##a##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##b"));
}

TEST_CASE("decompose: End fires at most once") {
  PartDictionary dict;
  dict.add(Kind::End, "ab");
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s = std::string(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
  // "abab" — End "ab" claims [2,4). Cannot fire again at [0,2).
  // Letters fill a, b.
  auto ids = decompose_word(dict, "abab");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Letter, "a##"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##b##"));
  CHECK(ids[2] == dict.lookup(Kind::End, "ab"));
}

TEST_CASE("decompose: delimiter known → one ID") {
  auto dict = make_test_dict();
  auto ids = decompose_delimiter(dict, " ");
  REQUIRE(ids.size() == 1);
  CHECK(ids[0] == dict.lookup(Kind::Delimiter, " "));
}

TEST_CASE("decompose: delimiter unknown → char fallback (no chars in dict)") {
  auto dict = make_test_dict();
  // "?!" — neither in dict as delim nor as single chars.
  auto ids = decompose_delimiter(dict, "?!");
  CHECK(ids.empty());
}

TEST_CASE("decompose: 1-char word → start-position singleton") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "a");
  REQUIRE(ids.size() == 1);
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
  auto ids = decompose_word(dict, "100");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Letter, "1##"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##0##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##0"));
}

TEST_CASE("decompose: mixed alphanumeric like '1st'") {
  auto dict = make_test_dict();
  auto ids = decompose_word(dict, "1st");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Letter, "1##"));
  CHECK(ids[1] == dict.lookup(Kind::Letter, "##s##"));
  CHECK(ids[2] == dict.lookup(Kind::Letter, "##t"));
}
