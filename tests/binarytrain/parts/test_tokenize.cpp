// ============================================================================
// tests/binarytrain/parts/test_tokenize.cpp
// ============================================================================

#include "binarytrain/parts/tokenize.hpp"
#include "doctest.h"

#include <string>
#include <vector>

using binarytrain::parts::StreamToken;
using binarytrain::parts::tokenize_stream;

namespace {

// Helpers.
bool is_word(const StreamToken& t) {
  return t.type == StreamToken::Type::Word;
}
bool is_delim(const StreamToken& t) {
  return t.type == StreamToken::Type::Delimiter;
}

}  // namespace

TEST_CASE("tokenize: empty input → no tokens") {
  auto toks = tokenize_stream("");
  CHECK(toks.empty());
}

TEST_CASE("tokenize: single word lowercased") {
  auto toks = tokenize_stream("Running");
  REQUIRE(toks.size() == 1);
  CHECK(is_word(toks[0]));
  CHECK(toks[0].value == "running");
}

TEST_CASE("tokenize: simple sentence with spaces and period") {
  auto toks = tokenize_stream("I am Running.");
  // Expected: I, ' ', am, ' ', running, '.'
  REQUIRE(toks.size() == 6);
  CHECK(toks[0].value == "i");           CHECK(is_word(toks[0]));
  CHECK(toks[1].value == " ");           CHECK(is_delim(toks[1]));
  CHECK(toks[2].value == "am");          CHECK(is_word(toks[2]));
  CHECK(toks[3].value == " ");           CHECK(is_delim(toks[3]));
  CHECK(toks[4].value == "running");     CHECK(is_word(toks[4]));
  CHECK(toks[5].value == ".");           CHECK(is_delim(toks[5]));
}

TEST_CASE("tokenize: double space collapses to tab class") {
  auto toks = tokenize_stream("a  b");  // two spaces
  REQUIRE(toks.size() == 3);
  CHECK(toks[0].value == "a");
  CHECK(toks[1].value == "\t");   // indent class
  CHECK(is_delim(toks[1]));
  CHECK(toks[2].value == "b");
}

TEST_CASE("tokenize: tab character is indent class") {
  auto toks = tokenize_stream("a\tb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == "\t");
}

TEST_CASE("tokenize: single newline is line-break class") {
  auto toks = tokenize_stream("a\nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == "\n");
  CHECK(is_delim(toks[1]));
}

TEST_CASE("tokenize: double newline is paragraph class") {
  auto toks = tokenize_stream("a\n\nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == "\n\n");
}

TEST_CASE("tokenize: triple newline collapses to paragraph class") {
  auto toks = tokenize_stream("a\n\n\nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == "\n\n");
}

TEST_CASE("tokenize: ellipsis collapses (3+ dots → one token)") {
  auto toks = tokenize_stream("hello... world");
  // hello, ..., ' ', world
  REQUIRE(toks.size() == 4);
  CHECK(toks[0].value == "hello");
  CHECK(toks[1].value == "...");
  CHECK(toks[2].value == " ");
  CHECK(toks[3].value == "world");
}

TEST_CASE("tokenize: four dots also collapse to ellipsis") {
  auto toks = tokenize_stream("hello.... world");
  REQUIRE(toks.size() == 4);
  CHECK(toks[1].value == "...");  // four dots normalize to ellipsis
}

TEST_CASE("tokenize: two dots stay as run (not ellipsis)") {
  auto toks = tokenize_stream("hello.. world");
  REQUIRE(toks.size() == 4);
  CHECK(toks[1].value == "..");  // kept literally (not ellipsis)
}

TEST_CASE("tokenize: double dash is em-dash class") {
  auto toks = tokenize_stream("yes--no");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == "--");
}

TEST_CASE("tokenize: triple dash collapses to em-dash") {
  auto toks = tokenize_stream("yes---no");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == "--");
}

TEST_CASE("tokenize: mixed punctuation kept literally") {
  auto toks = tokenize_stream("yes,no");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == ",");
}

TEST_CASE("tokenize: comma+space splits into two delimiters") {
  // ',' is punctuation, ' ' is whitespace — different categories → split.
  auto toks = tokenize_stream("a, b");
  REQUIRE(toks.size() == 4);
  CHECK(toks[0].value == "a");
  CHECK(toks[1].value == ",");
  CHECK(toks[2].value == " ");
  CHECK(toks[3].value == "b");
}

TEST_CASE("tokenize: words with digits stay together") {
  auto toks = tokenize_stream("iphone15");
  REQUIRE(toks.size() == 1);
  CHECK(toks[0].value == "iphone15");
  CHECK(is_word(toks[0]));
}

TEST_CASE("tokenize: paragraph break with intermediate space") {
  // The "\n  \n" pattern: newline, spaces, newline — newlines >= 2 →
  // paragraph class.
  auto toks = tokenize_stream("a\n  \nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].value == "\n\n");
}
