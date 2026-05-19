#include "binarycore/encoding/tokenizer.hpp"
#include "doctest.h"

using namespace binarycore;

TEST_CASE("empty input → empty token stream") {
  auto toks = tokenize("");
  CHECK(toks.empty());
}

TEST_CASE("single space-separated words") {
  auto toks = tokenize("hello world");
  REQUIRE(toks.size() == 2);
  CHECK(toks[0].is_word());
  CHECK(toks[0].text == "hello");
  CHECK(toks[1].is_word());
  CHECK(toks[1].text == "world");
}

TEST_CASE("punctuation splits into its own token") {
  auto toks = tokenize("hello, world!");
  REQUIRE(toks.size() == 4);
  CHECK(toks[0].text == "hello");
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Comma));
  CHECK(toks[2].text == "world");
  CHECK(toks[3].encoded == encode_symbol(bits::Symbol::Exclamation));
}

TEST_CASE("single newline emits Newline symbol") {
  auto toks = tokenize("a\nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[0].text == "a");
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Newline));
  CHECK(toks[2].text == "b");
}

TEST_CASE("two newlines emit ParagraphBreak symbol") {
  auto toks = tokenize("a\n\nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[0].text == "a");
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::ParagraphBreak));
  CHECK(toks[2].text == "b");
}

TEST_CASE("3+ newlines still emit ParagraphBreak") {
  auto toks = tokenize("a\n\n\n\nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::ParagraphBreak));
}

TEST_CASE("Windows line endings behave like \\n") {
  auto toks = tokenize("a\r\nb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Newline));

  auto toks2 = tokenize("a\r\n\r\nb");
  REQUIRE(toks2.size() == 3);
  CHECK(toks2[1].encoded == encode_symbol(bits::Symbol::ParagraphBreak));
}

TEST_CASE("old-Mac \\r line endings") {
  auto toks = tokenize("a\rb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Newline));
}

TEST_CASE("single space is skipped") {
  auto toks = tokenize("a b");
  REQUIRE(toks.size() == 2);
}

TEST_CASE("2+ spaces emit Indent symbol") {
  auto toks = tokenize("a  b");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Indent));
}

TEST_CASE("tab emits Indent symbol") {
  auto toks = tokenize("a\tb");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Indent));
}

TEST_CASE("mixed tabs and spaces → single Indent") {
  auto toks = tokenize("a \t  b");
  REQUIRE(toks.size() == 3);
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Indent));
}

TEST_CASE("digits are word characters") {
  auto toks = tokenize("abc123 def");
  REQUIRE(toks.size() == 2);
  CHECK(toks[0].text == "abc123");
  CHECK(toks[1].text == "def");
}

TEST_CASE("standalone digits form a word token") {
  auto toks = tokenize("42");
  REQUIRE(toks.size() == 1);
  CHECK(toks[0].is_word());
  CHECK(toks[0].text == "42");
}

TEST_CASE("hyphenated compound: 'well-known' → 3 tokens") {
  auto toks = tokenize("well-known");
  REQUIRE(toks.size() == 3);
  CHECK(toks[0].text == "well");
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Hyphen));
  CHECK(toks[2].text == "known");
}

TEST_CASE("email-like: 'a@b.c' → 5 tokens") {
  auto toks = tokenize("a@b.c");
  REQUIRE(toks.size() == 5);
  CHECK(toks[0].text == "a");
  CHECK(toks[1].encoded == encode_symbol(bits::Symbol::At));
  CHECK(toks[2].text == "b");
  CHECK(toks[3].encoded == encode_symbol(bits::Symbol::Period));
  CHECK(toks[4].text == "c");
}

TEST_CASE("paragraph with leading indent") {
  auto toks = tokenize("    hello world");
  REQUIRE(toks.size() == 3);
  CHECK(toks[0].encoded == encode_symbol(bits::Symbol::Indent));
  CHECK(toks[1].text == "hello");
  CHECK(toks[2].text == "world");
}
