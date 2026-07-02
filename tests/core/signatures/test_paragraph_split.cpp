// ============================================================================
// tests/core/signatures/test_paragraph_split.cpp
// ============================================================================

#include "paragraph_split.hpp"
#include "tokenize.hpp"
#include "doctest.h"

#include <string>
#include <vector>

using core::parts::StreamToken;
using core::signatures::delimiter_ends_paragraph;
using core::signatures::ParagraphRange;
using core::signatures::ParagraphSplitConfig;
using core::signatures::split_paragraphs;

namespace {

StreamToken word(std::string v) {
  return StreamToken{StreamToken::Type::Word, std::move(v)};
}
StreamToken delim(std::string v) {
  return StreamToken{StreamToken::Type::Delimiter, std::move(v)};
}

}  // namespace

// ---------------------------------------------------------------------------
// delimiter_ends_paragraph
// ---------------------------------------------------------------------------

TEST_CASE("paragraph: '\\n\\n' is a paragraph break") {
  CHECK(delimiter_ends_paragraph("\n\n"));
}

TEST_CASE("paragraph: '\\n' (single newline) is not a paragraph break") {
  CHECK_FALSE(delimiter_ends_paragraph("\n"));
}

TEST_CASE("paragraph: other delimiters are not paragraph breaks") {
  CHECK_FALSE(delimiter_ends_paragraph(" "));
  CHECK_FALSE(delimiter_ends_paragraph("\t"));
  CHECK_FALSE(delimiter_ends_paragraph("."));
  CHECK_FALSE(delimiter_ends_paragraph("!"));
  CHECK_FALSE(delimiter_ends_paragraph(","));
  CHECK_FALSE(delimiter_ends_paragraph("--"));
  CHECK_FALSE(delimiter_ends_paragraph(""));
}

// ---------------------------------------------------------------------------
// split_paragraphs (basic)
// ---------------------------------------------------------------------------

TEST_CASE("paragraph: empty stream returns empty") {
  std::vector<StreamToken> tokens;
  auto out = split_paragraphs(tokens);
  CHECK(out.empty());
}

TEST_CASE("paragraph: no paragraph delimiter -> one paragraph") {
  std::vector<StreamToken> tokens = {
      word("hello"), delim(" "), word("world"), delim("."),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 4);
}

TEST_CASE("paragraph: single newline does NOT split") {
  std::vector<StreamToken> tokens = {
      word("line"), delim("\n"), word("continues"),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 3);
}

TEST_CASE("paragraph: '\\n\\n' splits into two paragraphs") {
  std::vector<StreamToken> tokens = {
      word("first"), delim("\n\n"),
      word("second"),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 3);
}

TEST_CASE("paragraph: multiple '\\n\\n' delimiters yield multiple paragraphs") {
  std::vector<StreamToken> tokens = {
      word("one"), delim("\n\n"),
      word("two"), delim("\n\n"),
      word("three"),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 3);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 4);
  CHECK(out[2].start == 4);
  CHECK(out[2].end == 5);
}

TEST_CASE("paragraph: trailing tokens form a final unterminated paragraph") {
  std::vector<StreamToken> tokens = {
      word("first"), delim("\n\n"),
      word("trailing"), delim(" "), word("text"),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 5);
}

TEST_CASE("paragraph: '\\n\\n' only input still produces a paragraph") {
  std::vector<StreamToken> tokens = {delim("\n\n")};
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 1);
}

TEST_CASE("paragraph: leading '\\n\\n' produces an initial paragraph") {
  std::vector<StreamToken> tokens = {
      delim("\n\n"), word("actual"), delim(" "), word("content"),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 1);
  CHECK(out[1].start == 1);
  CHECK(out[1].end == 4);
}

TEST_CASE("paragraph: consecutive '\\n\\n' each end a paragraph") {
  std::vector<StreamToken> tokens = {
      word("a"), delim("\n\n"), delim("\n\n"), word("b"),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 3);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 3);
  CHECK(out[2].start == 3);
  CHECK(out[2].end == 4);
}

// ---------------------------------------------------------------------------
// Max-token cap (forced split fallback)
// ---------------------------------------------------------------------------

TEST_CASE("paragraph: cap forces split when no '\\n\\n' appears in long input") {
  std::vector<StreamToken> tokens = {
      word("a"), word("b"), word("c"), word("d"), word("e"), word("f"),
  };
  ParagraphSplitConfig cfg;
  cfg.max_tokens = 3;
  auto out = split_paragraphs(tokens, cfg);
  REQUIRE(out.size() == 2);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 3);
  CHECK(out[1].start == 3);
  CHECK(out[1].end == 6);
}

TEST_CASE("paragraph: cap doesn't oversplit short final remainder") {
  std::vector<StreamToken> tokens = {
      word("a"), word("b"), word("c"), word("d"),
  };
  ParagraphSplitConfig cfg;
  cfg.max_tokens = 3;
  auto out = split_paragraphs(tokens, cfg);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 3);
  CHECK(out[1].start == 3);
  CHECK(out[1].end == 4);
}

TEST_CASE("paragraph: '\\n\\n' takes precedence over cap when both could fire") {
  std::vector<StreamToken> tokens = {
      word("a"), delim("\n\n"), word("c"),
  };
  ParagraphSplitConfig cfg;
  cfg.max_tokens = 3;
  auto out = split_paragraphs(tokens, cfg);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 3);
}

TEST_CASE("paragraph: cap of 1 splits at every token") {
  std::vector<StreamToken> tokens = {
      word("a"), word("b"), word("c"),
  };
  ParagraphSplitConfig cfg;
  cfg.max_tokens = 1;
  auto out = split_paragraphs(tokens, cfg);
  REQUIRE(out.size() == 3);
  CHECK(out[0].end == 1);
  CHECK(out[1].end == 2);
  CHECK(out[2].end == 3);
}

TEST_CASE("paragraph: default config is permissive on short input") {
  std::vector<StreamToken> tokens = {
      word("a"), delim(" "), word("b"), delim(" "), word("c"),
  };
  auto out = split_paragraphs(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 5);
}
