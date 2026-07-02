// ============================================================================
// tests/core/signatures/test_sentence_split.cpp
// ============================================================================

#include "sentence_split.hpp"
#include "tokenize.hpp"
#include "doctest.h"

#include <string>
#include <vector>

using core::parts::StreamToken;
using core::signatures::delimiter_ends_sentence;
using core::signatures::is_sentence_end;
using core::signatures::SentenceRange;
using core::signatures::SentenceSplitConfig;
using core::signatures::split_sentences;

namespace {

StreamToken word(std::string v) {
  return StreamToken{StreamToken::Type::Word, std::move(v)};
}
StreamToken delim(std::string v) {
  return StreamToken{StreamToken::Type::Delimiter, std::move(v)};
}

}  // namespace

// ---------------------------------------------------------------------------
// delimiter_ends_sentence  (context-free primitive)
// ---------------------------------------------------------------------------

TEST_CASE("split: terminator detector recognizes . ! ?") {
  CHECK(delimiter_ends_sentence("."));
  CHECK(delimiter_ends_sentence("!"));
  CHECK(delimiter_ends_sentence("?"));
}

TEST_CASE("split: non-terminator delimiters do not end") {
  CHECK_FALSE(delimiter_ends_sentence(" "));
  CHECK_FALSE(delimiter_ends_sentence("\n"));
  CHECK_FALSE(delimiter_ends_sentence("\n\n"));
  CHECK_FALSE(delimiter_ends_sentence(","));
  CHECK_FALSE(delimiter_ends_sentence(";"));
  CHECK_FALSE(delimiter_ends_sentence("--"));
}

TEST_CASE("split: terminator inside compound delimiter still ends") {
  CHECK(delimiter_ends_sentence("..."));
  CHECK(delimiter_ends_sentence("!?"));
  CHECK(delimiter_ends_sentence("?!"));
  CHECK(delimiter_ends_sentence(" . "));
}

// ---------------------------------------------------------------------------
// split_sentences  (basic — preserved from original tests)
// ---------------------------------------------------------------------------

TEST_CASE("split: empty stream returns empty") {
  std::vector<StreamToken> tokens;
  auto out = split_sentences(tokens);
  CHECK(out.empty());
}

TEST_CASE("split: single sentence with terminator") {
  std::vector<StreamToken> tokens = {
      word("hello"), delim(" "), word("world"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 4);
}

TEST_CASE("split: two sentences") {
  std::vector<StreamToken> tokens = {
      word("hi"), delim("."),
      delim(" "),
      word("bye"), delim("?"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 5);
}

TEST_CASE("split: trailing tokens without terminator form a sentence") {
  std::vector<StreamToken> tokens = {
      word("hello"), delim("."),
      word("trailing"), delim(" "), word("text"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 5);
}

TEST_CASE("split: no terminators at all -> one sentence") {
  std::vector<StreamToken> tokens = {
      word("plain"), delim(" "), word("text"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 3);
}

TEST_CASE("split: consecutive terminators each end a (possibly empty) sentence") {
  std::vector<StreamToken> tokens = {
      word("wow"), delim("."), delim("!"), delim("?"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 3);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 2);
  CHECK(out[1].start == 2);
  CHECK(out[1].end == 3);
  CHECK(out[2].start == 3);
  CHECK(out[2].end == 4);
}

TEST_CASE("split: ellipsis terminates a sentence") {
  std::vector<StreamToken> tokens = {
      word("wait"), delim("..."),
      word("really"), delim("?"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 2);
  CHECK(out[1].end == 4);
}

TEST_CASE("split: terminator-only input still produces a sentence") {
  std::vector<StreamToken> tokens = {delim(".")};
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 1);
}

// ---------------------------------------------------------------------------
// Abbreviations (Rule 3)
// ---------------------------------------------------------------------------

TEST_CASE("split: abbreviation 'mr.' does not end sentence") {
  std::vector<StreamToken> tokens = {
      word("mr"), delim("."), delim(" "), word("smith"),
      delim(" "), word("said"), delim(" "), word("hi"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].start == 0);
  CHECK(out[0].end == 9);
}

TEST_CASE("split: abbreviation 'dr.' at end of stream is suppressed") {
  std::vector<StreamToken> tokens = {
      word("talk"), delim(" "), word("to"), delim(" "),
      word("dr"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 6);
}

TEST_CASE("split: two abbreviations and a real terminator at end") {
  std::vector<StreamToken> tokens = {
      word("dr"), delim("."), delim(" "), word("smith"),
      delim(" "), word("examined"), delim(" "),
      word("mr"), delim("."), delim(" "), word("jones"),
      delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 12);
}

TEST_CASE("split: standalone period after whitespace is not an abbreviation") {
  std::vector<StreamToken> tokens = {
      word("mr"), delim(" "), delim("."), delim(" "), word("hello"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 3);
}

TEST_CASE("split: custom abbreviation extends the default set") {
  SentenceSplitConfig cfg;
  cfg.abbreviations.insert("fig");
  std::vector<StreamToken> tokens = {
      word("see"), delim(" "), word("fig"), delim("."), delim(" "),
      word("for"), delim(" "), word("details"), delim("."),
  };
  auto out = split_sentences(tokens, cfg);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 9);
}

// ---------------------------------------------------------------------------
// Compound delimiters with trailing , or ;  (Rule 2)
// ---------------------------------------------------------------------------

TEST_CASE("split: compound '.,' (i.e., comma after period) does not end") {
  std::vector<StreamToken> tokens = {
      word("i.e"), delim(".,"), delim(" "), word("the"),
      delim(" "), word("rest"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 7);
}

TEST_CASE("split: compound '.;' does not end") {
  std::vector<StreamToken> tokens = {
      word("ok"), delim(".;"), delim(" "), word("hi"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 5);
}

// ---------------------------------------------------------------------------
// Acronyms  (Rule 4)
// ---------------------------------------------------------------------------

TEST_CASE("split: acronym 'u.s.a' mid-sentence does not end") {
  std::vector<StreamToken> tokens = {
      word("the"), delim(" "), word("u.s.a"), delim("."),
      delim(" "), word("is"), delim(" "), word("great"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 9);
}

TEST_CASE("split: acronym at end-of-stream DOES end sentence") {
  std::vector<StreamToken> tokens = {
      word("living"), delim(" "), word("in"), delim(" "),
      word("the"), delim(" "), word("u.s.a"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 8);
}

TEST_CASE("split: acronym before newline-bearing delim DOES end sentence") {
  std::vector<StreamToken> tokens = {
      word("u.s.a"), delim("."), delim("\n\n"),
      word("next"), delim(" "), word("paragraph"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 2);
  CHECK(out[1].end == 7);
}

// ---------------------------------------------------------------------------
// URL / email  (Rule 5)
// ---------------------------------------------------------------------------

TEST_CASE("split: '@' email intra-domain period is not a boundary") {
  std::vector<StreamToken> tokens = {
      word("email"), delim(" "),
      word("foo"), delim("@"), word("bar"), delim("."), word("com"),
      delim(" "), word("please"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 10);
}

TEST_CASE("split: '://' URL intra-domain period is not a boundary") {
  std::vector<StreamToken> tokens = {
      word("see"), delim(" "),
      word("http"), delim("://"), word("foo"), delim("."), word("com"),
      delim("/"), word("path"),
      delim(" "), word("for"), delim(" "), word("details"), delim("."),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 1);
  CHECK(out[0].end == 14);
}

TEST_CASE("split: period after URL still ends the sentence") {
  std::vector<StreamToken> tokens = {
      word("email"), delim(" "), word("me"), delim(" "),
      word("at"), delim(" "),
      word("foo"), delim("@"), word("bar"), delim("."), word("com"),
      delim("."), delim(" "),
      word("got"), delim(" "), word("it"), delim("?"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 12);
  CHECK(out[1].end == 17);
}

TEST_CASE("split: bare 'hello.world' (no URL marker) still splits") {
  std::vector<StreamToken> tokens = {
      word("hello"), delim("."), word("world"),
  };
  auto out = split_sentences(tokens);
  REQUIRE(out.size() == 2);
  CHECK(out[0].end == 2);
}

// ---------------------------------------------------------------------------
// is_sentence_end  (direct API exercise)
// ---------------------------------------------------------------------------

TEST_CASE("is_sentence_end: out-of-range index returns false") {
  std::vector<StreamToken> tokens = {word("hi")};
  SentenceSplitConfig cfg;
  CHECK_FALSE(is_sentence_end(tokens, 5, cfg));
  CHECK_FALSE(is_sentence_end(tokens, tokens.size(), cfg));
}

TEST_CASE("is_sentence_end: word tokens are never sentence ends") {
  std::vector<StreamToken> tokens = {
      word("hi"), delim(" "), word("end."),
  };
  SentenceSplitConfig cfg;
  CHECK_FALSE(is_sentence_end(tokens, 0, cfg));
  CHECK_FALSE(is_sentence_end(tokens, 2, cfg));
}
