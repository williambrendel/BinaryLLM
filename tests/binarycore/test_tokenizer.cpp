#include "binarycore/tokenizer.hpp"
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

TEST_CASE("3+ newlines still emit ParagraphBreak (not multiple)") {
    auto toks = tokenize("a\n\n\n\nb");
    REQUIRE(toks.size() == 3);
    CHECK(toks[1].encoded == encode_symbol(bits::Symbol::ParagraphBreak));
}

TEST_CASE("Windows line endings: \\r\\n behaves like \\n") {
    auto toks = tokenize("a\r\nb");
    REQUIRE(toks.size() == 3);
    CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Newline));

    auto toks2 = tokenize("a\r\n\r\nb");
    REQUIRE(toks2.size() == 3);
    CHECK(toks2[1].encoded == encode_symbol(bits::Symbol::ParagraphBreak));
}

TEST_CASE("old-Mac line endings: \\r alone") {
    auto toks = tokenize("a\rb");
    REQUIRE(toks.size() == 3);
    CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Newline));
}

TEST_CASE("single space is skipped (no token emitted)") {
    auto toks = tokenize("a b");
    REQUIRE(toks.size() == 2);
    CHECK(toks[0].text == "a");
    CHECK(toks[1].text == "b");
}

TEST_CASE("2+ spaces emit Indent symbol") {
    auto toks = tokenize("a  b");
    REQUIRE(toks.size() == 3);
    CHECK(toks[0].text == "a");
    CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Indent));
    CHECK(toks[2].text == "b");

    auto toks2 = tokenize("a    b");
    REQUIRE(toks2.size() == 3);
    CHECK(toks2[1].encoded == encode_symbol(bits::Symbol::Indent));
}

TEST_CASE("tab emits Indent symbol") {
    auto toks = tokenize("a\tb");
    REQUIRE(toks.size() == 3);
    CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Indent));
}

TEST_CASE("mixed tabs and spaces collapse to single Indent") {
    auto toks = tokenize("a \t  b");
    REQUIRE(toks.size() == 3);
    CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Indent));
}

TEST_CASE("trailing space at line end before \\n\\n is handled") {
    // "a  \n\nb" → word(a), paragraph_break, word(b)
    // The trailing 2 spaces would normally be an Indent, but they precede
    // newlines so are consumed as part of the newline-handling logic.
    auto toks = tokenize("a  \n\nb");
    // Acceptable: [word(a), indent, paragraph_break, word(b)]
    // OR:         [word(a), paragraph_break, word(b)] if we collapse trailing whitespace.
    // We document the actual behavior:
    bool found_pb = false;
    bool found_a = false, found_b = false;
    for (const auto& t : toks) {
        if (t.is_word() && t.text == "a") found_a = true;
        if (t.is_word() && t.text == "b") found_b = true;
        if (t.encoded == encode_symbol(bits::Symbol::ParagraphBreak)) found_pb = true;
    }
    CHECK(found_a);
    CHECK(found_b);
    CHECK(found_pb);
}

TEST_CASE("digits are word characters (join with adjacent letters)") {
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

TEST_CASE("hyphenated compound: 'well-known' → 3 tokens (word, hyphen, word)") {
    auto toks = tokenize("well-known");
    REQUIRE(toks.size() == 3);
    CHECK(toks[0].text == "well");
    CHECK(toks[1].encoded == encode_symbol(bits::Symbol::Hyphen));
    CHECK(toks[2].text == "known");
}

TEST_CASE("email-like: 'a@b.c' → 5 tokens (word, @, word, ., word)") {
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
    // 4 leading spaces → indent, then 'hello', space (skipped), 'world'
    REQUIRE(toks.size() == 3);
    CHECK(toks[0].encoded == encode_symbol(bits::Symbol::Indent));
    CHECK(toks[1].text == "hello");
    CHECK(toks[2].text == "world");
}
