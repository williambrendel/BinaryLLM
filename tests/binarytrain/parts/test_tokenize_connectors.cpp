// ============================================================================
// tests/binarytrain/parts/test_tokenize_connectors.cpp
// ============================================================================

#include "binarytrain/parts/tokenize.hpp"
#include "doctest.h"

#include <string>
#include <vector>

using namespace binarytrain::parts;

namespace {

struct Tok {
  StreamToken::Type type;
  std::string value;
};

std::vector<Tok> as_pairs(const std::vector<StreamToken>& v) {
  std::vector<Tok> out;
  out.reserve(v.size());
  for (const auto& t : v) out.push_back({t.type, t.value});
  return out;
}

constexpr auto W = StreamToken::Type::Word;
constexpr auto D = StreamToken::Type::Delimiter;

bool eq(const Tok& a, StreamToken::Type t, const char* v) {
  return a.type == t && a.value == v;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Apostrophe
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("apostrophe internal: Alice's") {
  auto t = as_pairs(tokenize_stream("Alice's"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "alice's"));
}

TEST_CASE("apostrophe internal: don't") {
  auto t = as_pairs(tokenize_stream("don't"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "don't"));
}

TEST_CASE("apostrophe internal: rock'n'roll") {
  auto t = as_pairs(tokenize_stream("rock'n'roll"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "rock'n'roll"));
}

TEST_CASE("apostrophe internal: y'all and ma'am") {
  auto t1 = as_pairs(tokenize_stream("y'all"));
  REQUIRE(t1.size() == 1);
  CHECK(eq(t1[0], W, "y'all"));
  auto t2 = as_pairs(tokenize_stream("ma'am"));
  REQUIRE(t2.size() == 1);
  CHECK(eq(t2[0], W, "ma'am"));
}

TEST_CASE("apostrophe trailing: mothers'") {
  auto t = as_pairs(tokenize_stream("mothers'"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "mothers'"));
}

TEST_CASE("apostrophe trailing in sentence: boys' room") {
  auto t = as_pairs(tokenize_stream("boys' room"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], W, "boys'"));
  CHECK(eq(t[2], W, "room"));
}

TEST_CASE("quote: 'hi'") {
  auto t = as_pairs(tokenize_stream("'hi'"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], D, "'"));
  CHECK(eq(t[1], W, "hi"));
  CHECK(eq(t[2], D, "'"));
}

TEST_CASE("quote: 'a'") {
  auto t = as_pairs(tokenize_stream("'a'"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], D, "'"));
  CHECK(eq(t[1], W, "a"));
  CHECK(eq(t[2], D, "'"));
}

TEST_CASE("quote then possessive: 'hi' for the kids' party") {
  auto t = as_pairs(tokenize_stream("'hi' for the kids' party"));
  REQUIRE(t.size() == 11);
  CHECK(eq(t[1], W, "hi"));
  CHECK(eq(t[8], W, "kids'"));
}

TEST_CASE("apostrophe leading + alnum: '90s") {
  auto t = as_pairs(tokenize_stream("'90s"));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], D, "'"));
  CHECK(eq(t[1], W, "90s"));
}

TEST_CASE("apostrophe alone is delimiter") {
  auto t = as_pairs(tokenize_stream("'"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], D, "'"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Hyphen
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("hyphen internal: ice-cream") {
  auto t = as_pairs(tokenize_stream("ice-cream"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "ice-cream"));
}

TEST_CASE("hyphen internal: mother-in-law") {
  auto t = as_pairs(tokenize_stream("mother-in-law"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "mother-in-law"));
}

TEST_CASE("hyphen: date 2024-01-15") {
  auto t = as_pairs(tokenize_stream("2024-01-15"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "2024-01-15"));
}

TEST_CASE("hyphen: phone 555-123-4567") {
  auto t = as_pairs(tokenize_stream("555-123-4567"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "555-123-4567"));
}

TEST_CASE("hyphen leading: -bar (no digit, sign rule doesn't fire)") {
  auto t = as_pairs(tokenize_stream("-bar"));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], D, "-"));
  CHECK(eq(t[1], W, "bar"));
}

TEST_CASE("hyphen trailing: co-") {
  auto t = as_pairs(tokenize_stream("co-"));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], W, "co"));
  CHECK(eq(t[1], D, "-"));
}

TEST_CASE("hyphen double is em-dash") {
  auto t = as_pairs(tokenize_stream("foo--bar"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[1], D, "--"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Ampersand
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ampersand internal: A&M, R&D, AT&T") {
  auto a = as_pairs(tokenize_stream("A&M"));
  REQUIRE(a.size() == 1); CHECK(eq(a[0], W, "a&m"));
  auto b = as_pairs(tokenize_stream("R&D"));
  REQUIRE(b.size() == 1); CHECK(eq(b[0], W, "r&d"));
  auto c = as_pairs(tokenize_stream("AT&T"));
  REQUIRE(c.size() == 1); CHECK(eq(c[0], W, "at&t"));
}

TEST_CASE("ampersand chained: R&D&E") {
  auto t = as_pairs(tokenize_stream("R&D&E"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "r&d&e"));
}

TEST_CASE("ampersand leading: &c") {
  auto t = as_pairs(tokenize_stream("&c"));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], D, "&"));
  CHECK(eq(t[1], W, "c"));
}

TEST_CASE("ampersand alone is delimiter") {
  auto t = as_pairs(tokenize_stream("&"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], D, "&"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Comma (digit-only)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("comma thousand separator: 1,000") {
  auto t = as_pairs(tokenize_stream("1,000"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "1,000"));
}

TEST_CASE("comma EU decimal: 3,14") {
  auto t = as_pairs(tokenize_stream("3,14"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "3,14"));
}

TEST_CASE("comma: 2,4,5-t (chemistry)") {
  auto t = as_pairs(tokenize_stream("2,4,5-t"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "2,4,5-t"));
}

TEST_CASE("comma: cis-1,4-cyclohexadiene") {
  auto t = as_pairs(tokenize_stream("cis-1,4-cyclohexadiene"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "cis-1,4-cyclohexadiene"));
}

TEST_CASE("comma between letters stays delimiter") {
  auto t = as_pairs(tokenize_stream("Bob,Alice"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[1], D, ","));
}

TEST_CASE("comma between letter and digit stays delimiter") {
  auto t = as_pairs(tokenize_stream("a,5"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[1], D, ","));
}

// ─────────────────────────────────────────────────────────────────────────────
// Period: decimal + acronym
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("decimal: 3.14") {
  auto t = as_pairs(tokenize_stream("3.14"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "3.14"));
}

TEST_CASE("decimal: 0.5") {
  auto t = as_pairs(tokenize_stream("0.5"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "0.5"));
}

TEST_CASE("decimal: 100.500") {
  auto t = as_pairs(tokenize_stream("100.500"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "100.500"));
}

TEST_CASE("decimal chained: 3.14.15") {
  auto t = as_pairs(tokenize_stream("3.14.15"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "3.14.15"));
}

TEST_CASE("decimal trailing dot stays sentence-end: 5.") {
  auto t = as_pairs(tokenize_stream("5."));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], W, "5"));
  CHECK(eq(t[1], D, "."));
}

TEST_CASE("decimal: digit.letter does NOT absorb (5.x)") {
  auto t = as_pairs(tokenize_stream("5.x"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], W, "5"));
  CHECK(eq(t[1], D, "."));
  CHECK(eq(t[2], W, "x"));
}

TEST_CASE("acronym still works: U.S.A.") {
  auto t = as_pairs(tokenize_stream("U.S.A."));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], W, "u.s.a"));
  CHECK(eq(t[1], D, "."));
}

TEST_CASE("acronym still works: i.e.") {
  auto t = as_pairs(tokenize_stream("i.e."));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], W, "i.e"));
  CHECK(eq(t[1], D, "."));
}

TEST_CASE("acronym: a.k.a.") {
  auto t = as_pairs(tokenize_stream("a.k.a."));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], W, "a.k.a"));
  CHECK(eq(t[1], D, "."));
}

TEST_CASE("title Mr. is NOT absorbed") {
  auto t = as_pairs(tokenize_stream("Mr."));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], W, "mr"));
  CHECK(eq(t[1], D, "."));
}

TEST_CASE("mid-word period not absorbed: abc.def") {
  auto t = as_pairs(tokenize_stream("abc.def"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], W, "abc"));
  CHECK(eq(t[1], D, "."));
  CHECK(eq(t[2], W, "def"));
}

TEST_CASE("letter.digit period: X.5 (acronym rule absorbs)") {
  auto t = as_pairs(tokenize_stream("X.5"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "x.5"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Mixed: thousand + decimal in same number
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("US-style number: 1,000.50") {
  auto t = as_pairs(tokenize_stream("1,000.50"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "1,000.50"));
}

TEST_CASE("EU-style number: 1.000,50") {
  auto t = as_pairs(tokenize_stream("1.000,50"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "1.000,50"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Leading sign
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("signed: -5 at start of input") {
  auto t = as_pairs(tokenize_stream("-5"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "-5"));
}

TEST_CASE("signed: +5 at start of input") {
  auto t = as_pairs(tokenize_stream("+5"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "+5"));
}

TEST_CASE("signed decimal: -3.14") {
  auto t = as_pairs(tokenize_stream("-3.14"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "-3.14"));
}

TEST_CASE("signed thousand+decimal: +1,000.50") {
  auto t = as_pairs(tokenize_stream("+1,000.50"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "+1,000.50"));
}

TEST_CASE("signed in sentence: temp -5 degrees") {
  auto t = as_pairs(tokenize_stream("temp -5 degrees"));
  REQUIRE(t.size() == 5);
  CHECK(eq(t[0], W, "temp"));
  CHECK(eq(t[1], D, " "));
  CHECK(eq(t[2], W, "-5"));
  CHECK(eq(t[3], D, " "));
  CHECK(eq(t[4], W, "degrees"));
}

TEST_CASE("signed after space: 5 -10") {
  auto t = as_pairs(tokenize_stream("5 -10"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], W, "5"));
  CHECK(eq(t[1], D, " "));
  CHECK(eq(t[2], W, "-10"));
}

TEST_CASE("range without space: 5-10 (hyphen connector wins)") {
  auto t = as_pairs(tokenize_stream("5-10"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "5-10"));
}

TEST_CASE("arithmetic: 5+5 (sign does NOT fire after alnum)") {
  auto t = as_pairs(tokenize_stream("5+5"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], W, "5"));
  CHECK(eq(t[1], D, "+"));
  CHECK(eq(t[2], W, "5"));
}

TEST_CASE("arithmetic with spaces: 5 + 5") {
  auto t = as_pairs(tokenize_stream("5 + 5"));
  REQUIRE(t.size() == 5);
  CHECK(eq(t[0], W, "5"));
  CHECK(eq(t[1], D, " "));
  CHECK(eq(t[2], D, "+"));
  CHECK(eq(t[3], D, " "));
  CHECK(eq(t[4], W, "5"));
}

TEST_CASE("em-dash beats sign: --5") {
  auto t = as_pairs(tokenize_stream("--5"));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], D, "--"));
  CHECK(eq(t[1], W, "5"));
}

TEST_CASE("em-dash beats sign with currency: --$5 (em-dash still wins)") {
  auto t = as_pairs(tokenize_stream("--$5"));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], D, "--"));
  CHECK(eq(t[1], W, "$5"));
}

TEST_CASE("space breaks sign: - 5") {
  auto t = as_pairs(tokenize_stream("- 5"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], D, "-"));
  CHECK(eq(t[1], D, " "));
  CHECK(eq(t[2], W, "5"));
}

TEST_CASE("standalone + and -") {
  auto p = as_pairs(tokenize_stream("+"));
  REQUIRE(p.size() == 1);
  CHECK(eq(p[0], D, "+"));
  auto m = as_pairs(tokenize_stream("-"));
  REQUIRE(m.size() == 1);
  CHECK(eq(m[0], D, "-"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Currency: $ as leading number-prefix
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("currency: $5") {
  auto t = as_pairs(tokenize_stream("$5"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "$5"));
}

TEST_CASE("currency: $1,234.50") {
  auto t = as_pairs(tokenize_stream("$1,234.50"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "$1,234.50"));
}

TEST_CASE("currency: -$5 (sign + currency)") {
  auto t = as_pairs(tokenize_stream("-$5"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "-$5"));
}

TEST_CASE("currency: +$5") {
  auto t = as_pairs(tokenize_stream("+$5"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "+$5"));
}

TEST_CASE("currency THE KEY CASE: -$1,234.50") {
  auto t = as_pairs(tokenize_stream("-$1,234.50"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "-$1,234.50"));
}

TEST_CASE("currency: +$1,000.50") {
  auto t = as_pairs(tokenize_stream("+$1,000.50"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "+$1,000.50"));
}

TEST_CASE("currency in sentence: $1,000.50 total") {
  auto t = as_pairs(tokenize_stream("$1,000.50 total"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], W, "$1,000.50"));
  CHECK(eq(t[1], D, " "));
  CHECK(eq(t[2], W, "total"));
}

TEST_CASE("currency in financial sentence: profit was -$1,234.50") {
  auto t = as_pairs(tokenize_stream("profit was -$1,234.50"));
  REQUIRE(t.size() == 5);
  CHECK(eq(t[0], W, "profit"));
  CHECK(eq(t[1], D, " "));
  CHECK(eq(t[2], W, "was"));
  CHECK(eq(t[3], D, " "));
  CHECK(eq(t[4], W, "-$1,234.50"));
}

TEST_CASE("currency in parens: ($5)") {
  // Punct run breaks at the start of the number prefix.
  auto t = as_pairs(tokenize_stream("($5)"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], D, "("));
  CHECK(eq(t[1], W, "$5"));
  CHECK(eq(t[2], D, ")"));
}

TEST_CASE("currency in parens with sign: (-$5)") {
  auto t = as_pairs(tokenize_stream("(-$5)"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], D, "("));
  CHECK(eq(t[1], W, "-$5"));
  CHECK(eq(t[2], D, ")"));
}

TEST_CASE("currency alone is delim: $") {
  auto t = as_pairs(tokenize_stream("$"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], D, "$"));
}

TEST_CASE("currency without digit follow: $$ (double dollar)") {
  auto t = as_pairs(tokenize_stream("$$"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], D, "$$"));
}

TEST_CASE("currency at end of number: 5$ (not absorbed)") {
  // Only leading currency is handled; trailing splits.
  auto t = as_pairs(tokenize_stream("5$"));
  REQUIRE(t.size() == 2);
  CHECK(eq(t[0], W, "5"));
  CHECK(eq(t[1], D, "$"));
}

TEST_CASE("currency after alnum: a$5 (boundary check blocks prefix)") {
  // raw[i-1] = 'a' is alnum, so the $ prefix rule does NOT fire.
  auto t = as_pairs(tokenize_stream("a$5"));
  REQUIRE(t.size() == 3);
  CHECK(eq(t[0], W, "a"));
  CHECK(eq(t[1], D, "$"));
  CHECK(eq(t[2], W, "5"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Mixed regression
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("brand with possessive: R&D's") {
  auto t = as_pairs(tokenize_stream("R&D's"));
  REQUIRE(t.size() == 1);
  CHECK(eq(t[0], W, "r&d's"));
}

TEST_CASE("compound words: mother-in-law's ice-cream") {
  auto t = as_pairs(tokenize_stream("My mother-in-law's ice-cream."));
  REQUIRE(t.size() == 6);
  CHECK(eq(t[2], W, "mother-in-law's"));
  CHECK(eq(t[4], W, "ice-cream"));
}

TEST_CASE("signed number in financial sentence: profit was -1,234.50") {
  auto t = as_pairs(tokenize_stream("profit was -1,234.50"));
  REQUIRE(t.size() == 5);
  CHECK(eq(t[0], W, "profit"));
  CHECK(eq(t[1], D, " "));
  CHECK(eq(t[2], W, "was"));
  CHECK(eq(t[3], D, " "));
  CHECK(eq(t[4], W, "-1,234.50"));
}
