// ============================================================================
// tests/binarytrain/parts/test_extractor.cpp
// ============================================================================

#include "binarytrain/parts/decomposer.hpp"
#include "binarytrain/parts/extractor.hpp"
#include "doctest.h"

#include <string>
#include <vector>

using binarytrain::parts::ExtractorConfig;
using binarytrain::parts::Kind;
using binarytrain::parts::kInvalidPartId;
using binarytrain::parts::PartDictionary;
using binarytrain::parts::PartExtractor;

TEST_CASE("extractor: empty corpus produces backstop atoms only") {
  PartExtractor ex;
  PartDictionary dict = ex.finalize();
  // Letter atoms (## position encoding):
  //   26 letters × 3 + 10 digits × 3 + 6 connectors × 3 = 126.
  CHECK(dict.count_of_kind(Kind::Letter) == 126);
  // 26 letters + 10 digits + 6 connectors = 42 single-char Whole atoms.
  CHECK(dict.count_of_kind(Kind::Whole) == 42);
  // 6 connector chars seeded unconditionally as Delimiter atoms.
  CHECK(dict.count_of_kind(Kind::Delimiter) == 6);
  CHECK(dict.count_of_kind(Kind::Start) == 0);
  CHECK(dict.count_of_kind(Kind::Mid) == 0);
  CHECK(dict.count_of_kind(Kind::End) == 0);
  CHECK(ex.last_peel_iterations() == 1);
}

TEST_CASE("extractor: every single letter and digit is a Whole atom") {
  PartExtractor ex;
  PartDictionary dict = ex.finalize();
  for (char c = 'a'; c <= 'z'; ++c) {
    CHECK(dict.lookup(Kind::Whole, std::string(1, c)) != kInvalidPartId);
  }
  for (char c = '0'; c <= '9'; ++c) {
    CHECK(dict.lookup(Kind::Whole, std::string(1, c)) != kInvalidPartId);
  }
}

TEST_CASE("extractor: every connector char is a Whole atom") {
  PartExtractor ex;
  PartDictionary dict = ex.finalize();
  CHECK(dict.lookup(Kind::Whole, "-") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "'") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "&") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, ",") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, ".") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "$") != kInvalidPartId);
}

TEST_CASE("extractor: Letter atoms use ## position encoding") {
  PartExtractor ex;
  PartDictionary dict = ex.finalize();
  // Letters: a##, ##a##, ##a.
  CHECK(dict.lookup(Kind::Letter, "a##")   != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "##a##") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "##a")   != kInvalidPartId);
  // Digits.
  CHECK(dict.lookup(Kind::Letter, "5##")   != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "##5##") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "##5")   != kInvalidPartId);
  // Old encoding strings must NOT be present (sanity check).
  CHECK(dict.lookup(Kind::Letter, "a-")  == kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "-a")  == kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "-a-") == kInvalidPartId);
}

TEST_CASE("extractor: every connector has positional Letter atoms") {
  // With the ## encoding (no collision for hyphen anymore), all 6
  // connector chars get the full positional set.
  PartExtractor ex;
  PartDictionary dict = ex.finalize();
  const char* connectors[] = {"-", "'", "&", ",", ".", "$"};
  for (const char* c : connectors) {
    const std::string s = c;
    CHECK(dict.lookup(Kind::Letter, s + "##")        != kInvalidPartId);
    CHECK(dict.lookup(Kind::Letter, "##" + s + "##") != kInvalidPartId);
    CHECK(dict.lookup(Kind::Letter, "##" + s)        != kInvalidPartId);
  }
}

TEST_CASE("extractor: hyphen positional Letter atoms are all distinct") {
  // Regression test for the collision the ## encoding fixes:
  //   start "-##" must differ from end "##-" (under "-" encoding both
  //   would have been "--").
  PartExtractor ex;
  PartDictionary dict = ex.finalize();
  const std::uint32_t hyphen_start = dict.lookup(Kind::Letter, "-##");
  const std::uint32_t hyphen_mid   = dict.lookup(Kind::Letter, "##-##");
  const std::uint32_t hyphen_end   = dict.lookup(Kind::Letter, "##-");
  CHECK(hyphen_start != kInvalidPartId);
  CHECK(hyphen_mid   != kInvalidPartId);
  CHECK(hyphen_end   != kInvalidPartId);
  CHECK(hyphen_start != hyphen_mid);
  CHECK(hyphen_mid   != hyphen_end);
  CHECK(hyphen_start != hyphen_end);
}

TEST_CASE("extractor: short words become Whole atoms by default") {
  PartExtractor ex;
  ex.add_word("the");
  ex.add_word("am");
  ex.add_word("verylongunlikelyword");
  PartDictionary dict = ex.finalize();
  CHECK(dict.lookup(Kind::Whole, "the") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "am") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "verylongunlikelyword") == kInvalidPartId);
}

TEST_CASE("extractor: whole_max_len gate respected") {
  ExtractorConfig cfg;
  cfg.whole_max_len = 3;
  PartExtractor ex(cfg);
  ex.add_word("the");
  ex.add_word("four");

  PartDictionary dict = ex.finalize();
  CHECK(dict.lookup(Kind::Whole, "the") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "four") == kInvalidPartId);
}

TEST_CASE("extractor: delimiters added in frequency order") {
  PartExtractor ex;
  for (int i = 0; i < 5; ++i) ex.add_delimiter(" ");
  for (int i = 0; i < 2; ++i) ex.add_delimiter(".");
  ex.add_delimiter("\n");

  PartDictionary dict = ex.finalize();
  CHECK(dict.lookup(Kind::Delimiter, " ") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, ".") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "\n") != kInvalidPartId);
}

TEST_CASE("extractor: connector chars always seeded as Delimiter atoms") {
  PartExtractor ex;
  ex.add_word("hello");
  ex.add_word("world");
  ex.add_delimiter(" ");
  ex.add_delimiter("!");
  ex.add_delimiter("?");

  PartDictionary dict = ex.finalize();

  CHECK(dict.lookup(Kind::Delimiter, "-") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "'") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "&") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, ",") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, ".") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "$") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, " ") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "!") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "?") != kInvalidPartId);

  CHECK(dict.count_of_kind(Kind::Delimiter) == 9);
}

TEST_CASE("extractor: observed connector delim is not double-counted") {
  PartExtractor ex;
  ex.add_word("a");
  ex.add_delimiter("-");
  ex.add_delimiter("-");
  ex.add_delimiter("'");

  PartDictionary dict = ex.finalize();
  CHECK(dict.lookup(Kind::Delimiter, "-") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "'") != kInvalidPartId);
  CHECK(dict.count_of_kind(Kind::Delimiter) == 6);
}

TEST_CASE("extractor: observed_word_count counts all add_word calls") {
  PartExtractor ex;
  ex.add_word("a");
  ex.add_word("b");
  ex.add_word("a");
  CHECK(ex.observed_word_count() == 3);
  CHECK(ex.distinct_word_count() == 2);
}

TEST_CASE("extractor: peel iteration converges on a simple morphology pattern") {
  PartExtractor ex;
  for (int i = 0; i < 100; ++i) {
    ex.add_word("running");
    ex.add_word("singing");
    ex.add_word("walking");
    ex.add_word("dancing");
    ex.add_word("eating");
    ex.add_word("driving");
    ex.add_word("running");
  }
  ex.add_word("alpha");
  ex.add_word("beta");

  PartDictionary dict = ex.finalize();

  CHECK(dict.lookup(Kind::End, "ing") != kInvalidPartId);

  CHECK(ex.last_peel_iterations() >= 1);
  CHECK(ex.last_peel_iterations() <= 20);
}

TEST_CASE("extractor: peel shrinks singleton runs and terminates") {
  ExtractorConfig cfg;
  cfg.whole_max_len = 5;
  PartExtractor ex(cfg);
  for (int i = 0; i < 20; ++i) {
    ex.add_word("running");
    ex.add_word("singing");
    ex.add_word("dancing");
    ex.add_word("walking");
    ex.add_word("eating");
    ex.add_word("ingredients");
    ex.add_word("important");
    ex.add_word("interesting");
    ex.add_word("information");
  }

  PartDictionary dict = ex.finalize();

  CHECK(ex.last_peel_iterations() >= 1);
  CHECK(ex.last_peel_iterations() <= 20);

  CHECK(dict.count_of_kind(Kind::Start) > 0);
  CHECK(dict.count_of_kind(Kind::Mid)   > 0);
  CHECK(dict.count_of_kind(Kind::End)   > 0);
}
