// ============================================================================
// tests/core/parts/test_dictionary.cpp
// ============================================================================

#include "dictionary.hpp"
#include "doctest.h"

#include <sstream>

using core::parts::Kind;
using core::parts::kInvalidPartId;
using core::parts::PartDictionary;

TEST_CASE("dictionary: empty has size 0") {
  PartDictionary dict;
  CHECK(dict.size() == 0);
}

TEST_CASE("dictionary: add returns sequential IDs") {
  PartDictionary dict;
  CHECK(dict.add(Kind::Start, "un") == 0);
  CHECK(dict.add(Kind::End, "ing") == 1);
  CHECK(dict.add(Kind::Mid, "port") == 2);
  CHECK(dict.size() == 3);
}

TEST_CASE("dictionary: re-add returns existing ID") {
  PartDictionary dict;
  CHECK(dict.add(Kind::Start, "un") == 0);
  CHECK(dict.add(Kind::Start, "un") == 0);  // same → 0
  CHECK(dict.add(Kind::End, "un") == 1);    // diff kind → new
}

TEST_CASE("dictionary: lookup returns kInvalidPartId for absent") {
  PartDictionary dict;
  dict.add(Kind::Start, "un");
  CHECK(dict.lookup(Kind::Start, "re") == kInvalidPartId);
  CHECK(dict.lookup(Kind::End, "un") == kInvalidPartId);  // wrong kind
}

TEST_CASE("dictionary: count_of_kind works") {
  PartDictionary dict;
  dict.add(Kind::Start, "un");
  dict.add(Kind::Start, "re");
  dict.add(Kind::End, "ing");
  CHECK(dict.count_of_kind(Kind::Start) == 2);
  CHECK(dict.count_of_kind(Kind::End) == 1);
  CHECK(dict.count_of_kind(Kind::Mid) == 0);
}

TEST_CASE("dictionary: save/load roundtrip preserves IDs") {
  PartDictionary dict;
  dict.add(Kind::Whole, "the");
  dict.add(Kind::Start, "un");
  dict.add(Kind::End, "ing");
  dict.add(Kind::Mid, "port");
  dict.add(Kind::Letter, "-a-");
  dict.add(Kind::Delimiter, " ");

  std::ostringstream oss;
  dict.save(oss);

  std::istringstream iss(oss.str());
  PartDictionary loaded = PartDictionary::load(iss);

  CHECK(loaded.size() == dict.size());
  CHECK(loaded.lookup(Kind::Whole, "the") == 0);
  CHECK(loaded.lookup(Kind::Start, "un") == 1);
  CHECK(loaded.lookup(Kind::End, "ing") == 2);
  CHECK(loaded.lookup(Kind::Mid, "port") == 3);
  CHECK(loaded.lookup(Kind::Letter, "-a-") == 4);
  CHECK(loaded.lookup(Kind::Delimiter, " ") == 5);
}

TEST_CASE("dictionary: save/load handles escape characters") {
  PartDictionary dict;
  dict.add(Kind::Delimiter, "\n");
  dict.add(Kind::Delimiter, "\n\n");
  dict.add(Kind::Delimiter, "\t");
  dict.add(Kind::Delimiter, "\\");

  std::ostringstream oss;
  dict.save(oss);

  std::istringstream iss(oss.str());
  PartDictionary loaded = PartDictionary::load(iss);

  CHECK(loaded.lookup(Kind::Delimiter, "\n") == 0);
  CHECK(loaded.lookup(Kind::Delimiter, "\n\n") == 1);
  CHECK(loaded.lookup(Kind::Delimiter, "\t") == 2);
  CHECK(loaded.lookup(Kind::Delimiter, "\\") == 3);
}

TEST_CASE("dictionary: has_xxx helpers work after build_indices") {
  PartDictionary dict;
  dict.add(Kind::Start, "un");
  dict.add(Kind::End, "ing");
  dict.add(Kind::Mid, "port");
  dict.add(Kind::Whole, "the");
  dict.add(Kind::Delimiter, " ");

  CHECK(dict.has_start("un"));
  CHECK_FALSE(dict.has_start("re"));
  CHECK(dict.has_end("ing"));
  CHECK(dict.has_mid("port"));
  CHECK(dict.has_whole("the"));
  CHECK(dict.has_delimiter(" "));
}
