// ============================================================================
// tests/core/parts/test_dict_io.cpp
// ============================================================================

#include "dict_augment.hpp"
#include "dict_io.hpp"
#include "dictionary.hpp"
#include "doctest.h"

#include <sstream>
#include <string>

using core::parts::augment_with_atoms;
using core::parts::Kind;
using core::parts::kInvalidPartId;
using core::parts::load_dict_binary;
using core::parts::load_dict_text;
using core::parts::PartDictionary;
using core::parts::save_dict_binary;
using core::parts::save_dict_text;

namespace {

// Builds a dict containing trained content + every augment-set part
// already inserted (the runtime dict an extractor would produce).
PartDictionary make_runtime_dict() {
  PartDictionary dict;

  // Trained multi-char Wholes.
  dict.add(Kind::Whole, "the");
  dict.add(Kind::Whole, "am");

  // Augment set (singletons + specials) — would also be added by
  // the extractor inline.
  augment_with_atoms(dict);

  // Trained Start / Mid / End.
  dict.add(Kind::Start, "cata");
  dict.add(Kind::Start, "un");
  dict.add(Kind::Mid,   "stro");
  dict.add(Kind::Mid,   "port");
  dict.add(Kind::End,   "phe");
  dict.add(Kind::End,   "ing");

  // Observed delimiters (not in augment set — newline, space).
  dict.add(Kind::Delimiter, " ");
  dict.add(Kind::Delimiter, "\n");

  return dict;
}

}  // namespace

// ---- augment_with_atoms ------------------------------------------

TEST_CASE("augment_with_atoms: adds 42 single-char Wholes") {
  PartDictionary dict;
  augment_with_atoms(dict);
  CHECK(dict.count_of_kind(Kind::Whole) == 42);  // 26 + 10 + 6
  CHECK(dict.lookup(Kind::Whole, "a") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "0") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Whole, "-") != kInvalidPartId);
}

TEST_CASE("augment_with_atoms: adds 126 positional Letter atoms") {
  PartDictionary dict;
  augment_with_atoms(dict);
  CHECK(dict.count_of_kind(Kind::Letter) == 126);  // (26+10+6)*3
  CHECK(dict.lookup(Kind::Letter, "a##")   != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "##a##") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "##a")   != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "0##")   != kInvalidPartId);
  CHECK(dict.lookup(Kind::Letter, "-##")   != kInvalidPartId);
}

TEST_CASE("augment_with_atoms: adds 6 connector Delimiters") {
  PartDictionary dict;
  augment_with_atoms(dict);
  CHECK(dict.count_of_kind(Kind::Delimiter) == 6);
  CHECK(dict.lookup(Kind::Delimiter, "-") != kInvalidPartId);
  CHECK(dict.lookup(Kind::Delimiter, "$") != kInvalidPartId);
}

TEST_CASE("augment_with_atoms: idempotent on second call") {
  PartDictionary dict;
  augment_with_atoms(dict);
  const std::size_t f1 = dict.size();
  augment_with_atoms(dict);
  CHECK(dict.size() == f1);
}

// ---- Text round-trip ---------------------------------------------

TEST_CASE("dict_io text: round-trip preserves trained content + augments on load") {
  auto orig = make_runtime_dict();

  std::ostringstream out;
  save_dict_text(orig, out);

  std::istringstream in(out.str());
  auto loaded = load_dict_text(in);

  // Every trained part must round-trip.
  CHECK(loaded.lookup(Kind::Whole, "the") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Whole, "am")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Start, "cata") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Start, "un")   != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Mid,   "stro") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Mid,   "port") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::End,   "phe")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::End,   "ing")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, " ")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "\n") != kInvalidPartId);

  // Augment set must be present after load.
  CHECK(loaded.lookup(Kind::Whole, "a")     != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "a##")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##a##")!= kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##a")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "-") != kInvalidPartId);

  // Per-kind counts should match. Augment is idempotent so totals
  // align (Letter and connector-Delim sets are identical; Whole set
  // identical too).
  CHECK(loaded.count_of_kind(Kind::Whole)     == orig.count_of_kind(Kind::Whole));
  CHECK(loaded.count_of_kind(Kind::Start)     == orig.count_of_kind(Kind::Start));
  CHECK(loaded.count_of_kind(Kind::Mid)       == orig.count_of_kind(Kind::Mid));
  CHECK(loaded.count_of_kind(Kind::End)       == orig.count_of_kind(Kind::End));
  CHECK(loaded.count_of_kind(Kind::Letter)    == orig.count_of_kind(Kind::Letter));
  CHECK(loaded.count_of_kind(Kind::Delimiter) == orig.count_of_kind(Kind::Delimiter));
}

TEST_CASE("dict_io text: emits ## convention and excludes augment set") {
  auto orig = make_runtime_dict();
  std::ostringstream out;
  save_dict_text(orig, out);
  // Prepend "\n" so the first line is also preceded by a newline,
  // letting us use "\nVALUE\n" patterns uniformly.
  const std::string s = "\n" + out.str();

  // Trained content present in expected encoding.
  CHECK(s.find("\nthe\n")     != std::string::npos);  // Whole
  CHECK(s.find("\ncata##\n")  != std::string::npos);  // Start
  CHECK(s.find("\n##stro##\n")!= std::string::npos);  // Mid
  CHECK(s.find("\n##phe\n")   != std::string::npos);  // End
  CHECK(s.find("[delim]")     != std::string::npos);

  // No single-char alphabetic Whole atoms in file.
  CHECK(s.find("\na\n")  == std::string::npos);
  CHECK(s.find("\nz\n")  == std::string::npos);

  // No Letter atoms in file (the "a##" line shape).
  CHECK(s.find("\na##\n")   == std::string::npos);
  CHECK(s.find("\n##a##\n") == std::string::npos);
  CHECK(s.find("\n##a\n")   == std::string::npos);
}

TEST_CASE("dict_io text: delimiter escapes round-trip") {
  PartDictionary dict;
  dict.add(Kind::Whole, "hi");
  dict.add(Kind::Delimiter, "\n");
  dict.add(Kind::Delimiter, "\t");
  dict.add(Kind::Delimiter, " ");
  dict.add(Kind::Delimiter, "\\");
  augment_with_atoms(dict);  // mimic runtime layout

  std::ostringstream out;
  save_dict_text(dict, out);
  std::istringstream in(out.str());
  auto loaded = load_dict_text(in);

  CHECK(loaded.lookup(Kind::Delimiter, "\n") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "\t") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, " ")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "\\") != kInvalidPartId);
}

// ---- Binary round-trip -------------------------------------------

TEST_CASE("dict_io binary: round-trip preserves trained content + augments on load") {
  auto orig = make_runtime_dict();

  std::ostringstream out(std::ios::binary);
  save_dict_binary(orig, out);

  std::istringstream in(out.str(), std::ios::binary);
  auto loaded = load_dict_binary(in);

  CHECK(loaded.lookup(Kind::Whole, "the")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Start, "cata") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Mid,   "stro") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::End,   "phe")  != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "\n") != kInvalidPartId);

  // Augment set.
  CHECK(loaded.lookup(Kind::Whole, "a")    != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "a##") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "-") != kInvalidPartId);

  CHECK(loaded.size() == orig.size());
}

TEST_CASE("dict_io binary: magic guards bad headers") {
  std::istringstream in("XXXX", std::ios::binary);
  CHECK_THROWS(load_dict_binary(in));
}