// ============================================================================
// tests/core/parts/test_dict_reorder.cpp
//
// Tests for dict_reorder.hpp side functions and end-to-end behavior:
// reordering the dict + running decompose_word produces different
// decompositions for the same word.
//
// Includes the canonical "catastrophe" example demonstrating that
// strategy reorderings yield different bags.
// ============================================================================

#include "decomposer.hpp"
#include "dict_reorder.hpp"
#include "dictionary.hpp"
#include "doctest.h"

#include <vector>

using core::parts::decompose_word;
using core::parts::Kind;
using core::parts::PartDictionary;
using core::parts::reorder_dict_by_type;
using core::parts::reorder_dict_by_type_then_level;
using core::parts::reorder_dict_by_level_then_type;

namespace {

void add_letter_atoms(PartDictionary& dict) {
  for (char c = 'a'; c <= 'z'; ++c) {
    const std::string s(1, c);
    dict.add(Kind::Letter, s + "##");
    dict.add(Kind::Letter, "##" + s + "##");
    dict.add(Kind::Letter, "##" + s);
  }
}

// Build the "catastrophe" dict from the design discussion. Insertion
// order:
//   cata##  ##stro##  ##phe  cat##  ##str##  ##ophe
PartDictionary make_catastrophe_dict() {
  PartDictionary dict;
  add_letter_atoms(dict);  // backstop
  dict.add(Kind::Start, "cata");
  dict.add(Kind::Mid,   "stro");
  dict.add(Kind::End,   "phe");
  dict.add(Kind::Start, "cat");
  dict.add(Kind::Mid,   "str");
  dict.add(Kind::End,   "ophe");
  return dict;
}

}  // namespace

// ---------------- reorder_by_type ----------------------------------

TEST_CASE("reorder_by_type: places S, then E, then M") {
  auto src = make_catastrophe_dict();
  auto reordered = reorder_dict_by_type(src);

  // Same parts must be reachable.
  CHECK(reordered.lookup(Kind::Start, "cata") != core::parts::kInvalidPartId);
  CHECK(reordered.lookup(Kind::Start, "cat")  != core::parts::kInvalidPartId);
  CHECK(reordered.lookup(Kind::End,   "phe")  != core::parts::kInvalidPartId);
  CHECK(reordered.lookup(Kind::End,   "ophe") != core::parts::kInvalidPartId);
  CHECK(reordered.lookup(Kind::Mid,   "stro") != core::parts::kInvalidPartId);
  CHECK(reordered.lookup(Kind::Mid,   "str")  != core::parts::kInvalidPartId);

  // S/M/E IDs ordered by kind, original sequence within kind.
  // Among the S/M/E parts (excluding Letter atoms), the new ID order
  // is: cata, cat, phe, ophe, stro, str.
  const auto id_cata = reordered.lookup(Kind::Start, "cata");
  const auto id_cat  = reordered.lookup(Kind::Start, "cat");
  const auto id_phe  = reordered.lookup(Kind::End,   "phe");
  const auto id_ophe = reordered.lookup(Kind::End,   "ophe");
  const auto id_stro = reordered.lookup(Kind::Mid,   "stro");
  const auto id_str  = reordered.lookup(Kind::Mid,   "str");

  CHECK(id_cata < id_cat);
  CHECK(id_cat  < id_phe);
  CHECK(id_phe  < id_ophe);
  CHECK(id_ophe < id_stro);
  CHECK(id_stro < id_str);
}

// ---------------- reorder_by_type_then_level -----------------------

TEST_CASE("reorder_by_type_then_level: kind first, longest L within") {
  auto src = make_catastrophe_dict();
  auto reordered = reorder_dict_by_type_then_level(src);

  // Expected S/M/E order:
  //   Starts (longest first): cata (L=4), cat (L=3)
  //   Ends   (longest first): ophe (L=4), phe (L=3)
  //   Mids   (longest first): stro (L=4), str (L=3)
  const auto id_cata = reordered.lookup(Kind::Start, "cata");
  const auto id_cat  = reordered.lookup(Kind::Start, "cat");
  const auto id_ophe = reordered.lookup(Kind::End,   "ophe");
  const auto id_phe  = reordered.lookup(Kind::End,   "phe");
  const auto id_stro = reordered.lookup(Kind::Mid,   "stro");
  const auto id_str  = reordered.lookup(Kind::Mid,   "str");

  CHECK(id_cata < id_cat);
  CHECK(id_cat  < id_ophe);
  CHECK(id_ophe < id_phe);
  CHECK(id_phe  < id_stro);
  CHECK(id_stro < id_str);
}

// ---------------- reorder_by_level_then_type -----------------------

TEST_CASE("reorder_by_level_then_type: L first, then S, E, M within") {
  auto src = make_catastrophe_dict();
  auto reordered = reorder_dict_by_level_then_type(src);

  // Expected S/M/E order:
  //   L=4: cata (S), ophe (E), stro (M)
  //   L=3: cat  (S), phe  (E), str  (M)
  const auto id_cata = reordered.lookup(Kind::Start, "cata");
  const auto id_ophe = reordered.lookup(Kind::End,   "ophe");
  const auto id_stro = reordered.lookup(Kind::Mid,   "stro");
  const auto id_cat  = reordered.lookup(Kind::Start, "cat");
  const auto id_phe  = reordered.lookup(Kind::End,   "phe");
  const auto id_str  = reordered.lookup(Kind::Mid,   "str");

  CHECK(id_cata < id_ophe);
  CHECK(id_ophe < id_stro);
  CHECK(id_stro < id_cat);
  CHECK(id_cat  < id_phe);
  CHECK(id_phe  < id_str);
}

// ---------------- End-to-end: catastrophe trace --------------------

TEST_CASE("catastrophe (original): peel cata → stro → phe") {
  auto dict = make_catastrophe_dict();
  // Dict order: cata, stro, phe, cat, str, ophe.
  // Iterate:
  //   Start "cata": prefix matches → claim [0,4)
  //   Mid "stro":   appears at pos 4 in "catastrophe" → claim [4,8)
  //   End "phe":    suffix matches → claim [8,11)
  //   Remaining S/M/E blocked or skipped.
  // No letters needed.
  auto ids = decompose_word(dict, "catastrophe");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "cata"));
  CHECK(ids[1] == dict.lookup(Kind::Mid,   "stro"));
  CHECK(ids[2] == dict.lookup(Kind::End,   "phe"));
}

TEST_CASE("catastrophe (type-level reorder): peel cata → ophe → str") {
  auto dict = reorder_dict_by_type_then_level(make_catastrophe_dict());
  // New dict S/E/M order:
  //   Starts: cata, cat
  //   Ends:   ophe, phe
  //   Mids:   stro, str
  // Iterate:
  //   Start "cata": claim [0,4)  → word now [4,11) free
  //   Start "cat":  skip (start claimed)
  //   End "ophe":   suffix "ophe" matches [7,11) → claim
  //   End "phe":    skip (end claimed)
  //   Mid "stro":   "stro" in remaining [4,7) = "str" → no match
  //   Mid "str":    "str" in remaining [4,7) → claim [4,7)
  // No letters needed.
  auto ids = decompose_word(dict, "catastrophe");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "cata"));
  CHECK(ids[1] == dict.lookup(Kind::Mid,   "str"));
  CHECK(ids[2] == dict.lookup(Kind::End,   "ophe"));
}

TEST_CASE("catastrophe (level-type reorder): peel cata → ophe → str") {
  auto dict = reorder_dict_by_level_then_type(make_catastrophe_dict());
  // New S/M/E order: cata, ophe, stro, cat, phe, str.
  // Iterate:
  //   Start "cata": claim [0,4)
  //   End "ophe":   suffix matches → claim [7,11)
  //   Mid "stro":   in remaining [4,7) = "str"? no
  //   Start "cat":  skip
  //   End "phe":    skip
  //   Mid "str":    in [4,7) = "str" → match, claim
  auto ids = decompose_word(dict, "catastrophe");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "cata"));
  CHECK(ids[1] == dict.lookup(Kind::Mid,   "str"));
  CHECK(ids[2] == dict.lookup(Kind::End,   "ophe"));
}

TEST_CASE("catastrophe (type reorder): peel cata → phe → stro") {
  auto dict = reorder_dict_by_type(make_catastrophe_dict());
  // New S/M/E order: cata, cat, phe, ophe, stro, str.
  // Iterate:
  //   Start "cata": claim [0,4)
  //   Start "cat":  skip
  //   End "phe":    suffix matches → claim [8,11)
  //   End "ophe":   skip
  //   Mid "stro":   in remaining [4,8) = "stro" → match, claim
  //   Mid "str":    blocked
  auto ids = decompose_word(dict, "catastrophe");
  REQUIRE(ids.size() == 3);
  CHECK(ids[0] == dict.lookup(Kind::Start, "cata"));
  CHECK(ids[1] == dict.lookup(Kind::Mid,   "stro"));
  CHECK(ids[2] == dict.lookup(Kind::End,   "phe"));
}

// ---------------- Idempotence / preservation -----------------------

TEST_CASE("reorder: F unchanged after reorder") {
  auto src = make_catastrophe_dict();
  CHECK(reorder_dict_by_type(src).size()             == src.size());
  CHECK(reorder_dict_by_type_then_level(src).size()  == src.size());
  CHECK(reorder_dict_by_level_then_type(src).size()  == src.size());
}
