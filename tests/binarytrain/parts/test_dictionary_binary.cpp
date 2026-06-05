// ============================================================================
// tests/binarytrain/parts/test_dictionary_binary.cpp
//
// Coverage for PartDictionary's binary serialization (save_binary /
// load_binary). The text format save / load is covered elsewhere.
// ============================================================================

#include "binarytrain/parts/dictionary.hpp"
#include "binarytrain/parts/extractor.hpp"
#include "doctest.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using binarytrain::parts::Kind;
using binarytrain::parts::kInvalidPartId;
using binarytrain::parts::PartDictionary;
using binarytrain::parts::PartExtractor;

namespace {

// Build a tiny dict with each kind populated.
PartDictionary make_small_dict() {
  PartDictionary dict;
  dict.add(Kind::Whole, "the");
  dict.add(Kind::Whole, "a");
  dict.add(Kind::Start, "un");
  dict.add(Kind::Start, "re");
  dict.add(Kind::Mid, "port");
  dict.add(Kind::End, "ing");
  dict.add(Kind::End, "ed");
  dict.add(Kind::Letter, "a##");
  dict.add(Kind::Letter, "##a##");
  dict.add(Kind::Letter, "##a");
  dict.add(Kind::Delimiter, " ");
  dict.add(Kind::Delimiter, "\n");
  dict.add(Kind::Delimiter, ".");
  return dict;
}

// Serialize then deserialize via the binary format.
PartDictionary roundtrip_binary(const PartDictionary& dict) {
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  dict.save_binary(ss);
  ss.seekg(0);
  return PartDictionary::load_binary(ss);
}

}  // namespace

TEST_CASE("binary: header has correct magic, version, num_kinds") {
  PartDictionary dict;  // empty (no add() calls)
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  dict.save_binary(ss);

  std::string s = ss.str();
  REQUIRE(s.size() >= 12);

  // Magic.
  CHECK(s[0] == 'B');
  CHECK(s[1] == 'D');
  CHECK(s[2] == 'I');
  CHECK(s[3] == 'C');
  // Version (u16 LE = 1).
  CHECK(static_cast<unsigned char>(s[4]) == 0x01);
  CHECK(static_cast<unsigned char>(s[5]) == 0x00);
  // Flags (u16 LE = 0).
  CHECK(static_cast<unsigned char>(s[6]) == 0x00);
  CHECK(static_cast<unsigned char>(s[7]) == 0x00);
  // num_kinds (u32 LE = 6).
  CHECK(static_cast<unsigned char>(s[8])  == 0x06);
  CHECK(static_cast<unsigned char>(s[9])  == 0x00);
  CHECK(static_cast<unsigned char>(s[10]) == 0x00);
  CHECK(static_cast<unsigned char>(s[11]) == 0x00);
}

TEST_CASE("binary: empty dict round-trips to empty dict") {
  PartDictionary dict;
  PartDictionary loaded = roundtrip_binary(dict);
  CHECK(loaded.size() == 0);
  CHECK(loaded.count_of_kind(Kind::Whole) == 0);
  CHECK(loaded.count_of_kind(Kind::Start) == 0);
  CHECK(loaded.count_of_kind(Kind::Mid) == 0);
  CHECK(loaded.count_of_kind(Kind::End) == 0);
  CHECK(loaded.count_of_kind(Kind::Letter) == 0);
  CHECK(loaded.count_of_kind(Kind::Delimiter) == 0);
}

TEST_CASE("binary: small dict round-trip preserves counts") {
  auto dict = make_small_dict();
  auto loaded = roundtrip_binary(dict);

  CHECK(loaded.size() == dict.size());
  CHECK(loaded.count_of_kind(Kind::Whole) == dict.count_of_kind(Kind::Whole));
  CHECK(loaded.count_of_kind(Kind::Start) == dict.count_of_kind(Kind::Start));
  CHECK(loaded.count_of_kind(Kind::Mid)   == dict.count_of_kind(Kind::Mid));
  CHECK(loaded.count_of_kind(Kind::End)   == dict.count_of_kind(Kind::End));
  CHECK(loaded.count_of_kind(Kind::Letter) ==
        dict.count_of_kind(Kind::Letter));
  CHECK(loaded.count_of_kind(Kind::Delimiter) ==
        dict.count_of_kind(Kind::Delimiter));
}

TEST_CASE("binary: small dict round-trip preserves all (kind, value) pairs") {
  auto dict = make_small_dict();
  auto loaded = roundtrip_binary(dict);

  // Every key in the original must be present in the reload.
  CHECK(loaded.lookup(Kind::Whole, "the") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Whole, "a") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Start, "un") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Start, "re") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Mid, "port") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::End, "ing") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::End, "ed") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "a##") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##a##") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##a") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, " ") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "\n") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, ".") != kInvalidPartId);
}

TEST_CASE("binary: delimiter with embedded whitespace round-trips byte-for-byte") {
  // Newlines and tabs are escaped by the text format but stored
  // verbatim by the binary format. Confirms binary doesn't mangle.
  PartDictionary dict;
  dict.add(Kind::Delimiter, "\n");
  dict.add(Kind::Delimiter, "\n\n");
  dict.add(Kind::Delimiter, "\t");
  dict.add(Kind::Delimiter, " ");
  auto loaded = roundtrip_binary(dict);
  CHECK(loaded.lookup(Kind::Delimiter, "\n") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "\n\n") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, "\t") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Delimiter, " ") != kInvalidPartId);
}

TEST_CASE("binary: full backstop dict (extractor output) round-trips") {
  PartExtractor ex;
  // Add a few words so peel produces some Start/Mid/End parts.
  for (int i = 0; i < 30; ++i) {
    ex.add_word("running");
    ex.add_word("singing");
    ex.add_word("walking");
  }
  ex.add_delimiter(" ");
  ex.add_delimiter(".");
  PartDictionary dict = ex.finalize();
  auto loaded = roundtrip_binary(dict);

  CHECK(loaded.size() == dict.size());
  CHECK(loaded.count_of_kind(Kind::Whole) == dict.count_of_kind(Kind::Whole));
  CHECK(loaded.count_of_kind(Kind::Letter) ==
        dict.count_of_kind(Kind::Letter));
  CHECK(loaded.count_of_kind(Kind::Delimiter) ==
        dict.count_of_kind(Kind::Delimiter));

  // Specific atoms.
  CHECK(loaded.lookup(Kind::Letter, "a##") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##a##") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##a") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "-##") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##-##") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Letter, "##-") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Whole, "-") != kInvalidPartId);
  CHECK(loaded.lookup(Kind::Whole, "$") != kInvalidPartId);
}

TEST_CASE("binary: IDs stable within each kind across save+load") {
  // Atoms within a kind are written in ID order and re-added in that
  // order on load. So a relative ordering within a kind is preserved.
  PartDictionary dict;
  dict.add(Kind::Whole, "alpha");   // first whole
  dict.add(Kind::Whole, "beta");    // second whole
  dict.add(Kind::Whole, "gamma");   // third whole

  auto loaded = roundtrip_binary(dict);

  const auto a_id = loaded.lookup(Kind::Whole, "alpha");
  const auto b_id = loaded.lookup(Kind::Whole, "beta");
  const auto g_id = loaded.lookup(Kind::Whole, "gamma");
  REQUIRE(a_id != kInvalidPartId);
  REQUIRE(b_id != kInvalidPartId);
  REQUIRE(g_id != kInvalidPartId);
  CHECK(a_id < b_id);
  CHECK(b_id < g_id);
}

TEST_CASE("binary: bad magic throws") {
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ss.write("XYZW", 4);
  // Pad enough bytes to not be too short.
  for (int i = 0; i < 100; ++i) ss.put('\0');
  ss.seekg(0);
  CHECK_THROWS_AS(PartDictionary::load_binary(ss), std::runtime_error);
}

TEST_CASE("binary: unsupported version throws") {
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ss.write("BDIC", 4);
  // Version u16 LE = 99.
  unsigned char v[2] = {0x63, 0x00};
  ss.write(reinterpret_cast<const char*>(v), 2);
  for (int i = 0; i < 100; ++i) ss.put('\0');
  ss.seekg(0);
  CHECK_THROWS_AS(PartDictionary::load_binary(ss), std::runtime_error);
}

TEST_CASE("binary: format is more compact than text format") {
  // Use the full extractor output as a representative dict.
  PartExtractor ex;
  for (int i = 0; i < 30; ++i) {
    ex.add_word("running");
    ex.add_word("singing");
    ex.add_word("walking");
    ex.add_word("dancing");
  }
  PartDictionary dict = ex.finalize();

  std::stringstream binss(std::ios::in | std::ios::out | std::ios::binary);
  dict.save_binary(binss);
  const std::size_t bin_size = binss.str().size();

  std::stringstream textss;
  dict.save(textss);
  const std::size_t text_size = textss.str().size();

  // Binary should be meaningfully smaller. Text overhead per atom is
  // ~10 bytes (id digits + tab + kind name + tab + newline), while
  // binary overhead is 2 bytes (u16 length). So we expect roughly
  // half the size or better.
  CHECK(bin_size < text_size);
}
