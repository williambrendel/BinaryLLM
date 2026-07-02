// ============================================================================
// tests/core/signatures/test_encoder.cpp
// ============================================================================

#include "binary_vec.hpp"
#include "candidate_file.hpp"
#include "dict_augment.hpp"
#include "dictionary.hpp"
#include "doctest.h"
#include "encoder.hpp"
#include "tokenize.hpp"

#include <cstddef>
#include <sstream>
#include <vector>

using binarycore::binary_vec::BinaryVecDynamic;
using core::parts::augment_with_atoms;
using core::parts::Kind;
using core::parts::PartDictionary;
using core::parts::StreamToken;
using core::signatures::CandidateFileReader;
using core::signatures::CandidateFileWriter;
using core::signatures::encode;

namespace {

constexpr std::size_t kChunkSize =
    binarycore::binary_vec::BigSparseBinaryVecDynamic::chunk_size;

PartDictionary make_test_dict() {
  PartDictionary dict;
  dict.add(Kind::Whole, "the");
  dict.add(Kind::Whole, "cat");
  dict.add(Kind::Whole, "sat");
  dict.add(Kind::Whole, "on");
  dict.add(Kind::Whole, "mat");
  augment_with_atoms(dict);
  return dict;
}

std::size_t popcount(const BinaryVecDynamic& v) {
  std::size_t c = 0;
  for (const auto& chunk : v.chunks) c += chunk.data.size();
  return c;
}

// Count set bits whose global position falls in [lo, hi).
std::size_t popcount_band(const BinaryVecDynamic& v,
                          std::size_t lo, std::size_t hi) {
  std::size_t c = 0;
  for (std::size_t k = 0; k < v.chunks.size(); ++k) {
    for (std::uint16_t local : v.chunks[k].data) {
      const std::size_t pos = k * kChunkSize + local;
      if (pos >= lo && pos < hi) ++c;
    }
  }
  return c;
}

std::vector<StreamToken> words(std::initializer_list<const char*> ws) {
  std::vector<StreamToken> out;
  bool first = true;
  for (const char* w : ws) {
    if (!first) out.push_back({StreamToken::Type::Delimiter, " "});
    out.push_back({StreamToken::Type::Word, w});
    first = false;
  }
  return out;
}

}  // namespace

TEST_CASE("encode: empty input → empty output") {
  auto dict = make_test_dict();
  std::vector<StreamToken> empty;
  CHECK(encode(dict, empty).empty());
}

TEST_CASE("encode: output dim is 3F") {
  auto dict = make_test_dict();
  auto toks = words({"the"});
  auto sigs = encode(dict, toks);
  REQUIRE(sigs.size() == 1);
  CHECK(sigs[0].dim == 3 * dict.size());
}

TEST_CASE("encode: single word — only current band populated") {
  auto dict = make_test_dict();
  const std::size_t F = dict.size();
  auto toks = words({"the"});
  auto sigs = encode(dict, toks);
  REQUIRE(sigs.size() == 1);
  CHECK(popcount_band(sigs[0], 0, F)        == 0);  // before empty
  CHECK(popcount_band(sigs[0], F, 2 * F)     > 0);  // current present
  CHECK(popcount_band(sigs[0], 2 * F, 3 * F) == 0); // after empty
}

TEST_CASE("encode: before band grows, after band shrinks across positions") {
  auto dict = make_test_dict();
  const std::size_t F = dict.size();
  auto toks = words({"the", "cat", "sat", "on", "mat"});
  auto sigs = encode(dict, toks);
  REQUIRE(sigs.size() == 5);

  // before band: position 0 empty, monotonic non-decreasing.
  CHECK(popcount_band(sigs[0], 0, F) == 0);
  for (std::size_t i = 0; i + 1 < sigs.size(); ++i) {
    CHECK(popcount_band(sigs[i + 1], 0, F) >= popcount_band(sigs[i], 0, F));
  }
  // after band: last position empty, monotonic non-increasing.
  CHECK(popcount_band(sigs.back(), 2 * F, 3 * F) == 0);
  for (std::size_t i = 0; i + 1 < sigs.size(); ++i) {
    CHECK(popcount_band(sigs[i], 2 * F, 3 * F) >=
          popcount_band(sigs[i + 1], 2 * F, 3 * F));
  }
}

TEST_CASE("encode: chunk data stays sorted-ascending (invariant)") {
  auto dict = make_test_dict();
  auto toks = words({"the", "cat", "sat", "on", "mat"});
  auto sigs = encode(dict, toks);
  for (const auto& v : sigs) {
    for (const auto& chunk : v.chunks) {
      for (std::size_t k = 1; k < chunk.data.size(); ++k) {
        CHECK(chunk.data[k - 1] < chunk.data[k]);  // strictly ascending
      }
    }
  }
}

TEST_CASE("encode: delimiters between words are skipped") {
  auto dict = make_test_dict();
  std::vector<StreamToken> toks = {
    {StreamToken::Type::Word, "the"},
    {StreamToken::Type::Delimiter, " "},
    {StreamToken::Type::Delimiter, ","},
    {StreamToken::Type::Word, "cat"},
    {StreamToken::Type::Delimiter, " "},
    {StreamToken::Type::Word, "sat"},
  };
  CHECK(encode(dict, toks).size() == 3);
}

TEST_CASE("encode: range restricts to [start, end)") {
  auto dict = make_test_dict();
  auto toks = words({"the", "cat", "sat", "on", "mat"});
  // tokens layout: w d w d w d w d w  → indices 0..8, words at 0,2,4,6,8
  auto sigs = encode(dict, toks, 2, 7);  // words at 2,4,6 → 3 words
  CHECK(sigs.size() == 3);
}

// ---- Candidate file round-trip -----------------------------------

TEST_CASE("candidate_file: round-trip preserves chunk data") {
  auto dict = make_test_dict();
  auto toks = words({"the", "cat", "sat", "on", "mat"});
  auto sigs = encode(dict, toks);

  std::stringstream buf(std::ios::in | std::ios::out | std::ios::binary);
  {
    CandidateFileWriter w(buf,
                          static_cast<std::uint32_t>(3 * dict.size()),
                          static_cast<std::uint64_t>(sigs.size()));
    for (const auto& s : sigs) w.write(s);
  }

  CandidateFileReader r(buf);
  CHECK(r.dim() == 3 * dict.size());
  CHECK(r.record_count() == sigs.size());

  std::vector<BinaryVecDynamic> read_back;
  BinaryVecDynamic tmp(3 * dict.size());
  while (r.read(tmp)) read_back.push_back(tmp);

  REQUIRE(read_back.size() == sigs.size());
  for (std::size_t i = 0; i < sigs.size(); ++i) {
    CHECK(popcount(read_back[i]) == popcount(sigs[i]));
    for (std::size_t k = 0; k < sigs[i].chunks.size(); ++k) {
      CHECK(read_back[i].chunks[k].data == sigs[i].chunks[k].data);
    }
  }
}

TEST_CASE("candidate_file: bad magic throws") {
  std::istringstream bad("XXXX----");
  CHECK_THROWS(CandidateFileReader{bad});
}
