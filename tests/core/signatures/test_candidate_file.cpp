// ============================================================================
// tests/core/signatures/test_candidate_file.cpp
// ============================================================================

#include "binary_vec.hpp"
#include "candidate_file.hpp"
#include "doctest.h"

#include <algorithm>
#include <sstream>
#include <vector>

using core::signatures::CandidateFileReader;
using core::signatures::CandidateFileWriter;
using BVD = binarycore::binary_vec::BigSparseBinaryVecDynamic;

namespace {

BVD make_vec(std::uint32_t dim, const std::vector<std::size_t>& global_bits) {
  BVD v(dim);
  constexpr std::size_t CS = BVD::chunk_size;
  for (std::size_t g : global_bits) {
    v.chunks[g / CS].data.push_back(static_cast<std::uint16_t>(g % CS));
  }
  for (auto& chunk : v.chunks) std::sort(chunk.data.begin(), chunk.data.end());
  return v;
}

bool same_bits(const BVD& a, const BVD& b) {
  if (a.dim != b.dim) return false;
  if (a.chunks.size() != b.chunks.size()) return false;
  for (std::size_t c = 0; c < a.chunks.size(); ++c)
    if (a.chunks[c].data != b.chunks[c].data) return false;
  return true;
}

}  // namespace

TEST_CASE("candidate_file: empty file round-trips header") {
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  { CandidateFileWriter w(ss, 99312, 0); }
  CandidateFileReader r(ss);
  CHECK(r.dim() == 99312);
  CHECK(r.record_count() == 0);
  CHECK_FALSE(r.has_next());
  BVD out;
  CHECK_FALSE(r.read(out));
}

TEST_CASE("candidate_file: single small-dim record round-trips") {
  const std::uint32_t dim = 100;
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  BVD v = make_vec(dim, {3, 17, 42, 99});
  { CandidateFileWriter w(ss, dim, 1); w.write(v); }
  CandidateFileReader r(ss);
  REQUIRE(r.record_count() == 1);
  BVD got;
  REQUIRE(r.read(got));
  CHECK(same_bits(v, got));
  CHECK_FALSE(r.has_next());
}

TEST_CASE("candidate_file: multi-record multi-chunk (3F-scale) round-trips") {
  const std::uint32_t F = 33104;
  const std::uint32_t dim = 3 * F;  // 99312 -> 2 chunks
  std::vector<BVD> vecs = {
      make_vec(dim, {5, F + 10, 2 * F + 1}),
      make_vec(dim, {0, 65534, 65535, 99311}),  // straddles chunk 0/1 boundary
      make_vec(dim, {}),
  };
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  {
    CandidateFileWriter w(ss, dim, vecs.size());
    for (const auto& v : vecs) w.write(v);
    CHECK(w.written() == vecs.size());
  }
  CandidateFileReader r(ss);
  REQUIRE(r.dim() == dim);
  REQUIRE(r.record_count() == vecs.size());
  for (std::size_t i = 0; i < vecs.size(); ++i) {
    REQUIRE(r.has_next());
    BVD got;
    REQUIRE(r.read(got));
    CHECK(same_bits(vecs[i], got));
  }
  CHECK_FALSE(r.has_next());
}

TEST_CASE("candidate_file: dim mismatch on write throws") {
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  CandidateFileWriter w(ss, 100, 1);
  BVD wrong(200);
  CHECK_THROWS(w.write(wrong));
}

TEST_CASE("candidate_file: bad magic on read throws") {
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ss.write("XXXX", 4);
  ss.write("\0\0\0\0", 4);
  CHECK_THROWS(CandidateFileReader{ss});
}
