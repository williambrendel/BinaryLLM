// ============================================================================
// tests/core/codebook/test_codebook.cpp
//
// Codebook container: normalization + D cap, inverted index, union-decode,
// CBK1 round-trip.
// ============================================================================

#include "codebook.hpp"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "doctest.h"

using core::codebook::Atom;
using core::codebook::Codebook;

TEST_CASE("add normalizes: sorts, dedups, caps to max_width") {
  Codebook cb(100, 4);  // D = 4
  const std::size_t k = cb.add(Atom{5, 1, 5, 3, 9, 1, 7});  // dup + unsorted + >D
  const Atom& a = cb.atom(k);
  CHECK(a.size() == 4);                        // capped to D
  CHECK(std::is_sorted(a.begin(), a.end()));   // sorted
  CHECK(a == Atom{1, 3, 5, 7});                // top-4 lowest after dedup+sort
}

TEST_CASE("inverted index: atoms_with_bit") {
  Codebook cb(50, 8);
  const std::size_t a0 = cb.add(Atom{1, 2, 3});
  const std::size_t a1 = cb.add(Atom{2, 3, 4});
  const std::size_t a2 = cb.add(Atom{40});
  CHECK(cb.atoms_with_bit(2) == std::vector<std::uint32_t>{
                                    static_cast<std::uint32_t>(a0),
                                    static_cast<std::uint32_t>(a1)});
  CHECK(cb.atoms_with_bit(1) == std::vector<std::uint32_t>{
                                    static_cast<std::uint32_t>(a0)});
  CHECK(cb.atoms_with_bit(40) == std::vector<std::uint32_t>{
                                     static_cast<std::uint32_t>(a2)});
  CHECK(cb.atoms_with_bit(7).empty());   // unused position
  CHECK(cb.atoms_with_bit(999).empty()); // out of range
}

TEST_CASE("inverted index rebuilds after set()") {
  Codebook cb(50, 8);
  cb.add(Atom{1, 2});
  CHECK(cb.atoms_with_bit(2).size() == 1);
  cb.set(0, Atom{5, 6});
  CHECK(cb.atoms_with_bit(2).empty());
  CHECK(cb.atoms_with_bit(5).size() == 1);
}

TEST_CASE("decode: union of fired masks, sorted-unique") {
  Codebook cb(50, 8);
  cb.add(Atom{1, 2, 3});
  cb.add(Atom{3, 4, 5});
  cb.add(Atom{40, 41});
  CHECK(cb.decode({0, 1}) == Atom{1, 2, 3, 4, 5});      // overlap at 3 dedups
  CHECK(cb.decode({2}) == Atom{40, 41});
  CHECK(cb.decode({}).empty());
}

TEST_CASE("CBK1 round-trip") {
  Codebook cb(1000, 16);
  cb.add(Atom{1, 2, 3});
  cb.add(Atom{100, 200, 300, 400});
  cb.add(Atom{999});

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  cb.save(ss);

  const std::string bytes = ss.str();
  REQUIRE(bytes.size() >= 4);
  CHECK(bytes.substr(0, 4) == "CBK1");

  const Codebook r = Codebook::load(ss);
  CHECK(r.dim() == cb.dim());
  CHECK(r.max_width() == cb.max_width());
  REQUIRE(r.size() == cb.size());
  for (std::size_t k = 0; k < cb.size(); ++k) CHECK(r.atom(k) == cb.atom(k));
}
