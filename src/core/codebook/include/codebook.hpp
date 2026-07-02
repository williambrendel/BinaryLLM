#pragma once

// ============================================================================
// core/codebook/codebook.hpp
//
// A learned sparse binary codebook: K atoms (codewords), each a sorted-unique
// list of ≤ D bit positions in the 3F space. This is the object pass 1 learns
// (spec §4) — distinct from parts::PartDictionary (the subword vocabulary).
//
// Container responsibilities only: storage, the D-width invariant, an inverted
// index (bit → atoms) that powers sublinear pursuit and the incoherence
// penalty, union-decode (spec §7.5.1), and CBK1 serialization. Learning lives
// in pass1.hpp; the α-threshold encoder in pursuit.hpp.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace core::codebook {

// One codeword: sorted, unique 3F bit positions, size ≤ max_width (D).
using Atom = std::vector<std::uint32_t>;

inline constexpr char kCbkMagic[4] = {'C', 'B', 'K', '1'};
inline constexpr std::uint32_t kCbkVersion = 1;

class Codebook {
 public:
  Codebook() = default;
  Codebook(std::size_t dim, std::uint32_t max_width)
      : dim_(dim), max_width_(max_width) {}

  std::size_t dim() const noexcept { return dim_; }
  std::uint32_t max_width() const noexcept { return max_width_; }
  std::size_t size() const noexcept { return atoms_.size(); }  // K

  const Atom& atom(std::size_t k) const { return atoms_[k]; }
  const std::vector<Atom>& atoms() const noexcept { return atoms_; }

  // Add / replace an atom. Input is normalized (sorted, deduped) and clipped to
  // the top max_width positions if longer. Invalidates the inverted index.
  std::size_t add(Atom bits);
  void set(std::size_t k, Atom bits);

  // Inverted index: atoms containing a given bit position. Lazily built;
  // add()/set() invalidate it. Returns an empty list for unused positions.
  const std::vector<std::uint32_t>& atoms_with_bit(std::uint32_t pos) const;

  // Union-decode: OR of the fired atoms' masks → sorted-unique 3F positions
  // (the non-learned decode / generation head, spec §7.5.1).
  std::vector<std::uint32_t> decode(const std::vector<std::uint32_t>& fired) const;

  // CBK1 binary format (little-endian):
  //   magic | version u32 | dim u32 | max_width u32 | count u32
  //   per atom: width u32, then width × u32 positions
  void save(std::ostream& out) const;
  static Codebook load(std::istream& in);

 private:
  std::size_t dim_ = 0;
  std::uint32_t max_width_ = 0;
  std::vector<Atom> atoms_;

  mutable bool index_built_ = false;
  mutable std::vector<std::vector<std::uint32_t>> index_;  // size dim_

  void normalize_(Atom& a) const;  // sort, unique, clip to max_width_
  void ensure_index_() const;
};

}  // namespace core::codebook
