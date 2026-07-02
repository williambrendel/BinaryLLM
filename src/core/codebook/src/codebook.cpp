// ============================================================================
// core/codebook/src/codebook.cpp
// ============================================================================

#include "codebook.hpp"

#include <algorithm>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace core::codebook {

namespace {

void put_u32(std::ostream& o, std::uint32_t v) {
  unsigned char b[4] = {static_cast<unsigned char>(v & 0xFF),
                        static_cast<unsigned char>((v >> 8) & 0xFF),
                        static_cast<unsigned char>((v >> 16) & 0xFF),
                        static_cast<unsigned char>((v >> 24) & 0xFF)};
  o.write(reinterpret_cast<const char*>(b), 4);
}
std::uint32_t get_u32(std::istream& i) {
  unsigned char b[4];
  i.read(reinterpret_cast<char*>(b), 4);
  return static_cast<std::uint32_t>(b[0]) |
         (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) |
         (static_cast<std::uint32_t>(b[3]) << 24);
}

const std::vector<std::uint32_t> kEmpty;

}  // namespace

void Codebook::normalize_(Atom& a) const {
  std::sort(a.begin(), a.end());
  a.erase(std::unique(a.begin(), a.end()), a.end());
  // Defensive: the learner pre-selects ≤ D bits; clip if a caller over-fills.
  if (max_width_ > 0 && a.size() > max_width_) a.resize(max_width_);
}

std::size_t Codebook::add(Atom bits) {
  normalize_(bits);
  atoms_.push_back(std::move(bits));
  index_built_ = false;
  return atoms_.size() - 1;
}

void Codebook::set(std::size_t k, Atom bits) {
  normalize_(bits);
  atoms_[k] = std::move(bits);
  index_built_ = false;
}

void Codebook::ensure_index_() const {
  if (index_built_) return;
  index_.assign(dim_, {});
  for (std::uint32_t k = 0; k < atoms_.size(); ++k)
    for (std::uint32_t pos : atoms_[k])
      if (pos < dim_) index_[pos].push_back(k);
  index_built_ = true;
}

const std::vector<std::uint32_t>& Codebook::atoms_with_bit(
    std::uint32_t pos) const {
  ensure_index_();
  if (pos >= dim_) return kEmpty;
  return index_[pos];
}

std::vector<std::uint32_t> Codebook::decode(
    const std::vector<std::uint32_t>& fired) const {
  std::vector<std::uint32_t> out;
  for (std::uint32_t k : fired)
    if (k < atoms_.size())
      out.insert(out.end(), atoms_[k].begin(), atoms_[k].end());
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

void Codebook::save(std::ostream& out) const {
  out.write(kCbkMagic, 4);
  put_u32(out, kCbkVersion);
  put_u32(out, static_cast<std::uint32_t>(dim_));
  put_u32(out, max_width_);
  put_u32(out, static_cast<std::uint32_t>(atoms_.size()));
  for (const Atom& a : atoms_) {
    put_u32(out, static_cast<std::uint32_t>(a.size()));
    for (std::uint32_t pos : a) put_u32(out, pos);
  }
}

Codebook Codebook::load(std::istream& in) {
  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || std::memcmp(magic, kCbkMagic, 4) != 0)
    throw std::runtime_error("Codebook::load: bad magic (not CBK1)");
  if (get_u32(in) != kCbkVersion)
    throw std::runtime_error("Codebook::load: unsupported version");
  const std::uint32_t dim = get_u32(in);
  const std::uint32_t max_width = get_u32(in);
  const std::uint32_t count = get_u32(in);
  Codebook cb(dim, max_width);
  cb.atoms_.reserve(count);
  for (std::uint32_t k = 0; k < count; ++k) {
    const std::uint32_t width = get_u32(in);
    Atom a(width);
    for (std::uint32_t j = 0; j < width; ++j) a[j] = get_u32(in);
    cb.atoms_.push_back(std::move(a));  // already sorted on disk; trust writer
  }
  return cb;
}

}  // namespace core::codebook
