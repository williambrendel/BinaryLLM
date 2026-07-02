// ============================================================================
// core/signatures/src/weight_table.cpp
// ============================================================================

#include "weight_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace core::signatures {

namespace {

constexpr std::size_t kChunkSize = BinaryVecDynamic::chunk_size;

void put_u32(std::ostream& o, std::uint32_t v) {
  unsigned char b[4] = {static_cast<unsigned char>(v & 0xFF),
                        static_cast<unsigned char>((v >> 8) & 0xFF),
                        static_cast<unsigned char>((v >> 16) & 0xFF),
                        static_cast<unsigned char>((v >> 24) & 0xFF)};
  o.write(reinterpret_cast<const char*>(b), 4);
}
void put_u16(std::ostream& o, std::uint16_t v) {
  unsigned char b[2] = {static_cast<unsigned char>(v & 0xFF),
                        static_cast<unsigned char>((v >> 8) & 0xFF)};
  o.write(reinterpret_cast<const char*>(b), 2);
}
std::uint32_t get_u32(std::istream& i) {
  unsigned char b[4];
  i.read(reinterpret_cast<char*>(b), 4);
  return static_cast<std::uint32_t>(b[0]) |
         (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) |
         (static_cast<std::uint32_t>(b[3]) << 24);
}
std::uint16_t get_u16(std::istream& i) {
  unsigned char b[2];
  i.read(reinterpret_cast<char*>(b), 2);
  return static_cast<std::uint16_t>(b[0]) |
         (static_cast<std::uint16_t>(b[1]) << 8);
}

}  // namespace

void SurprisalCounter::add(const BinaryVecDynamic& sig) {
  for (std::size_t k = 0; k < sig.chunks.size(); ++k) {
    const std::size_t base = k * kChunkSize;
    for (std::uint16_t local : sig.chunks[k].data) {
      const std::size_t g = base + local;
      if (g < df_.size()) ++df_[g];
    }
  }
  ++n_;
}

SurprisalTable SurprisalCounter::finalize(std::uint8_t quant_cap) const {
  SurprisalTable t;
  t.dim = static_cast<std::uint32_t>(df_.size());
  t.quant_cap = quant_cap == 0 ? 1 : quant_cap;
  t.weights.assign(df_.size(), 0);
  if (n_ == 0) return t;

  const double n = static_cast<double>(n_);
  // Peak surprisal over observed bits, for linear scaling into [1, cap].
  double w_max = 0.0;
  for (std::uint64_t d : df_) {
    if (d == 0) continue;
    const double w = std::log(n / static_cast<double>(d));
    if (w > w_max) w_max = w;
  }

  const double cap = static_cast<double>(t.quant_cap);
  for (std::size_t e = 0; e < df_.size(); ++e) {
    const std::uint64_t d = df_[e];
    if (d == 0) continue;                 // never fires: weight unused → 0
    if (w_max <= 0.0) {                    // every bit fires in every token
      t.weights[e] = 1;
      continue;
    }
    const double w = std::log(n / static_cast<double>(d));
    // w ∈ [0, w_max] → multiplicity ∈ [1, cap]; common bit → 1, rarest → cap.
    long m = std::lround(1.0 + (cap - 1.0) * (w / w_max));
    m = std::clamp<long>(m, 1, t.quant_cap);
    t.weights[e] = static_cast<std::uint16_t>(m);
  }
  return t;
}

void save_surprisal(const SurprisalTable& table, std::ostream& out) {
  out.write(kWgtMagic, 4);
  put_u32(out, kWgtVersion);
  put_u32(out, table.dim);
  const unsigned char cap_pad[4] = {table.quant_cap, 0, 0, 0};
  out.write(reinterpret_cast<const char*>(cap_pad), 4);
  if (table.weights.size() != table.dim) {
    throw std::runtime_error("save_surprisal: weights.size() != dim");
  }
  for (std::uint16_t w : table.weights) put_u16(out, w);
}

SurprisalTable load_surprisal(std::istream& in) {
  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 || std::memcmp(magic, kWgtMagic, 4) != 0) {
    throw std::runtime_error("load_surprisal: bad magic (not WGT1)");
  }
  const std::uint32_t version = get_u32(in);
  if (version != kWgtVersion) {
    throw std::runtime_error("load_surprisal: unsupported version");
  }
  SurprisalTable t;
  t.dim = get_u32(in);
  unsigned char cap_pad[4];
  in.read(reinterpret_cast<char*>(cap_pad), 4);
  t.quant_cap = cap_pad[0];
  t.weights.resize(t.dim);
  for (std::uint32_t e = 0; e < t.dim; ++e) t.weights[e] = get_u16(in);
  return t;
}

}  // namespace core::signatures
