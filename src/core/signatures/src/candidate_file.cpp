// ============================================================================
// core/signatures/src/candidate_file.cpp
// ============================================================================

#include "candidate_file.hpp"

#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace core::signatures {

namespace {

constexpr std::size_t kChunkSize = BVD::chunk_size;

std::size_t num_chunks_for_dim(std::uint32_t dim) {
  return (static_cast<std::size_t>(dim) + kChunkSize - 1) / kChunkSize;
}

void put_u32(std::ostream& o, std::uint32_t v) {
  unsigned char b[4] = {static_cast<unsigned char>(v & 0xFF),
                        static_cast<unsigned char>((v >> 8) & 0xFF),
                        static_cast<unsigned char>((v >> 16) & 0xFF),
                        static_cast<unsigned char>((v >> 24) & 0xFF)};
  o.write(reinterpret_cast<const char*>(b), 4);
}
void put_u64(std::ostream& o, std::uint64_t v) {
  unsigned char b[8];
  for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xFF);
  o.write(reinterpret_cast<const char*>(b), 8);
}
void put_u16(std::ostream& o, std::uint16_t v) {
  unsigned char b[2] = {static_cast<unsigned char>(v & 0xFF),
                        static_cast<unsigned char>((v >> 8) & 0xFF)};
  o.write(reinterpret_cast<const char*>(b), 2);
}
std::uint32_t get_u32(std::istream& i) {
  unsigned char b[4]; i.read(reinterpret_cast<char*>(b), 4);
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}
std::uint64_t get_u64(std::istream& i) {
  unsigned char b[8]; i.read(reinterpret_cast<char*>(b), 8);
  std::uint64_t v = 0;
  for (int k = 0; k < 8; ++k) v |= static_cast<std::uint64_t>(b[k]) << (8 * k);
  return v;
}
std::uint16_t get_u16(std::istream& i) {
  unsigned char b[2]; i.read(reinterpret_cast<char*>(b), 2);
  return static_cast<std::uint16_t>(b[0]) | (static_cast<std::uint16_t>(b[1]) << 8);
}

}  // namespace

CandidateFileWriter::CandidateFileWriter(std::ostream& out, std::uint32_t dim,
                                         std::uint64_t record_count)
    : out_(out), dim_(dim), record_count_(record_count) {
  out_.write(kCandMagic, 4);
  put_u32(out_, kCandVersion);
  put_u32(out_, dim_);
  put_u64(out_, record_count_);
}

void CandidateFileWriter::write(const BVD& candidate) {
  if (candidate.dim != dim_) {
    throw std::runtime_error("CandidateFileWriter: dim mismatch");
  }
  const std::size_t nchunks = num_chunks_for_dim(dim_);
  for (std::size_t c = 0; c < nchunks; ++c) {
    if (c < candidate.chunks.size()) {
      const auto& data = candidate.chunks[c].data;
      put_u32(out_, static_cast<std::uint32_t>(data.size()));
      for (std::uint16_t local : data) put_u16(out_, local);
    } else {
      put_u32(out_, 0);
    }
  }
  ++written_;
}

CandidateFileReader::CandidateFileReader(std::istream& in) : in_(in) {
  char magic[4];
  in_.read(magic, 4);
  if (in_.gcount() != 4 || std::memcmp(magic, kCandMagic, 4) != 0) {
    throw std::runtime_error("CandidateFileReader: bad magic (not CND3)");
  }
  const std::uint32_t version = get_u32(in_);
  if (version != kCandVersion) {
    throw std::runtime_error("CandidateFileReader: unsupported version");
  }
  dim_ = get_u32(in_);
  record_count_ = get_u64(in_);
}

bool CandidateFileReader::read(BVD& out) {
  if (read_ >= record_count_) return false;
  out = BVD(dim_);
  const std::size_t nchunks = num_chunks_for_dim(dim_);
  for (std::size_t c = 0; c < nchunks; ++c) {
    const std::uint32_t count = get_u32(in_);
    auto& data = out.chunks[c].data;
    data.clear();
    data.reserve(count);
    for (std::uint32_t k = 0; k < count; ++k) data.push_back(get_u16(in_));
  }
  ++read_;
  return true;
}

}  // namespace core::signatures
