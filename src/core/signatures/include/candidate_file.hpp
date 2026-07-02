#pragma once

// ============================================================================
// core/signatures/candidate_file.hpp
//
// Persist a stream of 3F candidate signatures to a compact binary file
// so the clustering stage can re-read them without re-encoding the
// corpus each run.
//
// Each candidate is a flat BinaryVecDynamic of dimension 3F (the banded
// before|current|after signature produced by encode(): before=[0,F),
// current=[F,2F), after=[2F,3F), global indices). Storing the whole 3F
// as ONE vector (not three F-dim bands) means a reader can feed it
// straight into jaccard() with no band-stitching — which is exactly
// what the Gonzales clusterer needs.
//
// On-disk format (little-endian):
//   magic        : 4 bytes  = "CND3"
//   version      : uint32   = 1
//   dim          : uint32   = 3F (signature dimension)
//   record_count : uint64
//   per record, per chunk c in [0, ceil(dim/65535)):
//     count      : uint32          (set bits in this chunk)
//     locals     : uint16 * count  (sorted ascending LOCAL indices)
//
// num_chunks is derived from dim, not stored. Pure binary: no floats.
// ============================================================================

#include "binary_vec.hpp"

#include <cstdint>
#include <iosfwd>

namespace core::signatures {

using BVD = binarycore::binary_vec::BigSparseBinaryVecDynamic;

inline constexpr char kCandMagic[4] = {'C', 'N', 'D', '3'};
inline constexpr std::uint32_t kCandVersion = 1;

class CandidateFileWriter {
 public:
  CandidateFileWriter(std::ostream& out, std::uint32_t dim,
                      std::uint64_t record_count);
  void write(const BVD& candidate);
  std::uint64_t written() const noexcept { return written_; }

 private:
  std::ostream& out_;
  std::uint32_t dim_;
  std::uint64_t record_count_;
  std::uint64_t written_ = 0;
};

class CandidateFileReader {
 public:
  explicit CandidateFileReader(std::istream& in);
  std::uint32_t dim() const noexcept { return dim_; }
  std::uint64_t record_count() const noexcept { return record_count_; }
  std::uint64_t read_so_far() const noexcept { return read_; }
  bool has_next() const noexcept { return read_ < record_count_; }
  bool read(BVD& out);

 private:
  std::istream& in_;
  std::uint32_t dim_ = 0;
  std::uint64_t record_count_ = 0;
  std::uint64_t read_ = 0;
};

}  // namespace core::signatures
