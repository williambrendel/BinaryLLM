// ============================================================================
// core/signatures/src/encoder.cpp
// ============================================================================

#include "encoder.hpp"

#include "tokenize.hpp"      // ascii_lowercase
#include "word_encoder.hpp"  // encode_word

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace core::signatures {

using binarycore::binary_vec::BinaryVecDynamic;
using core::parts::StreamToken;

namespace {

constexpr std::size_t kChunkSize =
    binarycore::binary_vec::BigSparseBinaryVecDynamic::chunk_size;

// Append bit `pos` to a 3F-dim vector. Caller restores per-chunk sort
// at the end (bands write into shared chunks out of order).
inline void push_bit(BinaryVecDynamic& v, std::size_t pos) {
  const std::size_t k = pos / kChunkSize;
  v.chunks[k].data.push_back(static_cast<std::uint16_t>(pos % kChunkSize));
}

}  // namespace

std::vector<BinaryVecDynamic> encode(
    const core::parts::PartDictionary& dict,
    std::span<const StreamToken> tokens,
    std::size_t start,
    std::size_t end) {
  const std::size_t F = dict.size();
  const std::size_t dim3 = 3 * F;
  if (end > tokens.size()) end = tokens.size();

  std::vector<std::size_t> word_idx;
  if (end > start) word_idx.reserve(end - start);
  for (std::size_t i = start; i < end; ++i) {
    if (tokens[i].type == StreamToken::Type::Word) word_idx.push_back(i);
  }
  const std::size_t N = word_idx.size();

  std::vector<BinaryVecDynamic> out;
  out.reserve(N);
  for (std::size_t i = 0; i < N; ++i) out.emplace_back(dim3);
  if (N == 0) return out;

  // Pass 1: current band (offset +F).
  std::vector<std::vector<std::uint32_t>> current(N);
  for (std::size_t i = 0; i < N; ++i) {
    const std::string w = core::parts::ascii_lowercase(tokens[word_idx[i]].value);
    current[i] = encode_word(dict, w);  // sorted-unique part ids in [0, F)
    for (std::uint32_t id : current[i]) push_bit(out[i], F + id);
  }

  // Pass 2: forward prefix-OR → before band (offset 0).
  std::vector<std::uint32_t> acc, merged;
  for (std::size_t i = 0; i < N; ++i) {
    for (std::uint32_t id : acc) push_bit(out[i], id);
    merged.clear();
    std::set_union(acc.begin(), acc.end(),
                   current[i].begin(), current[i].end(),
                   std::back_inserter(merged));
    acc.swap(merged);
  }

  // Pass 3: backward suffix-OR → after band (offset 2F).
  acc.clear();
  for (std::size_t i = N; i-- > 0; ) {
    for (std::uint32_t id : acc) push_bit(out[i], 2 * F + id);
    merged.clear();
    std::set_union(acc.begin(), acc.end(),
                   current[i].begin(), current[i].end(),
                   std::back_inserter(merged));
    acc.swap(merged);
  }

  // Bands wrote into shared chunks out of order; restore sort.
  for (auto& v : out) {
    for (auto& chunk : v.chunks) {
      std::sort(chunk.data.begin(), chunk.data.end());
    }
  }

  return out;
}

std::vector<BinaryVecDynamic> encode_windowed(
    const core::parts::PartDictionary& dict,
    std::span<const StreamToken> tokens,
    std::size_t radius,
    std::size_t start,
    std::size_t end) {
  const std::size_t F = dict.size();
  const std::size_t dim3 = 3 * F;
  if (end > tokens.size()) end = tokens.size();

  std::vector<std::size_t> word_idx;
  if (end > start) word_idx.reserve(end - start);
  for (std::size_t i = start; i < end; ++i) {
    if (tokens[i].type == StreamToken::Type::Word) word_idx.push_back(i);
  }
  const std::size_t N = word_idx.size();

  std::vector<BinaryVecDynamic> out;
  out.reserve(N);
  for (std::size_t i = 0; i < N; ++i) out.emplace_back(dim3);
  if (N == 0) return out;

  // current band (offset +F).
  std::vector<std::vector<std::uint32_t>> current(N);
  for (std::size_t i = 0; i < N; ++i) {
    const std::string w = core::parts::ascii_lowercase(tokens[word_idx[i]].value);
    current[i] = encode_word(dict, w);
    for (std::uint32_t id : current[i]) push_bit(out[i], F + id);
  }

  // before/after: OR-pool only over a bounded window of `radius` words each side.
  std::vector<std::uint32_t> acc, merged;
  auto pool_range = [&](std::size_t lo, std::size_t hi) -> const std::vector<std::uint32_t>& {
    acc.clear();
    for (std::size_t j = lo; j < hi; ++j) {
      merged.clear();
      std::set_union(acc.begin(), acc.end(), current[j].begin(), current[j].end(),
                     std::back_inserter(merged));
      acc.swap(merged);
    }
    return acc;
  };
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t lo = (i > radius) ? i - radius : 0;
    for (std::uint32_t id : pool_range(lo, i)) push_bit(out[i], id);           // before → [0,F)
    const std::size_t hi = std::min(N, i + radius + 1);
    for (std::uint32_t id : pool_range(i + 1, hi)) push_bit(out[i], 2 * F + id);  // after → [2F,3F)
  }

  for (auto& v : out) {
    for (auto& chunk : v.chunks) std::sort(chunk.data.begin(), chunk.data.end());
  }
  return out;
}

}  // namespace core::signatures
