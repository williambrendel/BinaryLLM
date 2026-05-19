#pragma once

// ============================================================================
// binarycore/containers/detail/binary_vec_128.hpp
// ----------------------------------------------------------------------------
// Explicit specialization of BinaryVec<128>.
//
// Two chunks. (i >> 6) selects chunk 0 or 1; (i & 63) is the bit within it.
// All operations are pairs of uint64_t ops — straight-line code, no loops.
//
// On ARM NEON, the compiler may fuse the two operations into a single 128-bit
// SIMD instruction since the std::array storage is contiguous.
// On x86 SSE2, similarly.
// ============================================================================

#include "binary_vec_generic.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace binarycore {

template <>
struct BinaryVec<128> {
  static constexpr std::size_t Dims = 128;
  static constexpr std::size_t Chunks = 2;
  std::array<uint64_t, 2> data{};

  constexpr bool get(std::size_t i) const noexcept {
    return (data[i >> 6] >> (i & 63)) & 1ULL;
  }
  constexpr void set(std::size_t i) noexcept {
    data[i >> 6] |= (1ULL << (i & 63));
  }
  constexpr void clear(std::size_t i) noexcept {
    data[i >> 6] &= ~(1ULL << (i & 63));
  }
  constexpr void assign(std::size_t i, bool v) noexcept {
    if (v)
      set(i);
    else
      clear(i);
  }

  constexpr BinaryVec operator&(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] & o.data[0], data[1] & o.data[1]}};
  }
  constexpr BinaryVec operator|(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] | o.data[0], data[1] | o.data[1]}};
  }
  constexpr BinaryVec operator^(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] ^ o.data[0], data[1] ^ o.data[1]}};
  }
  constexpr BinaryVec operator~() const noexcept {
    return BinaryVec{{~data[0], ~data[1]}};
  }

  constexpr BinaryVec& operator&=(const BinaryVec& o) noexcept {
    data[0] &= o.data[0];
    data[1] &= o.data[1];
    return *this;
  }
  constexpr BinaryVec& operator|=(const BinaryVec& o) noexcept {
    data[0] |= o.data[0];
    data[1] |= o.data[1];
    return *this;
  }
  constexpr BinaryVec& operator^=(const BinaryVec& o) noexcept {
    data[0] ^= o.data[0];
    data[1] ^= o.data[1];
    return *this;
  }

  constexpr bool operator==(const BinaryVec& o) const noexcept {
    return data[0] == o.data[0] && data[1] == o.data[1];
  }
  constexpr bool operator!=(const BinaryVec& o) const noexcept {
    return !(*this == o);
  }
};

}  // namespace binarycore
