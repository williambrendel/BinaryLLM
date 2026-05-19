#pragma once

// ============================================================================
// binarycore/containers/detail/binary_vec_256.hpp
// ----------------------------------------------------------------------------
// Explicit specialization of BinaryVec<256>.
//
// Four chunks. Fits in one AVX2 256-bit YMM register on supporting CPUs.
// On ARM NEON it spans two 128-bit registers. Either way: very fast.
// ============================================================================

#include "binary_vec_generic.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace binarycore {

template <>
struct BinaryVec<256> {
  static constexpr std::size_t Dims = 256;
  static constexpr std::size_t Chunks = 4;
  std::array<uint64_t, 4> data{};

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
    return BinaryVec{{data[0] & o.data[0], data[1] & o.data[1],
                      data[2] & o.data[2], data[3] & o.data[3]}};
  }
  constexpr BinaryVec operator|(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] | o.data[0], data[1] | o.data[1],
                      data[2] | o.data[2], data[3] | o.data[3]}};
  }
  constexpr BinaryVec operator^(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] ^ o.data[0], data[1] ^ o.data[1],
                      data[2] ^ o.data[2], data[3] ^ o.data[3]}};
  }
  constexpr BinaryVec operator~() const noexcept {
    return BinaryVec{{~data[0], ~data[1], ~data[2], ~data[3]}};
  }

  constexpr BinaryVec& operator&=(const BinaryVec& o) noexcept {
    data[0] &= o.data[0];
    data[1] &= o.data[1];
    data[2] &= o.data[2];
    data[3] &= o.data[3];
    return *this;
  }
  constexpr BinaryVec& operator|=(const BinaryVec& o) noexcept {
    data[0] |= o.data[0];
    data[1] |= o.data[1];
    data[2] |= o.data[2];
    data[3] |= o.data[3];
    return *this;
  }
  constexpr BinaryVec& operator^=(const BinaryVec& o) noexcept {
    data[0] ^= o.data[0];
    data[1] ^= o.data[1];
    data[2] ^= o.data[2];
    data[3] ^= o.data[3];
    return *this;
  }

  constexpr bool operator==(const BinaryVec& o) const noexcept {
    return data[0] == o.data[0] && data[1] == o.data[1] &&
           data[2] == o.data[2] && data[3] == o.data[3];
  }
  constexpr bool operator!=(const BinaryVec& o) const noexcept {
    return !(*this == o);
  }
};

}  // namespace binarycore
