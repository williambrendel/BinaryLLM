#pragma once

// ============================================================================
// binarycore/containers/detail/binary_vec_512.hpp
// ----------------------------------------------------------------------------
// Explicit specialization of BinaryVec<512>.
//
// Eight chunks. Fits in one AVX-512 ZMM register on supporting CPUs. With
// VPOPCNTQ (AVX-512BW), a 512-dim popcount becomes a single SIMD instruction.
// On ARM NEON it spans four 128-bit registers.
//
// For the value-returning ops we keep the full 8-way unroll because the
// result has to be constructed in-place. For the in-place ops we use a
// compile-time-bounded loop, which the compiler unrolls automatically.
// ============================================================================

#include "binary_vec_generic.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace binarycore {

template <>
struct BinaryVec<512> {
  static constexpr std::size_t Dims = 512;
  static constexpr std::size_t Chunks = 8;
  std::array<uint64_t, 8> data{};

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
                      data[2] & o.data[2], data[3] & o.data[3],
                      data[4] & o.data[4], data[5] & o.data[5],
                      data[6] & o.data[6], data[7] & o.data[7]}};
  }
  constexpr BinaryVec operator|(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] | o.data[0], data[1] | o.data[1],
                      data[2] | o.data[2], data[3] | o.data[3],
                      data[4] | o.data[4], data[5] | o.data[5],
                      data[6] | o.data[6], data[7] | o.data[7]}};
  }
  constexpr BinaryVec operator^(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] ^ o.data[0], data[1] ^ o.data[1],
                      data[2] ^ o.data[2], data[3] ^ o.data[3],
                      data[4] ^ o.data[4], data[5] ^ o.data[5],
                      data[6] ^ o.data[6], data[7] ^ o.data[7]}};
  }
  constexpr BinaryVec operator~() const noexcept {
    return BinaryVec{{~data[0], ~data[1], ~data[2], ~data[3], ~data[4],
                      ~data[5], ~data[6], ~data[7]}};
  }

  constexpr BinaryVec& operator&=(const BinaryVec& o) noexcept {
    for (std::size_t i = 0; i < 8; ++i) data[i] &= o.data[i];
    return *this;
  }
  constexpr BinaryVec& operator|=(const BinaryVec& o) noexcept {
    for (std::size_t i = 0; i < 8; ++i) data[i] |= o.data[i];
    return *this;
  }
  constexpr BinaryVec& operator^=(const BinaryVec& o) noexcept {
    for (std::size_t i = 0; i < 8; ++i) data[i] ^= o.data[i];
    return *this;
  }

  constexpr bool operator==(const BinaryVec& o) const noexcept {
    for (std::size_t i = 0; i < 8; ++i) {
      if (data[i] != o.data[i]) return false;
    }
    return true;
  }
  constexpr bool operator!=(const BinaryVec& o) const noexcept {
    return !(*this == o);
  }
};

}  // namespace binarycore
