#pragma once

// ============================================================================
// binarycore/containers/detail/binary_vec_64.hpp
// ----------------------------------------------------------------------------
// Explicit specialization of BinaryVec<64>.
//
// One chunk. No (i >> 6) needed because there's only one chunk; the bit
// position i is the shift amount within that chunk directly.
//
// Every operation is one machine instruction on any 64-bit CPU: AND/OR/XOR
// are single-cycle, NOT is one cycle. This is the simplest, fastest possible
// implementation.
// ============================================================================

#include "binary_vec_generic.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace binarycore {

template <>
struct BinaryVec<64> {
  static constexpr std::size_t Dims = 64;
  static constexpr std::size_t Chunks = 1;
  std::array<uint64_t, 1> data{};

  // For a single-chunk vector, the bit position i IS the shift amount.
  // No chunk-selection math needed.
  constexpr bool get(std::size_t i) const noexcept {
    return (data[0] >> i) & 1ULL;
  }
  constexpr void set(std::size_t i) noexcept { data[0] |= (1ULL << i); }
  constexpr void clear(std::size_t i) noexcept { data[0] &= ~(1ULL << i); }
  constexpr void assign(std::size_t i, bool v) noexcept {
    if (v)
      set(i);
    else
      clear(i);
  }

  // Each bitwise op is a single uint64_t operation — one instruction.
  constexpr BinaryVec operator&(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] & o.data[0]}};
  }
  constexpr BinaryVec operator|(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] | o.data[0]}};
  }
  constexpr BinaryVec operator^(const BinaryVec& o) const noexcept {
    return BinaryVec{{data[0] ^ o.data[0]}};
  }
  constexpr BinaryVec operator~() const noexcept {
    return BinaryVec{{~data[0]}};
  }

  constexpr BinaryVec& operator&=(const BinaryVec& o) noexcept {
    data[0] &= o.data[0];
    return *this;
  }
  constexpr BinaryVec& operator|=(const BinaryVec& o) noexcept {
    data[0] |= o.data[0];
    return *this;
  }
  constexpr BinaryVec& operator^=(const BinaryVec& o) noexcept {
    data[0] ^= o.data[0];
    return *this;
  }

  constexpr bool operator==(const BinaryVec& o) const noexcept {
    return data[0] == o.data[0];
  }
  constexpr bool operator!=(const BinaryVec& o) const noexcept {
    return data[0] != o.data[0];
  }
};

}  // namespace binarycore
