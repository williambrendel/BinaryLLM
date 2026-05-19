#pragma once

// ============================================================================
// binarycore/containers/detail/binary_vec_generic.hpp
// ----------------------------------------------------------------------------
// The GENERIC template for BinaryVec<DIMS>. Works for any positive multiple
// of 64. For the four canonical sizes (64, 128, 256, 512) there are explicit
// specializations in sibling files that override this with unrolled code.
//
// Users should NOT include this directly — include
//   binarycore/containers/binary_vec.hpp
// instead, which pulls in this and all the specializations.
//
// Index conventions:
//   chunk index of bit i      = i >> 6   (i / 64)
//   bit position within chunk = i & 63   (i % 64)
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>

namespace binarycore {

template <std::size_t DIMS>
struct BinaryVec {
  // Compile-time validation. static_assert fires at compile time, so misuse
  // is caught before the program runs.
  static_assert(DIMS > 0, "DIMS must be positive");
  static_assert(DIMS % 64 == 0, "DIMS must be a multiple of 64");

  // Public compile-time constants. Accessible as BinaryVec<128>::Dims, etc.
  static constexpr std::size_t Dims = DIMS;
  static constexpr std::size_t Chunks = DIMS / 64;

  // Storage. std::array is stack-allocated, contiguous, fixed-size — no heap,
  // no pointer chasing. The trailing `{}` value-initializes all chunks to
  // zero, so default-constructed BinaryVec is the zero vector.
  std::array<uint64_t, Chunks> data{};

  // -------- Bit access --------
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

  // -------- Bitwise operations (return new vector) --------
  constexpr BinaryVec operator&(const BinaryVec& o) const noexcept {
    BinaryVec r;
    for (std::size_t i = 0; i < Chunks; ++i) r.data[i] = data[i] & o.data[i];
    return r;
  }
  constexpr BinaryVec operator|(const BinaryVec& o) const noexcept {
    BinaryVec r;
    for (std::size_t i = 0; i < Chunks; ++i) r.data[i] = data[i] | o.data[i];
    return r;
  }
  constexpr BinaryVec operator^(const BinaryVec& o) const noexcept {
    BinaryVec r;
    for (std::size_t i = 0; i < Chunks; ++i) r.data[i] = data[i] ^ o.data[i];
    return r;
  }
  constexpr BinaryVec operator~() const noexcept {
    BinaryVec r;
    for (std::size_t i = 0; i < Chunks; ++i) r.data[i] = ~data[i];
    return r;
  }

  // -------- In-place bitwise operations --------
  constexpr BinaryVec& operator&=(const BinaryVec& o) noexcept {
    for (std::size_t i = 0; i < Chunks; ++i) data[i] &= o.data[i];
    return *this;
  }
  constexpr BinaryVec& operator|=(const BinaryVec& o) noexcept {
    for (std::size_t i = 0; i < Chunks; ++i) data[i] |= o.data[i];
    return *this;
  }
  constexpr BinaryVec& operator^=(const BinaryVec& o) noexcept {
    for (std::size_t i = 0; i < Chunks; ++i) data[i] ^= o.data[i];
    return *this;
  }

  // -------- Equality --------
  constexpr bool operator==(const BinaryVec& o) const noexcept {
    for (std::size_t i = 0; i < Chunks; ++i) {
      if (data[i] != o.data[i]) return false;
    }
    return true;
  }
  constexpr bool operator!=(const BinaryVec& o) const noexcept {
    return !(*this == o);
  }
};

}  // namespace binarycore
