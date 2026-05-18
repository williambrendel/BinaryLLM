#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace binarycore {

// ============================================================================
// BinaryVec<DIMS>
// ----------------------------------------------------------------------------
// Fixed-size binary vector storing DIMS bits in (DIMS / 64) uint64_t chunks.
// DIMS must be a positive multiple of 64.
//
// Operations are integer-only: AND, OR, XOR, popcount-based dot product,
// popcount-based Hamming distance. No floating-point anywhere.
//
// The compiler maps std::popcount to:
//   - x86 with SSE4.2:    POPCNT instruction (1 cycle per 64 bits)
//   - x86 with AVX-512:   VPOPCNTQ across multiple lanes
//   - ARM with NEON:      VCNT instruction
//   - Anywhere else:      software fallback (~10-15 cycles)
//
// std::array with compile-time CHUNKS means small sizes (64, 128, 256, 512)
// unroll completely; the compiler sees fixed loop bounds and emits straight-line
// code.
// ============================================================================
template <std::size_t DIMS>
class BinaryVec {
    static_assert(DIMS > 0, "DIMS must be positive");
    static_assert(DIMS % 64 == 0, "DIMS must be a multiple of 64");

public:
    static constexpr std::size_t Dims   = DIMS;
    static constexpr std::size_t Chunks = DIMS / 64;

    constexpr BinaryVec() noexcept : data_{} {}

    // Construct from explicit chunk values (mainly for tests).
    constexpr explicit BinaryVec(std::array<uint64_t, Chunks> chunks) noexcept
        : data_(chunks) {}

    // Direct access to underlying storage.
    constexpr std::array<uint64_t, Chunks>&       chunks() noexcept { return data_; }
    constexpr const std::array<uint64_t, Chunks>& chunks() const noexcept { return data_; }

    // Read/write a single bit.
    constexpr bool get(std::size_t i) const noexcept {
        return (data_[i / 64] >> (i % 64)) & 1ULL;
    }
    constexpr void set(std::size_t i) noexcept {
        data_[i / 64] |= (1ULL << (i % 64));
    }
    constexpr void clear(std::size_t i) noexcept {
        data_[i / 64] &= ~(1ULL << (i % 64));
    }
    constexpr void assign(std::size_t i, bool value) noexcept {
        if (value) set(i);
        else clear(i);
    }

    // Total number of set bits.
    constexpr std::size_t popcount() const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < Chunks; ++i) {
            total += static_cast<std::size_t>(std::popcount(data_[i]));
        }
        return total;
    }

    // Bitwise operations (return new vector).
    constexpr BinaryVec operator&(const BinaryVec& o) const noexcept {
        BinaryVec r;
        for (std::size_t i = 0; i < Chunks; ++i) r.data_[i] = data_[i] & o.data_[i];
        return r;
    }
    constexpr BinaryVec operator|(const BinaryVec& o) const noexcept {
        BinaryVec r;
        for (std::size_t i = 0; i < Chunks; ++i) r.data_[i] = data_[i] | o.data_[i];
        return r;
    }
    constexpr BinaryVec operator^(const BinaryVec& o) const noexcept {
        BinaryVec r;
        for (std::size_t i = 0; i < Chunks; ++i) r.data_[i] = data_[i] ^ o.data_[i];
        return r;
    }
    constexpr BinaryVec operator~() const noexcept {
        BinaryVec r;
        for (std::size_t i = 0; i < Chunks; ++i) r.data_[i] = ~data_[i];
        return r;
    }

    // In-place bitwise operations.
    constexpr BinaryVec& operator&=(const BinaryVec& o) noexcept {
        for (std::size_t i = 0; i < Chunks; ++i) data_[i] &= o.data_[i];
        return *this;
    }
    constexpr BinaryVec& operator|=(const BinaryVec& o) noexcept {
        for (std::size_t i = 0; i < Chunks; ++i) data_[i] |= o.data_[i];
        return *this;
    }
    constexpr BinaryVec& operator^=(const BinaryVec& o) noexcept {
        for (std::size_t i = 0; i < Chunks; ++i) data_[i] ^= o.data_[i];
        return *this;
    }

    // Equality.
    constexpr bool operator==(const BinaryVec& o) const noexcept {
        for (std::size_t i = 0; i < Chunks; ++i) {
            if (data_[i] != o.data_[i]) return false;
        }
        return true;
    }
    constexpr bool operator!=(const BinaryVec& o) const noexcept { return !(*this == o); }

    // ------------------------------------------------------------------------
    // Similarity primitives.
    // ------------------------------------------------------------------------

    // Hamming distance: count of bit positions where the two vectors differ.
    // Range: [0, DIMS].
    constexpr std::size_t hamming(const BinaryVec& o) const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < Chunks; ++i) {
            total += static_cast<std::size_t>(std::popcount(data_[i] ^ o.data_[i]));
        }
        return total;
    }

    // Dot product: count of bit positions where both vectors have 1.
    // Equivalent to popcount(*this & o) but avoids constructing the intermediate.
    // Range: [0, DIMS].
    constexpr std::size_t dot(const BinaryVec& o) const noexcept {
        std::size_t total = 0;
        for (std::size_t i = 0; i < Chunks; ++i) {
            total += static_cast<std::size_t>(std::popcount(data_[i] & o.data_[i]));
        }
        return total;
    }

    // Similarity normalized to [0, 1]: 1 - hamming/DIMS.
    // Returns float here (only place we touch floats in this module); useful
    // for human-readable scoring. Inference paths should use raw hamming() or
    // dot() to stay integer.
    float similarity(const BinaryVec& o) const noexcept {
        return 1.0f - static_cast<float>(hamming(o)) / static_cast<float>(DIMS);
    }

private:
    std::array<uint64_t, Chunks> data_;
};

// Convenience aliases.
using BinaryVec64  = BinaryVec<64>;
using BinaryVec128 = BinaryVec<128>;
using BinaryVec256 = BinaryVec<256>;
using BinaryVec512 = BinaryVec<512>;

} // namespace binarycore
