#pragma once

// ============================================================================
// binarycore/containers/binary_vec.hpp
// ----------------------------------------------------------------------------
// PUBLIC umbrella header for BinaryVec.
//
// Include THIS file to use BinaryVec<N> at any size. It pulls in the generic
// template (`detail/binary_vec_generic.hpp`) plus all four explicit
// specializations for the canonical sizes 64, 128, 256, 512.
//
// The convenience type aliases at the bottom let you write `BinaryVec64`
// instead of `BinaryVec<64>` if you prefer.
//
// Internal structure:
//   detail/
//     binary_vec_generic.hpp   ← fallback for arbitrary multiples of 64
//     binary_vec_64.hpp        ← specialization with unrolled ops
//     binary_vec_128.hpp
//     binary_vec_256.hpp
//     binary_vec_512.hpp
//
// The `detail/` folder convention means "implementation files, do not
// include these directly — include the umbrella header instead." Boost,
// the STL, and Eigen all use this pattern.
// ============================================================================

// IMPORTANT: include the generic template FIRST. The specializations need
// the primary template to already be declared, so they can correctly mark
// themselves as `template <>` (full specialization).
#include "detail/binary_vec_generic.hpp"

// Then the specializations override the primary template for these sizes.
// Order among the specializations doesn't matter — they're independent.
#include "detail/binary_vec_128.hpp"
#include "detail/binary_vec_256.hpp"
#include "detail/binary_vec_512.hpp"
#include "detail/binary_vec_64.hpp"

namespace binarycore {

// ============================================================================
// Convenience type aliases
// ----------------------------------------------------------------------------
// `using X = Y;` is the modern C++ syntax for "X is another name for Y"
// (replaces the older `typedef` syntax). Lets users write BinaryVec64
// instead of BinaryVec<64>.
// ============================================================================
using BinaryVec64 = BinaryVec<64>;
using BinaryVec128 = BinaryVec<128>;
using BinaryVec256 = BinaryVec<256>;
using BinaryVec512 = BinaryVec<512>;
using BinaryVec1024 = BinaryVec<1024>; // generic template
using BinaryVec2048 = BinaryVec<2048>; // generic template
using BinaryVec4096 = BinaryVec<4096>; // generic template

} // namespace binarycore
