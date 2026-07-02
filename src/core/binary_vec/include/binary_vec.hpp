#pragma once

// ============================================================================
// binarycore::binary_vec::binary_vec.hpp
//
// Public entry point for the BinaryVec family. Pulls in all four
// tier implementations and exposes two aliases:
//
//   BinaryVec<Dim>       — compile-time dim. Dispatches to the right
//                          tier via std::conditional_t:
//
//                            Dim ≤ 4096          → DenseBinaryVec<Dim>      (tier 1)
//                            4097 ≤ Dim ≤ 65535  → SparseBinaryVec<Dim>     (tier 2)
//                            Dim ≥ 65536          → BigSparseBinaryVec<Dim>  (tier 3)
//
//                          Use whenever the dim is a model constant
//                          or otherwise known at compile time.
//
//   BinaryVecDynamic     — runtime dim. Currently aliases the only
//                          runtime variant we provide:
//                          BigSparseBinaryVecDynamic (tier 3 family,
//                          but works for any dim — 1-chunk case
//                          covers ≤ 65535 too).
//
//                          Use when the dim is determined at runtime
//                          — typically the input-signature dim
//                          computed once the dictionary is loaded.
//
// Each underlying type exposes jaccard(a, b). Sparse / BigSparse /
// BigSparseBinaryVecDynamic also expose intersection_size as a
// reusable helper. ADL resolves the right overload regardless of
// which alias the caller used.
//
// Callers should ONLY include this header — the four implementation
// headers in this directory are internal includes.
// ============================================================================

#include "big_sparse_binary_vec.hpp"
#include "big_sparse_binary_vec_dynamic.hpp"
#include "dense_binary_vec.hpp"
#include "sparse_binary_vec.hpp"

#include <cstddef>
#include <type_traits>

namespace binarycore::binary_vec {

template <std::size_t Dim>
using BinaryVec = std::conditional_t<
    (Dim <= 4096),  DenseBinaryVec<Dim>,
    std::conditional_t<(Dim <= 65535), SparseBinaryVec<Dim>,
                                       BigSparseBinaryVec<Dim>>>;

using BinaryVecDynamic = BigSparseBinaryVecDynamic;

}  // namespace binarycore::binary_vec
