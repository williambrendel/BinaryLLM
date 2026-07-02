#pragma once

// ============================================================================
// core/math/dense.hpp
//
// Tier-1 dense float32 kernels for the build-time numerics (Frequent
// Directions, Online Dictionary Learning). Contiguous inner loops that
// auto-vectorize (fast-math is enabled on this TU only), std::thread
// row-parallelism, and symmetric-Gram halving — dependency-free, portable.
//
// These are the swappable kernel layer: everything above calls matmul() /
// gram() / symeig_desc(), so a future escalation to NEON intrinsics or a
// linked BLAS is a localized change here, not a rewrite.
//
// FLOAT, BUILD-TIME ONLY. Never on the integer inference path (spec §7.2).
// ============================================================================

#include <cstddef>

namespace core::math {

// C[m×n] = A[m×k] · B[k×n]. All row-major, contiguous. Threaded over rows
// of C; distinct threads write distinct rows (no synchronization needed).
void matmul(const float* A, std::size_t m, std::size_t k, const float* B,
            std::size_t n, float* C);

// G[m×m] = A[m×d] · A[m×d]ᵀ. Symmetric: computes the upper triangle and
// mirrors. Threaded over rows.
void gram(const float* A, std::size_t m, std::size_t d, float* G);

// Symmetric eigendecomposition of G[m×m] via cyclic Jacobi (computed in
// double for stability, results returned as float). On return:
//   evals[0..m)  — eigenvalues in DESCENDING order
//   evecs[m×m]   — row-major; column j is the eigenvector for evals[j]
// Gin is not modified. Intended for small m (a few hundred): the sketch's
// ℓ×ℓ Gram matrix, never the d-dimensional space.
void symeig_desc(const float* Gin, std::size_t m, float* evals, float* evecs);

}  // namespace core::math
