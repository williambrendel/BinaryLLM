#pragma once

// ============================================================================
// binaryoptim/dominant_set.hpp
//
// Pavan-Pelillo dominant-set solver via replicator dynamics, generalized to
// the gamma-scaled simplex.
//
// Objective (maximized):
//   alpha^T A alpha + alpha^T b - alpha^T diag(beta) alpha
//
// Constraint:
//   alpha_i >= 0,   sum_i alpha_i = gamma
//
// Inputs:
//   A      symmetric N x N matrix, row-major contiguous. Diagonal of A is
//          ignored (diagonal penalty comes from beta).
//   b      length N, linear term.
//   beta   length N, diagonal regularization (>= 0). Higher beta[i]
//          suppresses candidate i.
//   gamma  total simplex mass. gamma = 1 reproduces standard Pavan-Pelillo.
//
// The solver is a pure primitive: it does NOT do pre-filtering of inputs or
// post-thresholding of alpha. Compose with helpers in thresholds.hpp.
//
// --------------------------------------------------------------------------
// API tiers
// --------------------------------------------------------------------------
// Two public overloads:
//
//   1. span-based convenience form
//        Solver allocates the output and the workspace internally.
//        Returns a DominantSetResult that owns the alpha vector.
//        Use when you don't care about allocation.
//
//   2. pointer-based control form
//        Caller owns the output (alpha_inout, non-null required).
//        Workspace buffers are optional — pass non-null to avoid internal
//        allocation, leave null to let the solver allocate.
//        Returns a DominantSetStats (iterations, converged) — the alpha
//        lives in the caller's buffer.
//
// --------------------------------------------------------------------------
// Output initialization (auto-detected)
// --------------------------------------------------------------------------
// In the pointer form, the contents of alpha_inout at call time are
// inspected:
//   - All zeros (or sum <= 0)        -> uniform init alpha_i = gamma / N
//   - Any positive entries           -> negatives clamped to 0, then
//                                       rescaled to sum = gamma
// If init_epsilon > 0, random jitter in [0, init_epsilon) is added per
// entry before the final rescaling. This is anti-trap insurance: the
// replicator update keeps any alpha_i = 0 at 0 forever, so a tiny epsilon
// keeps every component alive at the start of the run.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace binaryoptim {

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
template <typename T = double>
struct DominantSetOptions {
  std::size_t   max_iter     = 300;     // hard iteration cap
  T             tol          = T(1e-5); // L-inf tolerance on alpha for early exit
  T             gamma        = T(1);    // total mass on the simplex
  T             init_epsilon = T(0);    // 0 = fully deterministic init
  std::uint64_t seed         = 0;       // used only when init_epsilon > 0
};

// ---------------------------------------------------------------------------
// Workspace
//
// Optional caller-owned buffers used by the inner loop. Pass non-null to
// avoid internal allocation; pass null (the default) to let the solver
// allocate per call.
//
// Sizes (when non-null):
//   M       must point to at least N*N elements
//   M_alpha must point to at least N elements
// ---------------------------------------------------------------------------
template <typename T = double>
struct DominantSetWorkspace {
  T* M = nullptr;
  T* M_alpha = nullptr;
};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------
struct DominantSetStats {
  std::size_t iterations = 0;
  bool        converged  = false;
};

template <typename T = double>
struct DominantSetResult {
  std::vector<T>   alpha;       // length N, sums to gamma
  std::size_t      iterations = 0;
  bool             converged  = false;
};

// ---------------------------------------------------------------------------
// Pointer-based control form.
//
// alpha_inout (REQUIRED, non-null): caller-owned, length at least N.
//   On entry: see "Output initialization (auto-detected)" above.
//   On exit:  contains the solved alpha (sum == gamma).
// ws: optional workspace buffers; solver allocates any null entries.
// ---------------------------------------------------------------------------
template <typename T = double>
DominantSetStats dominant_set(
    const T* A, const T* b, const T* beta,
    std::size_t N,
    T* alpha_inout,
    DominantSetWorkspace<T> ws = {},
    const DominantSetOptions<T>& opts = {});

// ---------------------------------------------------------------------------
// Span-based convenience form. Allocates output and workspace internally.
// ---------------------------------------------------------------------------
template <typename T = double>
DominantSetResult<T> dominant_set(
    std::span<const T> A,
    std::span<const T> b,
    std::span<const T> beta,
    std::size_t N,
    const DominantSetOptions<T>& opts = {});

}  // namespace binaryoptim
