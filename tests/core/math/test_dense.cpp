// ============================================================================
// tests/core/math/test_dense.cpp
//
// Tier-1 dense kernels vs. naive references: matmul, symmetric gram, and the
// Jacobi eigensolver (reconstruction, orthonormality, descending order).
// ============================================================================

#include "dense.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "doctest.h"

namespace {

std::vector<float> naive_matmul(const std::vector<float>& A, std::size_t m,
                                std::size_t k, const std::vector<float>& B,
                                std::size_t n) {
  std::vector<float> C(m * n, 0.0f);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      float s = 0.0f;
      for (std::size_t l = 0; l < k; ++l) s += A[i * k + l] * B[l * n + j];
      C[i * n + j] = s;
    }
  return C;
}

}  // namespace

TEST_CASE("matmul matches naive reference") {
  // Sizes above the parallel_for threshold to exercise threading.
  const std::size_t m = 70, k = 33, n = 40;
  std::vector<float> A(m * k), B(k * n);
  for (std::size_t i = 0; i < A.size(); ++i)
    A[i] = static_cast<float>((i * 7 + 3) % 11) - 5.0f;
  for (std::size_t i = 0; i < B.size(); ++i)
    B[i] = static_cast<float>((i * 5 + 1) % 9) - 4.0f;

  std::vector<float> C(m * n);
  core::math::matmul(A.data(), m, k, B.data(), n, C.data());
  const std::vector<float> ref = naive_matmul(A, m, k, B, n);

  for (std::size_t i = 0; i < C.size(); ++i)
    CHECK(C[i] == doctest::Approx(ref[i]));
}

TEST_CASE("gram is symmetric and equals A·Aᵀ") {
  const std::size_t m = 66, d = 25;
  std::vector<float> A(m * d);
  for (std::size_t i = 0; i < A.size(); ++i)
    A[i] = static_cast<float>((i * 3 + 2) % 7) - 3.0f;

  std::vector<float> G(m * m);
  core::math::gram(A.data(), m, d, G.data());

  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < m; ++j) {
      float s = 0.0f;
      for (std::size_t l = 0; l < d; ++l) s += A[i * d + l] * A[j * d + l];
      CHECK(G[i * m + j] == doctest::Approx(s));
      CHECK(G[i * m + j] == doctest::Approx(G[j * m + i]));  // symmetric
    }
}

TEST_CASE("symeig_desc: diagonal matrix → eigenvalues are the diagonal") {
  const std::size_t m = 3;
  std::vector<float> G = {5, 0, 0, 0, 1, 0, 0, 0, 3};
  std::vector<float> ev(m), V(m * m);
  core::math::symeig_desc(G.data(), m, ev.data(), V.data());
  CHECK(ev[0] == doctest::Approx(5.0f));
  CHECK(ev[1] == doctest::Approx(3.0f));
  CHECK(ev[2] == doctest::Approx(1.0f));  // descending
}

TEST_CASE("symeig_desc: reconstruction, orthonormality, eigen-equation") {
  const std::size_t m = 5;
  // Build a symmetric matrix.
  std::vector<float> G(m * m);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = i; j < m; ++j) {
      const float v = static_cast<float>((i * 2 + j * 3 + 1) % 5);
      G[i * m + j] = v;
      G[j * m + i] = v;
    }

  std::vector<float> ev(m), V(m * m);
  core::math::symeig_desc(G.data(), m, ev.data(), V.data());

  // Descending order.
  for (std::size_t i = 1; i < m; ++i) CHECK(ev[i] <= ev[i - 1] + 1e-4f);

  // Columns orthonormal: VᵀV = I.
  for (std::size_t a = 0; a < m; ++a)
    for (std::size_t b = 0; b < m; ++b) {
      double dot = 0.0;
      for (std::size_t r = 0; r < m; ++r) dot += V[r * m + a] * V[r * m + b];
      CHECK(dot == doctest::Approx(a == b ? 1.0 : 0.0).epsilon(1e-4));
    }

  // Reconstruction: V·diag(ev)·Vᵀ == G.
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < m; ++j) {
      double s = 0.0;
      for (std::size_t r = 0; r < m; ++r) s += V[i * m + r] * ev[r] * V[j * m + r];
      CHECK(s == doctest::Approx(G[i * m + j]).epsilon(1e-4));
    }
}
