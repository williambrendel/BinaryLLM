// ============================================================================
// core/math/src/dense.cpp
// ============================================================================

#include "dense.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace core::math {

namespace {

// Run fn(lo, hi) over a partition of [0, n) across up to 8 threads. Serial
// for small n (thread setup would dominate). Callers must ensure the ranges
// touch disjoint output so no locking is needed.
template <class Fn>
void parallel_for(std::size_t n, Fn&& fn) {
  unsigned hw = std::thread::hardware_concurrency();
  unsigned nt = (hw == 0) ? 1u : std::min(hw, 8u);
  if (n < 64 || nt <= 1) {
    fn(std::size_t{0}, n);
    return;
  }
  if (static_cast<std::size_t>(nt) > n) nt = static_cast<unsigned>(n);
  const std::size_t block = (n + nt - 1) / nt;
  std::vector<std::thread> pool;
  pool.reserve(nt);
  for (unsigned t = 0; t < nt; ++t) {
    const std::size_t lo = static_cast<std::size_t>(t) * block;
    const std::size_t hi = std::min(n, lo + block);
    if (lo >= hi) break;
    pool.emplace_back([&fn, lo, hi] { fn(lo, hi); });
  }
  for (auto& th : pool) th.join();
}

// Like parallel_for but assigns rows round-robin (i = t, t+nt, …). Use when
// per-row work is uneven (e.g. the triangular gram) so heavy and light rows
// spread evenly across threads instead of piling onto thread 0.
template <class Fn>
void parallel_for_strided(std::size_t n, Fn&& fn) {
  unsigned hw = std::thread::hardware_concurrency();
  unsigned nt = (hw == 0) ? 1u : std::min(hw, 8u);
  if (n < 64 || nt <= 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  if (static_cast<std::size_t>(nt) > n) nt = static_cast<unsigned>(n);
  std::vector<std::thread> pool;
  pool.reserve(nt);
  for (unsigned t = 0; t < nt; ++t) {
    pool.emplace_back([&fn, t, nt, n] {
      for (std::size_t i = t; i < n; i += nt) fn(i);
    });
  }
  for (auto& th : pool) th.join();
}

double sign_of(double a, double b) { return b >= 0.0 ? std::fabs(a) : -std::fabs(a); }

// Householder reduction of a symmetric n×n matrix z (row-major, in place) to
// tridiagonal form. On return z holds the accumulated orthogonal transform,
// d the diagonal, e the sub-diagonal (e[0] = 0). Standard tred2.
void tred2(std::vector<double>& z, int n, std::vector<double>& d,
           std::vector<double>& e) {
  for (int i = n - 1; i >= 1; --i) {
    const int l = i - 1;
    double h = 0.0, scale = 0.0;
    if (l > 0) {
      for (int k = 0; k <= l; ++k) scale += std::fabs(z[i * n + k]);
      if (scale == 0.0) {
        e[i] = z[i * n + l];
      } else {
        for (int k = 0; k <= l; ++k) {
          z[i * n + k] /= scale;
          h += z[i * n + k] * z[i * n + k];
        }
        double f = z[i * n + l];
        double g = (f >= 0.0) ? -std::sqrt(h) : std::sqrt(h);
        e[i] = scale * g;
        h -= f * g;
        z[i * n + l] = f - g;
        f = 0.0;
        for (int j = 0; j <= l; ++j) {
          z[j * n + i] = z[i * n + j] / h;
          g = 0.0;
          for (int k = 0; k <= j; ++k) g += z[j * n + k] * z[i * n + k];
          for (int k = j + 1; k <= l; ++k) g += z[k * n + j] * z[i * n + k];
          e[j] = g / h;
          f += e[j] * z[i * n + j];
        }
        const double hh = f / (h + h);
        for (int j = 0; j <= l; ++j) {
          f = z[i * n + j];
          e[j] = g = e[j] - hh * f;
          for (int k = 0; k <= j; ++k)
            z[j * n + k] -= (f * e[k] + g * z[i * n + k]);
        }
      }
    } else {
      e[i] = z[i * n + l];
    }
    d[i] = h;
  }
  d[0] = 0.0;
  e[0] = 0.0;
  for (int i = 0; i < n; ++i) {
    const int l = i - 1;
    if (d[i] != 0.0) {
      for (int j = 0; j <= l; ++j) {
        double g = 0.0;
        for (int k = 0; k <= l; ++k) g += z[i * n + k] * z[k * n + j];
        for (int k = 0; k <= l; ++k) z[k * n + j] -= g * z[k * n + i];
      }
    }
    d[i] = z[i * n + i];
    z[i * n + i] = 1.0;
    for (int j = 0; j <= l; ++j) {
      z[j * n + i] = 0.0;
      z[i * n + j] = 0.0;
    }
  }
}

// Implicit-shift QL on the tridiagonal (d, e), accumulating eigenvectors into
// Q. Q is stored TRANSPOSED — row j holds eigenvector-column j's components —
// so each QL rotation updates two contiguous rows (vectorizes) instead of
// striding down columns by n (the dominant cost of the naive layout).
//
// Uses an explicit relative deflation tolerance instead of the classic
// `fabs(e[m]) + dd == dd` rounding trick (which -ffast-math may fold to `==0`).
void tqli(std::vector<double>& d, std::vector<double>& e, int n,
          std::vector<double>& Q) {
  for (int i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0.0;
  for (int l = 0; l < n; ++l) {
    int iter = 0, m;
    do {
      for (m = l; m < n - 1; ++m) {
        const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
        if (std::fabs(e[m]) <= 1e-300 + 1e-15 * dd) break;
      }
      if (m != l) {
        if (iter++ == 50) break;  // fail-safe; convergence is expected far sooner
        double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        double r = std::hypot(g, 1.0);
        g = d[m] - d[l] + e[l] / (g + sign_of(r, g));
        double s = 1.0, c = 1.0, p = 0.0;
        int i;
        for (i = m - 1; i >= l; --i) {
          double f = s * e[i];
          const double b = c * e[i];
          r = std::hypot(f, g);
          e[i + 1] = r;
          if (r == 0.0) {
            d[i + 1] -= p;
            e[m] = 0.0;
            break;
          }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          double* qi = Q.data() + static_cast<std::size_t>(i) * n;
          double* qi1 = qi + n;  // row i+1
          for (int k = 0; k < n; ++k) {
            f = qi1[k];
            qi1[k] = s * qi[k] + c * f;
            qi[k] = c * qi[k] - s * f;
          }
        }
        if (r == 0.0 && i >= l) continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m != l);
  }
}

}  // namespace

// Column-block width. Sized so the reused operand block (≈ rows × kBlock
// floats) stays resident in L2 across the inner reuse, which is what turns
// these kernels from memory-bandwidth-bound into compute-bound at d ≈ 100k.
namespace {
constexpr std::size_t kBlock = 2048;
}  // namespace

void matmul(const float* A, std::size_t m, std::size_t k, const float* B,
            std::size_t n, float* C) {
  std::fill(C, C + m * n, 0.0f);
  // Block over output columns (outer, serial) so the B column-block is loaded
  // once and reused by every output row; parallelize the rows within a block.
  // ikj order keeps the inner loop over j contiguous (vectorizes).
  for (std::size_t j0 = 0; j0 < n; j0 += kBlock) {
    const std::size_t j1 = std::min(n, j0 + kBlock);
    parallel_for(m, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        const float* arow = A + i * k;
        float* crow = C + i * n;
        for (std::size_t l = 0; l < k; ++l) {
          const float a = arow[l];
          const float* brow = B + l * n;
          for (std::size_t j = j0; j < j1; ++j) crow[j] += a * brow[j];
        }
      }
    });
  }
}

void gram(const float* A, std::size_t m, std::size_t d, float* G) {
  std::fill(G, G + m * m, 0.0f);
  // Block over d (outer) so the A column-block for all rows stays cached and
  // is reused across every (i,j) pair; parallelize rows within a block. Each
  // G[i][j] (j≥i) is accumulated across blocks by the single thread owning i.
  for (std::size_t l0 = 0; l0 < d; l0 += kBlock) {
    const std::size_t l1 = std::min(d, l0 + kBlock);
    parallel_for_strided(m, [&](std::size_t i) {
      const float* ai = A + i * d;
      for (std::size_t j = i; j < m; ++j) {
        const float* aj = A + j * d;
        float s = 0.0f;
        for (std::size_t l = l0; l < l1; ++l) s += ai[l] * aj[l];
        G[i * m + j] += s;
      }
    });
  }
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < i; ++j) G[i * m + j] = G[j * m + i];
}

void symeig_desc(const float* Gin, std::size_t m, float* evals, float* evecs) {
  if (m == 0) return;
  const int n = static_cast<int>(m);

  std::vector<double> z(m * m), d(m), e(m);
  for (std::size_t i = 0; i < m * m; ++i) z[i] = static_cast<double>(Gin[i]);

  tred2(z, n, d, e);

  // Transpose the transform into Q (row j = column j of z) so tqli's rotations
  // are contiguous. Q[col*n + row] = z[row*n + col].
  std::vector<double> Q(m * m);
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) Q[b * n + a] = z[a * n + b];

  tqli(d, e, n, Q);  // Q rows are the eigenvectors after this

  // Order eigenpairs by descending eigenvalue.
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d[a] > d[b]; });
  for (int j = 0; j < n; ++j) {
    evals[j] = static_cast<float>(d[idx[j]]);
    const double* q = Q.data() + static_cast<std::size_t>(idx[j]) * n;
    for (int r = 0; r < n; ++r) evecs[r * n + j] = static_cast<float>(q[r]);
  }
}

}  // namespace core::math
