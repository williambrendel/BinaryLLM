#include "binaryoptim/dominant_set.hpp"
#include "binaryoptim/thresholds.hpp"
#include "doctest.h"

#include <cmath>
#include <numeric>
#include <span>
#include <vector>

using namespace binaryoptim;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace {

std::vector<double> two_cluster_affinity(std::size_t nA, std::size_t nB,
                                         double intra = 1.0, double inter = 0.0) {
  const std::size_t N = nA + nB;
  std::vector<double> A(N * N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      if (i == j) continue;
      const bool same = (i < nA) == (j < nA);
      A[i * N + j] = same ? intra : inter;
    }
  }
  return A;
}

double sum(const std::vector<double>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}

// Wrap raw vectors as spans for the convenience form.
template <typename T>
std::span<const T> as_span(const std::vector<T>& v) {
  return std::span<const T>(v.data(), v.size());
}

}  // namespace

// =============================================================================
// Span-based convenience form
// =============================================================================

TEST_CASE("span form: trivial 1-element problem") {
  std::vector<double> A = {0.0};
  std::vector<double> b = {0.0};
  std::vector<double> beta = {0.0};
  auto r = dominant_set<double>(as_span(A), as_span(b), as_span(beta), 1);
  REQUIRE(r.alpha.size() == 1);
  CHECK(r.alpha[0] == doctest::Approx(1.0));
}

TEST_CASE("span form: alpha sums to gamma (default = 1)") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  auto r = dominant_set<double>(as_span(A), as_span(b), as_span(beta), 6);
  CHECK(sum(r.alpha) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("span form: alpha sums to gamma when gamma != 1") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  DominantSetOptions<double> opts;
  opts.gamma = 5.0;
  auto r = dominant_set<double>(as_span(A), as_span(b), as_span(beta), 6, opts);
  CHECK(sum(r.alpha) == doctest::Approx(5.0).epsilon(1e-5));
}

TEST_CASE("span form: identifies the dense cluster") {
  std::vector<double> A(36, 0.0);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      if (i != j) A[i * 6 + j] = 1.0;
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  auto r = dominant_set<double>(as_span(A), as_span(b), as_span(beta), 6);

  const double dense  = r.alpha[0] + r.alpha[1] + r.alpha[2];
  const double sparse = r.alpha[3] + r.alpha[4] + r.alpha[5];
  CHECK(dense > 0.95);
  CHECK(sparse < 0.05);
}

TEST_CASE("span form: converges for well-conditioned problem") {
  auto A = two_cluster_affinity(4, 4);
  std::vector<double> b(8, 0.0);
  std::vector<double> beta(8, 0.0);
  auto r = dominant_set<double>(as_span(A), as_span(b), as_span(beta), 8);
  CHECK(r.converged);
  CHECK(r.iterations < 300);
}

TEST_CASE("span form: linear term b biases the result") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b    = {0, 0, 0, 1, 1, 1};
  std::vector<double> beta(6, 0.0);
  auto r = dominant_set<double>(as_span(A), as_span(b), as_span(beta), 6);

  const double first  = r.alpha[0] + r.alpha[1] + r.alpha[2];
  const double second = r.alpha[3] + r.alpha[4] + r.alpha[5];
  CHECK(second > first);
}

TEST_CASE("span form: per-candidate beta suppresses one node") {
  std::vector<double> A = {
    0, 1, 1, 1,
    1, 0, 1, 1,
    1, 1, 0, 1,
    1, 1, 1, 0,
  };
  std::vector<double> b(4, 0.0);
  std::vector<double> beta = {2.0, 0.0, 0.0, 0.0};
  auto r = dominant_set<double>(as_span(A), as_span(b), as_span(beta), 4);
  CHECK(r.alpha[0] < r.alpha[1]);
  CHECK(r.alpha[0] < r.alpha[2]);
  CHECK(r.alpha[0] < r.alpha[3]);
}

// =============================================================================
// Pointer-based control form
// =============================================================================

TEST_CASE("pointer form: caller-owned alpha buffer (all-zero -> uniform init)") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  std::vector<double> alpha(6, 0.0);

  auto stats = dominant_set<double>(
      A.data(), b.data(), beta.data(), 6,
      alpha.data());

  CHECK(sum(alpha) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(stats.converged);
}

TEST_CASE("pointer form: warm-start with provided alpha (rescaled to gamma)") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);

  // Pre-seed with a non-normalized distribution biased toward the second cluster.
  std::vector<double> alpha = {0.1, 0.1, 0.1, 0.5, 0.5, 0.5};

  auto stats = dominant_set<double>(
      A.data(), b.data(), beta.data(), 6,
      alpha.data());

  CHECK(sum(alpha) == doctest::Approx(1.0).epsilon(1e-6));
  // Second cluster should dominate (it had the head start and same density).
  const double first  = alpha[0] + alpha[1] + alpha[2];
  const double second = alpha[3] + alpha[4] + alpha[5];
  CHECK(second > first);
}

TEST_CASE("pointer form: provided alpha with negatives is clamped") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  std::vector<double> alpha = {-1.0, 0.5, 0.5, 0.5, 0.5, 0.5};

  dominant_set<double>(A.data(), b.data(), beta.data(), 6, alpha.data());

  CHECK(alpha[0] >= 0.0);
  CHECK(sum(alpha) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("pointer form: workspace eliminates internal allocation") {
  const std::size_t N = 6;
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(N, 0.0);
  std::vector<double> beta(N, 0.0);
  std::vector<double> alpha(N, 0.0);

  // Caller-owned workspace buffers, sized for N.
  std::vector<double> M_buf(N * N);
  std::vector<double> Malpha_buf(N);
  DominantSetWorkspace<double> ws{M_buf.data(), Malpha_buf.data()};

  auto stats = dominant_set<double>(
      A.data(), b.data(), beta.data(), N,
      alpha.data(), ws);

  CHECK(sum(alpha) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(stats.converged);
}

TEST_CASE("pointer form: partial workspace (M_alpha only) — solver allocates M") {
  const std::size_t N = 6;
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(N, 0.0);
  std::vector<double> beta(N, 0.0);
  std::vector<double> alpha(N, 0.0);

  std::vector<double> Malpha_buf(N);
  DominantSetWorkspace<double> ws{nullptr, Malpha_buf.data()};

  auto stats = dominant_set<double>(
      A.data(), b.data(), beta.data(), N,
      alpha.data(), ws);

  CHECK(sum(alpha) == doctest::Approx(1.0).epsilon(1e-6));
  CHECK(stats.converged);
}

TEST_CASE("pointer form: init_epsilon adds jitter and breaks zero-traps") {
  // Seed with exactly one nonzero component — without epsilon, the
  // replicator would lock all other components at 0.
  const std::size_t N = 4;
  std::vector<double> A(N * N, 1.0);
  for (std::size_t i = 0; i < N; ++i) A[i * N + i] = 0.0;
  std::vector<double> b(N, 0.0);
  std::vector<double> beta(N, 0.0);
  std::vector<double> alpha = {1.0, 0.0, 0.0, 0.0};

  DominantSetOptions<double> opts;
  opts.init_epsilon = 1e-3;
  opts.seed = 42;

  dominant_set<double>(A.data(), b.data(), beta.data(), N,
                       alpha.data(), {}, opts);

  // With epsilon, all components should be alive (>0).
  for (std::size_t i = 0; i < N; ++i) {
    CHECK(alpha[i] > 0.0);
  }
  CHECK(sum(alpha) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("pointer form: deterministic by default (no init_epsilon)") {
  const std::size_t N = 6;
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(N, 0.0);
  std::vector<double> beta(N, 0.0);
  std::vector<double> alpha1(N, 0.0);
  std::vector<double> alpha2(N, 0.0);

  dominant_set<double>(A.data(), b.data(), beta.data(), N, alpha1.data());
  dominant_set<double>(A.data(), b.data(), beta.data(), N, alpha2.data());

  for (std::size_t i = 0; i < N; ++i) {
    CHECK(alpha1[i] == doctest::Approx(alpha2[i]));
  }
}

// =============================================================================
// Threshold utilities
// =============================================================================

TEST_CASE("top_k_indices returns the k largest entries, sorted descending") {
  std::vector<double> alpha = {0.1, 0.5, 0.2, 0.05, 0.4};
  auto top = top_k_indices(alpha, 3);
  REQUIRE(top.size() == 3);
  CHECK(top[0] == 1);
  CHECK(top[1] == 4);
  CHECK(top[2] == 2);
}

TEST_CASE("top_k_indices handles k > n gracefully") {
  std::vector<double> alpha = {0.5, 0.5};
  auto top = top_k_indices(alpha, 10);
  CHECK(top.size() == 2);
}

TEST_CASE("log_ratio_gap_threshold finds the elbow") {
  std::vector<double> alpha = {0.3, 0.3, 0.3, 0.001, 0.001};
  double t = log_ratio_gap_threshold(alpha, std::log(3.0));
  CHECK(alpha[0] > t);
  CHECK(alpha[3] <= t);
}

TEST_CASE("log_ratio_gap_threshold returns 0 when no significant gap") {
  std::vector<double> alpha = {0.25, 0.25, 0.25, 0.25};
  double t = log_ratio_gap_threshold(alpha, std::log(3.0));
  CHECK(t == 0.0);
}

TEST_CASE("entropy_threshold keeps ~exp(H) entries") {
  std::vector<double> alpha = {0.25, 0.25, 0.25, 0.25};
  double t = entropy_threshold(alpha);
  CHECK(t == 0.0);

  std::vector<double> sharp = {0.9, 0.05, 0.025, 0.025};
  double t2 = entropy_threshold(sharp);
  CHECK(t2 > 0.0);
  CHECK(sharp[0] > t2);
}
