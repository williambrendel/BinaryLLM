#include "binaryoptim/dominant_set.hpp"
#include "binaryoptim/thresholds.hpp"
#include "doctest.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace binaryoptim;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace {

// Build a symmetric affinity matrix with constant inter-cluster weight.
// Two clusters: indices [0..nA) and [nA..nA+nB).
//   intra: high (1.0), inter: low (0.0)
// Diagonal is set to 0; the solver ignores it.
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

}  // namespace

// -----------------------------------------------------------------------------
// Dominant-set solver basics
// -----------------------------------------------------------------------------

TEST_CASE("dominant_set: trivial 1-element problem") {
  std::vector<double> A = {0.0};
  std::vector<double> b = {0.0};
  std::vector<double> beta = {0.0};
  auto r = dominant_set<double>(A, b, beta, 1);
  CHECK(r.alpha.size() == 1);
  CHECK(r.alpha[0] == doctest::Approx(1.0));
}

TEST_CASE("dominant_set: alpha sums to gamma (default = 1)") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  auto r = dominant_set<double>(A, b, beta, 6);
  CHECK(sum(r.alpha) == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("dominant_set: alpha sums to gamma when gamma != 1") {
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  DominantSetOptions<double> opts;
  opts.gamma = 5.0;
  auto r = dominant_set<double>(A, b, beta, 6, opts);
  CHECK(sum(r.alpha) == doctest::Approx(5.0).epsilon(1e-5));
}

TEST_CASE("dominant_set: identifies the only dense cluster") {
  // 3-cluster: dense [0..3), sparse [3..6) (no internal edges).
  std::vector<double> A(36, 0.0);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      if (i != j) A[i * 6 + j] = 1.0;

  std::vector<double> b(6, 0.0);
  std::vector<double> beta(6, 0.0);
  auto r = dominant_set<double>(A, b, beta, 6);

  double mass_dense = r.alpha[0] + r.alpha[1] + r.alpha[2];
  double mass_sparse = r.alpha[3] + r.alpha[4] + r.alpha[5];
  CHECK(mass_dense > 0.95);
  CHECK(mass_sparse < 0.05);
}

TEST_CASE("dominant_set: converges for well-conditioned problem") {
  auto A = two_cluster_affinity(4, 4);
  std::vector<double> b(8, 0.0);
  std::vector<double> beta(8, 0.0);
  auto r = dominant_set<double>(A, b, beta, 8);
  CHECK(r.converged);
  CHECK(r.iterations < 300);
}

TEST_CASE("dominant_set: linear term b biases the result") {
  // Two equal-size, equal-density clusters; b prefers the second.
  auto A = two_cluster_affinity(3, 3);
  std::vector<double> b = {0, 0, 0, 1, 1, 1};
  std::vector<double> beta(6, 0.0);
  auto r = dominant_set<double>(A, b, beta, 6);

  double mass_first = r.alpha[0] + r.alpha[1] + r.alpha[2];
  double mass_second = r.alpha[3] + r.alpha[4] + r.alpha[5];
  CHECK(mass_second > mass_first);
}

TEST_CASE("dominant_set: per-candidate beta suppresses one node") {
  // 4-clique. Penalize node 0 heavily; expect it to have minimal alpha mass.
  std::vector<double> A = {
    0, 1, 1, 1,
    1, 0, 1, 1,
    1, 1, 0, 1,
    1, 1, 1, 0,
  };
  std::vector<double> b(4, 0.0);
  std::vector<double> beta = {2.0, 0.0, 0.0, 0.0};
  auto r = dominant_set<double>(A, b, beta, 4);
  CHECK(r.alpha[0] < r.alpha[1]);
  CHECK(r.alpha[0] < r.alpha[2]);
  CHECK(r.alpha[0] < r.alpha[3]);
}

// -----------------------------------------------------------------------------
// Threshold utilities
// -----------------------------------------------------------------------------

TEST_CASE("top_k_indices returns the k largest entries, sorted descending") {
  std::vector<double> alpha = {0.1, 0.5, 0.2, 0.05, 0.4};
  auto top = top_k_indices(alpha, 3);
  REQUIRE(top.size() == 3);
  CHECK(top[0] == 1);  // 0.5
  CHECK(top[1] == 4);  // 0.4
  CHECK(top[2] == 2);  // 0.2
}

TEST_CASE("top_k_indices handles k > n gracefully") {
  std::vector<double> alpha = {0.5, 0.5};
  auto top = top_k_indices(alpha, 10);
  CHECK(top.size() == 2);
}

TEST_CASE("log_ratio_gap_threshold finds the elbow") {
  // Three big, two tiny → gap between the 3rd and 4th.
  std::vector<double> alpha = {0.3, 0.3, 0.3, 0.001, 0.001};
  double t = log_ratio_gap_threshold(alpha, std::log(3.0));
  // Threshold should be ~0.001 (the larger of the two tiny ones), and the
  // three big entries should all clear it.
  CHECK(alpha[0] > t);
  CHECK(alpha[3] <= t);
}

TEST_CASE("log_ratio_gap_threshold returns 0 when no significant gap") {
  std::vector<double> alpha = {0.25, 0.25, 0.25, 0.25};
  double t = log_ratio_gap_threshold(alpha, std::log(3.0));
  CHECK(t == 0.0);
}

TEST_CASE("entropy_threshold keeps ~exp(H) entries") {
  // 4 equal entries → H = log(4), exp(H) = 4 → keep all 4.
  std::vector<double> alpha = {0.25, 0.25, 0.25, 0.25};
  double t = entropy_threshold(alpha);
  CHECK(t == 0.0);  // ceil(exp(H)) >= n → return 0 (keep all)

  // Sharp distribution → small effective support
  std::vector<double> sharp = {0.9, 0.05, 0.025, 0.025};
  double t2 = entropy_threshold(sharp);
  CHECK(t2 > 0.0);  // some entries get pruned
  CHECK(sharp[0] > t2);
}
