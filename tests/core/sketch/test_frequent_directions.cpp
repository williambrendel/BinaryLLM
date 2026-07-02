// ============================================================================
// tests/core/sketch/test_frequent_directions.cpp
//
// Frequent Directions contracts (spec §4.4):
//   - lossless covariance recovery when data rank ≤ ℓ (even through shrinks)
//   - deterministic FD error bound 0 ≤ AᵀA − BᵀB ⪯ (‖A‖_F²/ℓ)·I when rank > ℓ
//   - energy is never created; singular values non-increasing
// ============================================================================

#include "frequent_directions.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "doctest.h"

using binarycore::binary_vec::BinaryVecDynamic;
using core::sketch::FrequentDirections;

namespace {

BinaryVecDynamic sig(std::size_t dim, const std::vector<std::uint32_t>& bits) {
  BinaryVecDynamic v(dim);
  constexpr std::size_t cs = BinaryVecDynamic::chunk_size;
  for (std::uint32_t g : bits)
    v.chunks[g / cs].data.push_back(static_cast<std::uint16_t>(g % cs));
  for (auto& c : v.chunks) std::sort(c.data.begin(), c.data.end());
  return v;
}

}  // namespace

TEST_CASE("FD: lossless covariance when data rank <= ell (through shrinks)") {
  const std::size_t d = 24, ell = 8;
  // Three disjoint patterns → rank 3 ≤ ell.
  const std::vector<std::vector<std::uint32_t>> pats = {
      {0, 1, 2, 3}, {8, 9, 10, 11}, {16, 17, 18}};
  const int reps = 10;  // 30 rows > buffer 2ell=16 → forces shrink

  FrequentDirections fd(d, ell);
  std::vector<std::vector<std::uint32_t>> fed;
  for (int r = 0; r < reps; ++r)
    for (const auto& p : pats) {
      fd.add(sig(d, p));
      fed.push_back(p);
    }
  fd.finalize();

  // Exact AᵀA[i][j] = number of fed rows with both i and j set.
  auto exact = [&](std::uint32_t i, std::uint32_t j) {
    int c = 0;
    for (const auto& row : fed) {
      const bool hi = std::find(row.begin(), row.end(), i) != row.end();
      const bool hj = std::find(row.begin(), row.end(), j) != row.end();
      if (hi && hj) ++c;
    }
    return static_cast<double>(c);
  };

  // Rank ≤ ell → FD is lossless: BᵀB == AᵀA.
  CHECK(fd.bit_correlation(0, 1) == doctest::Approx(exact(0, 1)).epsilon(1e-3));
  CHECK(fd.bit_correlation(0, 0) == doctest::Approx(exact(0, 0)).epsilon(1e-3));
  CHECK(fd.bit_correlation(0, 8) == doctest::Approx(exact(0, 8)).epsilon(1e-3));  // 0
  CHECK(fd.bit_correlation(16, 18) ==
        doctest::Approx(exact(16, 18)).epsilon(1e-3));
  CHECK(fd.rank() == 3);
}

TEST_CASE("FD: deterministic error bound when data rank > ell") {
  const std::size_t d = 40, ell = 3;
  // Eight distinct single-bit-cluster patterns → rank 8 > ell.
  std::vector<std::vector<std::uint32_t>> pats;
  for (std::uint32_t k = 0; k < 8; ++k)
    pats.push_back({5 * k, 5 * k + 1, 5 * k + 2});

  FrequentDirections fd(d, ell);
  std::vector<std::vector<std::uint32_t>> fed;
  for (int r = 0; r < 6; ++r)
    for (const auto& p : pats) {
      fd.add(sig(d, p));
      fed.push_back(p);
    }
  fd.finalize();

  // ‖A‖_F² = total set bits (binary rows).
  double frob2 = 0.0;
  for (const auto& row : fed) frob2 += static_cast<double>(row.size());
  const double bound = frob2 / static_cast<double>(ell);

  for (std::uint32_t i = 0; i < 40; ++i) {
    // exact df_i = rows with bit i set.
    double df = 0.0;
    for (const auto& row : fed)
      if (std::find(row.begin(), row.end(), i) != row.end()) df += 1.0;
    const double est = fd.bit_correlation(i, i);
    CHECK(est <= df + 1e-3);            // FD underestimates (BᵀB ⪯ AᵀA)
    CHECK(df - est <= bound + 1e-3);    // within the deterministic bound
  }
}

TEST_CASE("FD: singular values non-increasing; energy not created") {
  const std::size_t d = 30, ell = 4;
  std::vector<std::vector<std::uint32_t>> pats;
  for (std::uint32_t k = 0; k < 6; ++k) pats.push_back({4 * k, 4 * k + 1});

  FrequentDirections fd(d, ell);
  double frob2 = 0.0;
  for (int r = 0; r < 5; ++r)
    for (const auto& p : pats) {
      fd.add(sig(d, p));
      frob2 += static_cast<double>(p.size());
    }
  fd.finalize();

  for (std::size_t i = 1; i < fd.rank(); ++i)
    CHECK(fd.singular_value(i) <= fd.singular_value(i - 1) + 1e-4f);

  double energy = 0.0;
  for (std::size_t i = 0; i < fd.rank(); ++i)
    energy += static_cast<double>(fd.singular_value(i)) * fd.singular_value(i);
  CHECK(energy <= frob2 + 1e-3);  // Σσ² = ‖B‖_F² ≤ ‖A‖_F²
}
