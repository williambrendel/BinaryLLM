// ============================================================================
// core/pass1/src/pass1.cpp
// ============================================================================

#include "pass1.hpp"

#include "info_jaccard.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <utility>

namespace core::pass1 {

namespace {

constexpr std::size_t kChunk =
    binarycore::binary_vec::BigSparseBinaryVecDynamic::chunk_size;

// The D bits of `sig` with the highest surprisal weight (ties by position),
// returned sorted ascending.
core::codebook::Atom top_d_by_weight(const Signature& sig,
                                     const std::vector<std::uint32_t>& w,
                                     std::uint32_t D) {
  core::codebook::Atom out = sig;
  if (out.size() > D) {
    std::partial_sort(out.begin(), out.begin() + D, out.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        if (w[a] != w[b]) return w[a] > w[b];
                        return a < b;
                      });
    out.resize(D);
    std::sort(out.begin(), out.end());
  }
  return out;
}

}  // namespace

Signature to_positions(
    const binarycore::binary_vec::BigSparseBinaryVecDynamic& v) {
  Signature out;
  for (std::size_t k = 0; k < v.chunks.size(); ++k) {
    const std::size_t base = k * kChunk;
    for (std::uint16_t local : v.chunks[k].data)
      out.push_back(static_cast<std::uint32_t>(base + local));
  }
  return out;  // globally sorted: chunks in order, locals sorted within
}

Pass1Learner::Pass1Learner(std::size_t dim, std::vector<std::uint32_t> weights,
                           Config cfg)
    : dim_(dim),
      weights_(std::move(weights)),
      cfg_(cfg),
      cb_(dim, cfg.D),
      acc_(cfg.K),
      nk_(cfg.K, 0.0),
      fired_ever_(cfg.K, 0) {
  for (std::size_t k = 0; k < cfg_.K; ++k) cb_.add({});  // K empty slots
}

void Pass1Learner::seed(const core::sketch::FrequentDirections& fd,
                        const std::vector<Signature>& warmup) {
  const std::size_t rank = fd.rank();
  const std::vector<float>& factor = fd.factor();

  // Ridge γ = mean eigenvalue (heuristic — stabilizes low-σ directions).
  std::vector<double> s2(rank);
  double gamma = 0.0;
  for (std::size_t r = 0; r < rank; ++r) {
    const double s = fd.singular_value(r);
    s2[r] = s * s;
    gamma += s2[r];
  }
  gamma = (rank > 0) ? gamma / static_cast<double>(rank) : 1.0;

  // Ridge-leverage of each warmup signature off the FD factor:
  //   Σ_r ⟨x, v_r⟩² / (σ_r² + γ),  ⟨x, v_r⟩ = (factor_r · x) / σ_r.
  std::mt19937_64 rng(cfg_.rng_seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::vector<std::pair<double, std::size_t>> keys;  // (reservoir key, index)
  keys.reserve(warmup.size());
  for (std::size_t i = 0; i < warmup.size(); ++i) {
    double lev = 0.0;
    for (std::size_t r = 0; r < rank; ++r) {
      const float* row = factor.data() + r * dim_;
      double p = 0.0;
      for (std::uint32_t e : warmup[i]) p += row[e];
      const double denom = s2[r] * (s2[r] + gamma);
      if (denom > 0.0) lev += (p * p) / denom;
    }
    if (lev <= 0.0) continue;
    // Efraimidis–Spirakis weighted reservoir key: log(u)/lev, take the largest.
    keys.emplace_back(std::log(unif(rng)) / lev, i);
  }

  const std::size_t take = std::min(cfg_.K, keys.size());
  std::partial_sort(
      keys.begin(), keys.begin() + take, keys.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });
  for (std::size_t t = 0; t < take; ++t)
    cb_.set(t, top_d_by_weight(warmup[keys[t].second], weights_, cfg_.D));
}

void Pass1Learner::observe(const Signature& sig) {
  const auto fired = core::codebook::pursuit_encode(cb_, sig, weights_, cfg_.pursuit);
  for (std::uint32_t k : fired) {
    nk_[k] += 1.0;
    fired_ever_[k] = 1;
    auto& b = acc_[k];
    for (std::uint32_t e : sig) b[e] += static_cast<double>(weights_[e]);
  }
  ++seen_;
  if (cfg_.refresh_every != 0 && seen_ % cfg_.refresh_every == 0) refresh();
}

void Pass1Learner::refresh_atom_(std::size_t k,
                                 const std::vector<std::uint32_t>& bit_degree) {
  const auto& b = acc_[k];
  if (b.empty()) return;

  // congestion(e) = other atoms already holding bit e (snapshot minus self).
  const core::codebook::Atom& cur = cb_.atom(k);
  const std::unordered_set<std::uint32_t> own(cur.begin(), cur.end());

  // score(e) = b_k[e] − λ·congestion(e)·w_e ; keep the top-D positive scores.
  std::vector<std::pair<std::uint32_t, double>> items;  // (pos, score)
  items.reserve(b.size());
  for (const auto& [e, mass] : b) {
    std::uint32_t cong = (e < bit_degree.size()) ? bit_degree[e] : 0;
    if (own.count(e)) cong -= 1;  // don't count k against itself
    const double score =
        mass - cfg_.lambda * static_cast<double>(cong) *
                   static_cast<double>(weights_[e]);
    if (score > 0.0) items.emplace_back(e, score);
  }
  if (items.empty()) {
    cb_.set(k, {});  // fully congested → emptied (dead; re-seed handles later)
    return;
  }
  const std::size_t take = std::min<std::size_t>(items.size(), cfg_.D);
  std::partial_sort(items.begin(), items.begin() + take, items.end(),
                    [](const auto& a, const auto& b2) {
                      if (a.second != b2.second) return a.second > b2.second;
                      return a.first < b2.first;
                    });
  core::codebook::Atom atom;
  atom.reserve(take);
  for (std::size_t i = 0; i < take; ++i) atom.push_back(items[i].first);
  std::sort(atom.begin(), atom.end());
  cb_.set(k, std::move(atom));
}

void Pass1Learner::apply_refresh_(bool do_decay) {
  // Snapshot per-bit atom degree BEFORE updating any atom, so the incoherence
  // penalty is consistent across the round (order-independent).
  std::vector<std::uint32_t> bit_degree(dim_, 0);
  for (std::size_t k = 0; k < cb_.size(); ++k)
    for (std::uint32_t e : cb_.atom(k))
      if (e < dim_) ++bit_degree[e];

  for (std::size_t k = 0; k < cfg_.K; ++k)
    if (nk_[k] > 0.0) refresh_atom_(k, bit_degree);

  if (!do_decay) return;
  // Exponential-window forgetting; prune fully-decayed entries to stay bounded.
  for (std::size_t k = 0; k < cfg_.K; ++k) {
    nk_[k] *= cfg_.decay;
    auto& b = acc_[k];
    for (auto it = b.begin(); it != b.end();) {
      it->second *= cfg_.decay;
      if (it->second < 1e-6) it = b.erase(it);
      else ++it;
    }
  }
}

void Pass1Learner::refresh() { apply_refresh_(true); }

void Pass1Learner::finalize() { apply_refresh_(false); }

double Pass1Learner::utilization() const {
  std::size_t used = 0;
  for (char f : fired_ever_) used += (f != 0);
  return cfg_.K ? static_cast<double>(used) / static_cast<double>(cfg_.K) : 0.0;
}

double Pass1Learner::mean_recall(const std::vector<Signature>& sigs) const {
  const std::uint32_t* w = weights_.data();
  double sum = 0.0;
  std::size_t n = 0;
  for (const auto& sig : sigs) {
    if (sig.empty()) continue;
    const auto fired = core::codebook::pursuit_encode(cb_, sig, weights_, cfg_.pursuit);
    const auto recon = cb_.decode(fired);
    const double ic = static_cast<double>(
        binarycore::sparse::info_content<std::uint32_t>(sig.data(), sig.size(), w));
    if (ic <= 0.0) continue;
    const double wd = static_cast<double>(
        binarycore::sparse::weighted_dot<std::uint32_t>(
            recon.data(), recon.size(), sig.data(), sig.size(), w));
    sum += wd / ic;
    ++n;
  }
  return n ? sum / static_cast<double>(n) : 0.0;
}

}  // namespace core::pass1
