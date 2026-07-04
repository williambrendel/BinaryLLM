// ============================================================================
// apps/context_parts_eval/main.cpp
//
// Held-out generalization eval for Pass-1 context-parts — the honest yardstick.
// Build the part dictionary (Variant B) on a TRAIN corpus, then reconstruct
// UNSEEN test-corpus contexts and compare against a single-bit baseline of the
// same train bits. Reports surprisal-weighted coverage at matched activation
// budget B (= number of fired units allowed per context, greedily chosen), so
// multi-bit parts and single bits are compared fairly per fired unit.
//
// The intended win is the sparse regime: at small B, a multi-bit part explains
// more of a novel context than the same count of atoms.
//
// Usage:
//   context_parts_eval <dict.bin> <train_corpus> <test_corpus>
//                      [--radius R] [--k-max K] [--s-min N]
// ============================================================================

#include "parts2f.hpp"

#include "dict_io.hpp"
#include "dictionary.hpp"
#include "encoder.hpp"
#include "paragraph_split.hpp"
#include "sentence_split.hpp"
#include "tokenize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Sig = std::vector<std::uint32_t>;
using binarycore::binary_vec::BinaryVecDynamic;
using core::parts::StreamToken;

bool clause_delim(const std::string& v) {
  return v.find(',') != std::string::npos || v.find(';') != std::string::npos ||
         v.find(':') != std::string::npos;
}

// Windowed multi-scale 2F signatures for a corpus (matches context_parts_build).
std::vector<Sig> build_sigs(const core::parts::PartDictionary& dict, const std::string& path,
                            std::size_t radius, std::uint32_t F) {
  constexpr std::size_t CS = binarycore::binary_vec::BigSparseBinaryVecDynamic::chunk_size;
  std::ifstream in(path);
  std::ostringstream ss; ss << in.rdbuf();
  auto tokens = core::parts::tokenize_stream(ss.str());
  std::vector<Sig> out;
  auto emit = [&](const BinaryVecDynamic& s) {
    Sig c;
    for (std::size_t k = 0; k < s.chunks.size(); ++k)
      for (std::uint16_t lo : s.chunks[k].data) {
        std::uint32_t g = static_cast<std::uint32_t>(k * CS + lo);
        if (!(g >= F && g < 2 * F)) c.push_back(g);
      }
    if (c.empty()) return;
    std::sort(c.begin(), c.end());
    out.push_back(std::move(c));
  };
  auto add = [&](std::size_t a, std::size_t b) {
    for (const auto& s : core::signatures::encode_windowed(dict, tokens, radius, a, b)) emit(s);
  };
  for (const auto& pr : core::signatures::split_paragraphs(tokens)) add(pr.start, pr.end);
  for (const auto& sr : core::signatures::split_sentences(tokens)) add(sr.start, sr.end);
  for (const auto& sr : core::signatures::split_sentences(tokens)) {
    std::size_t cs = sr.start;
    for (std::size_t i = sr.start; i < sr.end; ++i)
      if (tokens[i].type == StreamToken::Type::Delimiter && clause_delim(tokens[i].value)) {
        if (i + 1 > cs) add(cs, i + 1);
        cs = i + 1;
      }
    if (cs < sr.end) add(cs, sr.end);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0]
              << " <dict.bin> <train_corpus> <test_corpus> [--radius R] [--k-max K] [--s-min N]\n";
    return 2;
  }
  const std::string dp = argv[1], train = argv[2], test = argv[3];
  std::size_t radius = 4;
  core::parts2f::Config cfg; cfg.variant = 'B'; cfg.s_min = 20; cfg.K_max = 8192;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
    if (a == "--radius") { const char* v = next(); if (v) radius = std::stoul(v); }
    else if (a == "--k-max") { const char* v = next(); if (v) cfg.K_max = std::stoul(v); }
    else if (a == "--s-min") { const char* v = next(); if (v) cfg.s_min = std::stoul(v); }
  }

  std::ifstream din(dp, std::ios::binary);
  if (!din) { std::cerr << "cannot open dict\n"; return 1; }
  auto dict = core::parts::load_dict_binary(din);
  const std::uint32_t F = static_cast<std::uint32_t>(dict.size());

  core::parts2f::Dataset ds;
  ds.dim = 3 * F;
  ds.sigs = build_sigs(dict, train, radius, F);
  const std::size_t Ntr = ds.sigs.size();
  if (Ntr == 0) { std::cerr << "no train signatures\n"; return 1; }
  ds.build_index();
  ds.w.assign(ds.dim, 0.0);
  const double wmax = std::log(static_cast<double>(Ntr));
  std::size_t active = 0;
  for (std::size_t e = 0; e < ds.dim; ++e)
    if (ds.c_e[e] > 0) { ds.w[e] = std::log(double(Ntr) / double(ds.c_e[e])); ++active; }
  std::cout << "train sigs=" << Ntr << " active_2F_bits=" << active << " test=" << test << "\n";

  auto parts = core::parts2f::build_parts(ds, cfg);
  std::vector<double> pinfo(parts.size());
  std::vector<std::vector<std::uint32_t>> cwINV(ds.dim);
  double wsum = 0.0;
  for (std::uint32_t j = 0; j < parts.size(); ++j) {
    pinfo[j] = ds.info(parts[j].bits);
    wsum += parts[j].bits.size();
    for (std::uint32_t b : parts[j].bits) cwINV[b].push_back(j);
  }
  // single-bit baseline: top-K train bits by info-contribution w_e·c_e
  std::vector<std::pair<double, std::uint32_t>> mb;
  for (std::size_t e = 0; e < ds.dim; ++e)
    if (ds.c_e[e] > 0) mb.emplace_back(ds.w[e] * double(ds.c_e[e]), e);
  const std::size_t Ksb = std::min<std::size_t>(cfg.K_max, mb.size());
  std::partial_sort(mb.begin(), mb.begin() + Ksb, mb.end(),
                    [](auto& a, auto& b) { return a.first > b.first; });
  std::vector<char> inSB(ds.dim, 0);
  for (std::size_t i = 0; i < Ksb; ++i) inSB[mb[i].second] = 1;
  std::cout << "parts=" << parts.size() << " (avg_w=" << (parts.empty() ? 0.0 : wsum / parts.size())
            << ")  single-bit dict=" << Ksb << "\n\n";

  auto winfo = [&](std::uint32_t b) { return ds.w[b] > 0.0 ? ds.w[b] : wmax; };
  auto test_sigs = build_sigs(dict, test, radius, F);
  const std::size_t Nte = test_sigs.size();

  const std::vector<std::size_t> Bs = {1, 2, 3, 5, 8, 999};  // 999 = unlimited
  std::vector<double> covP(Bs.size(), 0.0), covS(Bs.size(), 0.0);
  double firedP = 0.0, presS = 0.0;
  std::vector<char> seen(parts.size(), 0);
  for (const Sig& x : test_sigs) {
    double xinfo = 0.0;
    for (std::uint32_t b : x) xinfo += winfo(b);
    if (xinfo <= 0.0) continue;
    std::vector<std::uint32_t> cand;
    for (std::uint32_t b : x)
      for (std::uint32_t j : cwINV[b])
        if (!seen[j]) { seen[j] = 1; cand.push_back(j); }
    std::vector<std::uint32_t> fired;
    for (std::uint32_t j : cand) {
      seen[j] = 0;
      if (ds.cont(x, parts[j].bits, pinfo[j]) >= parts[j].t_p) fired.push_back(j);
    }
    firedP += fired.size();
    std::vector<std::pair<double, std::uint32_t>> pres;
    for (std::uint32_t b : x) if (inSB[b]) pres.emplace_back(ds.w[b], b);
    std::sort(pres.begin(), pres.end(), [](auto& a, auto& b) { return a.first > b.first; });
    presS += pres.size();

    std::vector<char> covx, used;
    for (std::size_t bi = 0; bi < Bs.size(); ++bi) {
      const std::size_t B = Bs[bi];
      covx.assign(x.size(), 0);
      used.assign(fired.size(), 0);
      double ip = 0.0;
      for (std::size_t step = 0; step < B && step < fired.size(); ++step) {
        std::size_t bestj = fired.size();
        double bestg = 0.0;
        std::vector<std::size_t> besthit;
        for (std::size_t t = 0; t < fired.size(); ++t) {
          if (used[t]) continue;
          double g = 0.0;
          std::vector<std::size_t> hit;
          const auto& pb = parts[fired[t]].bits;
          std::size_t i = 0, j = 0;
          while (i < x.size() && j < pb.size()) {
            if (x[i] < pb[j]) ++i;
            else if (pb[j] < x[i]) ++j;
            else { if (!covx[i]) { g += winfo(x[i]); hit.push_back(i); } ++i; ++j; }
          }
          if (g > bestg) { bestg = g; bestj = t; besthit.swap(hit); }
        }
        if (bestj == fired.size()) break;
        used[bestj] = 1; ip += bestg;
        for (std::size_t h : besthit) covx[h] = 1;
      }
      covP[bi] += ip / xinfo;
      double is = 0.0;
      for (std::size_t t = 0; t < pres.size() && t < B; ++t) is += ds.w[pres[t].second];
      covS[bi] += is / xinfo;
    }
  }
  std::cout << "held-out contexts=" << Nte << "  avg fired/context: parts="
            << firedP / Nte << " single-bit=" << presS / Nte << "\n\n";
  std::cout << "  budget B   parts_cov   singlebit_cov   winner\n";
  for (std::size_t bi = 0; bi < Bs.size(); ++bi) {
    const double cp = covP[bi] / Nte, cs = covS[bi] / Nte;
    std::printf("  %-8s   %.4f      %.4f        %s\n",
                Bs[bi] == 999 ? "unlim" : std::to_string(Bs[bi]).c_str(), cp, cs,
                cp > cs ? "PARTS" : "single");
  }
  return 0;
}
