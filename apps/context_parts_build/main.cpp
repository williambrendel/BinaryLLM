// ============================================================================
// apps/context_parts_build/main.cpp
//
// Pass-1 context-part discovery CLI (spec §15). Builds windowed 2F context
// signatures from a dictionary + corpora, recomputes surprisal on the sample
// set, runs parts2f Variant A or B, writes a CPRT dictionary artifact, and
// prints a stats report (admission signal per round, support/width/t_p hists).
//
// Usage:
//   context_parts_build <dict.bin> <out.cprt> <corpus1> [corpus2 ...]
//                       [--variant A|B] [--radius R] [--s-min N]
//                       [--k-max K] [--width D]
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
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using binarycore::binary_vec::BinaryVecDynamic;
using core::parts::StreamToken;

bool is_clause_delim(const std::string& v) {
  return v.find(',') != std::string::npos || v.find(';') != std::string::npos ||
         v.find(':') != std::string::npos;
}

void usage(const char* p) {
  std::cerr << "usage: " << p
            << " <dict.bin> <out.cprt> <corpus1> [corpus2 ...]"
               " [--variant A|B] [--radius R] [--s-min N] [--k-max K] [--width D]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { usage(argv[0]); return 2; }
  const std::string dict_path = argv[1];
  const std::string out_path = argv[2];
  std::vector<std::string> corpora;
  core::parts2f::Config cfg;  // variant='B', s_min=20, K_max=8192, D=256 defaults
  std::size_t radius = 4;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
    if (a == "--variant") { const char* v = next(); if (v) cfg.variant = v[0]; }
    else if (a == "--radius") { const char* v = next(); if (v) radius = std::stoul(v); }
    else if (a == "--s-min") { const char* v = next(); if (v) cfg.s_min = std::stoul(v); }
    else if (a == "--k-max") { const char* v = next(); if (v) cfg.K_max = std::stoul(v); }
    else if (a == "--width") { const char* v = next(); if (v) cfg.D = std::stoul(v); }
    else corpora.push_back(a);
  }
  if (corpora.empty() || (cfg.variant != 'A' && cfg.variant != 'B')) { usage(argv[0]); return 2; }

  std::ifstream din(dict_path, std::ios::binary);
  if (!din) { std::cerr << "cannot open dict\n"; return 1; }
  auto dict = core::parts::load_dict_binary(din);
  const std::uint32_t F = static_cast<std::uint32_t>(dict.size());
  constexpr std::size_t CS = binarycore::binary_vec::BigSparseBinaryVecDynamic::chunk_size;

  core::parts2f::Dataset ds;
  ds.dim = 3 * F;
  auto emit = [&](const BinaryVecDynamic& s) {
    std::vector<std::uint32_t> c;
    for (std::size_t k = 0; k < s.chunks.size(); ++k)
      for (std::uint16_t lo : s.chunks[k].data) {
        std::uint32_t g = static_cast<std::uint32_t>(k * CS + lo);
        if (!(g >= F && g < 2 * F)) c.push_back(g);  // 2F = [L|R], drop identity C
      }
    if (c.empty()) return;
    std::sort(c.begin(), c.end());
    ds.sigs.push_back(std::move(c));
  };
  for (const auto& path : corpora) {
    std::ifstream in(path);
    if (!in) { std::cerr << "cannot open corpus " << path << "\n"; return 1; }
    std::ostringstream ss; ss << in.rdbuf();
    auto tokens = core::parts::tokenize_stream(ss.str());
    auto add = [&](std::size_t a, std::size_t b) {
      for (const auto& s : core::signatures::encode_windowed(dict, tokens, radius, a, b)) emit(s);
    };
    for (const auto& pr : core::signatures::split_paragraphs(tokens)) add(pr.start, pr.end);
    for (const auto& sr : core::signatures::split_sentences(tokens)) add(sr.start, sr.end);
    for (const auto& sr : core::signatures::split_sentences(tokens)) {
      std::size_t cs = sr.start;
      for (std::size_t i = sr.start; i < sr.end; ++i)
        if (tokens[i].type == StreamToken::Type::Delimiter && is_clause_delim(tokens[i].value)) {
          if (i + 1 > cs) add(cs, i + 1);
          cs = i + 1;
        }
      if (cs < sr.end) add(cs, sr.end);
    }
  }
  const std::size_t N = ds.sigs.size();
  if (N == 0) { std::cerr << "no signatures\n"; return 1; }
  ds.build_index();
  ds.w.assign(ds.dim, 0.0);
  std::size_t active = 0;
  for (std::size_t e = 0; e < ds.dim; ++e)
    if (ds.c_e[e] > 0) { ds.w[e] = std::log(double(N) / double(ds.c_e[e])); ++active; }

  std::cout << "signatures=" << N << " active_2F_bits=" << active
            << " radius=" << radius << " variant=" << cfg.variant
            << " s_min=" << cfg.s_min << " K_max=" << cfg.K_max << "\n";

  core::parts2f::BuildStats st;
  auto parts = core::parts2f::build_parts(ds, cfg, &st);

  std::ofstream os(out_path, std::ios::binary);
  if (!os) { std::cerr << "cannot open output\n"; return 1; }
  core::parts2f::save_cprt(os, parts, cfg, 2 * F);

  // stats report
  std::size_t wsum = 0, admitted = 0;
  for (const auto& p : parts) wsum += p.bits.size();
  for (const auto& r : st.rounds) if (r.admitted) ++admitted;
  std::cout << "parts=" << parts.size()
            << " avg_width=" << (parts.empty() ? 0.0 : double(wsum) / parts.size())
            << " rounds=" << st.rounds.size() << " admitted=" << admitted << "\n";
  if (cfg.variant == 'B')
    std::cout << "coverage=" << (st.total_info > 0 ? st.covered_info / st.total_info : 0.0)
              << " (surprisal explained)\n";
  std::cout << "wrote " << out_path << " (CPRT, " << parts.size() << " parts)\n";
  return 0;
}
