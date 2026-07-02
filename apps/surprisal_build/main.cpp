// ============================================================================
// apps/surprisal_build/main.cpp
//
// Build the frozen per-typed-3F-bit surprisal weight table (WGT1) from a
// dictionary + corpora (spec §7.2, §7.5.0).
//
// Streams sentence-scoped 3F signatures (one per Word token, context =
// its own sentence — a clean bounded window, no double counting), tallies
// per-position document frequency, and writes w_e = log(N/df_e) quantized
// to integer multiplicities [1, cap].
//
// Usage:
//   surprisal_build <dict> <out.wgt1> <corpus1> [corpus2 ...] [--cap N]
// ============================================================================

#include "dict_io.hpp"
#include "dictionary.hpp"
#include "encoder.hpp"
#include "sentence_split.hpp"
#include "tokenize.hpp"
#include "weight_table.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using core::parts::PartDictionary;
using core::parts::StreamToken;
using core::parts::tokenize_stream;
using core::signatures::encode;
using core::signatures::split_sentences;
using core::signatures::SurprisalCounter;

namespace {

void usage(const char* p) {
  std::cerr << "usage: " << p
            << " <dict> <out.wgt1> <corpus1> [corpus2 ...] [--cap N]\n";
}

bool ends_with(std::string_view s, std::string_view suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

PartDictionary load_dict(const std::string& path) {
  if (ends_with(path, ".bin")) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) { std::cerr << "cannot open dict\n"; std::exit(1); }
    return core::parts::load_dict_binary(in);
  }
  std::ifstream in(path);
  if (!in.is_open()) { std::cerr << "cannot open dict\n"; std::exit(1); }
  return core::parts::load_dict_text(in);
}

std::string slurp(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { usage(argv[0]); return 2; }

  const std::string dict_path = argv[1];
  const std::string out_path = argv[2];
  std::vector<std::string> corpora;
  std::uint8_t cap = core::signatures::kDefaultQuantCap;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--cap" && i + 1 < argc) {
      cap = static_cast<std::uint8_t>(std::stoul(argv[++i]));
    } else if (arg.rfind("--cap=", 0) == 0) {
      cap = static_cast<std::uint8_t>(std::stoul(arg.substr(6)));
    } else {
      corpora.push_back(arg);
    }
  }
  if (corpora.empty()) { usage(argv[0]); return 2; }

  const auto dict = load_dict(dict_path);
  const std::size_t F = dict.size();
  const std::size_t dim = 3 * F;
  std::cerr << "dict F = " << F << "  (3F = " << dim << "), cap = "
            << static_cast<int>(cap) << "\n";

  SurprisalCounter counter(dim);
  for (const auto& path : corpora) {
    const auto tokens = tokenize_stream(slurp(path));
    std::size_t sigs = 0;
    for (const auto& sr : split_sentences(tokens)) {
      for (const auto& s : encode(dict, tokens, sr.start, sr.end)) {
        counter.add(s);
        ++sigs;
      }
    }
    std::cerr << "  " << path << ": " << sigs << " signatures\n";
  }
  std::cerr << "total signatures: " << counter.token_count() << "\n";

  const auto table = counter.finalize(cap);

  std::ofstream out(out_path, std::ios::binary);
  if (!out.is_open()) { std::cerr << "cannot open " << out_path << "\n"; return 1; }
  core::signatures::save_surprisal(table, out);

  // Coverage report: how many positions ever fired.
  std::size_t nonzero = 0;
  for (std::uint16_t w : table.weights) nonzero += (w != 0);
  std::cerr << "wrote " << out_path << ": dim=" << table.dim << ", "
            << nonzero << " positions active ("
            << (dim ? 100.0 * static_cast<double>(nonzero) / static_cast<double>(dim) : 0.0)
            << "%)\n";
  return 0;
}
