// ============================================================================
// apps/pass1_train/main.cpp
//
// End-to-end pass-1 training (spec §4): dict + WGT1 + corpora → sentence-scoped
// 3F signatures → Frequent Directions factor → optional hub-strip → ridge-
// leverage seed → stream/refresh → freeze → codebook (CBK1).
//
// Reports utilization, mean surprisal-weighted reconstruction recall, and
// timing. A modest --K keeps the smoke quick.
//
// Usage:
//   pass1_train <dict> <wgt1> <out.cbk1> <corpus...>
//               [--K n] [--D n] [--ell n] [--strip f] [--lambda x]
// ============================================================================

#include "codebook.hpp"
#include "recon_metrics.hpp"
#include "pursuit.hpp"
#include "dict_io.hpp"
#include "dictionary.hpp"
#include "encoder.hpp"
#include "frequent_directions.hpp"
#include "pass1.hpp"
#include "sentence_split.hpp"
#include "tokenize.hpp"
#include "weight_table.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using core::pass1::Pass1Learner;
using core::pass1::Signature;
using core::pass1::to_positions;

namespace {

using Clock = std::chrono::steady_clock;
long ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

bool ends_with(std::string_view s, std::string_view suf) {
  return s.size() >= suf.size() &&
         s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string slurp(const std::string& p) {
  std::ifstream in(p);
  if (!in.is_open()) { std::cerr << "cannot open " << p << "\n"; std::exit(1); }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

core::parts::PartDictionary load_dict(const std::string& p) {
  if (ends_with(p, ".bin")) {
    std::ifstream in(p, std::ios::binary);
    return core::parts::load_dict_binary(in);
  }
  std::ifstream in(p);
  return core::parts::load_dict_text(in);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: " << argv[0]
              << " <dict> <wgt1> <out.cbk1> <corpus...>"
                 " [--K n] [--D n] [--ell n] [--strip f] [--lambda x]\n";
    return 2;
  }
  const std::string dict_path = argv[1];
  const std::string wgt_path = argv[2];
  const std::string out_path = argv[3];
  std::vector<std::string> corpora;
  std::size_t K = 8192, ell = 256;
  std::uint32_t D = 256;
  double strip = 0.0, lambda = 0.5;
  std::uint32_t alpha_num = 9, alpha_den = 10;
  std::size_t max_fired = 32;
  double cboost = 1.0;  // multiply C-band (identity) weights to emphasize it
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--K") K = std::stoul(next());
    else if (a == "--D") D = static_cast<std::uint32_t>(std::stoul(next()));
    else if (a == "--ell") ell = std::stoul(next());
    else if (a == "--strip") strip = std::stod(next());
    else if (a == "--lambda") lambda = std::stod(next());
    else if (a == "--alpha") {
      alpha_num = static_cast<std::uint32_t>(std::lround(std::stod(next()) * 10000.0));
      alpha_den = 10000;
    }
    else if (a == "--maxfired") max_fired = std::stoul(next());
    else if (a == "--cboost") cboost = std::stod(next());
    else corpora.push_back(a);
  }

  const auto dict = load_dict(dict_path);
  const std::size_t dim = 3 * dict.size();

  std::ifstream win(wgt_path, std::ios::binary);
  if (!win.is_open()) { std::cerr << "cannot open " << wgt_path << "\n"; return 1; }
  const auto table = core::signatures::load_surprisal(win);
  if (table.dim != dim) {
    std::cerr << "WGT1 dim " << table.dim << " != 3F " << dim << "\n";
    return 1;
  }
  std::vector<std::uint32_t> weights(table.weights.begin(), table.weights.end());

  // Build the sentence-scoped signature stream (as position lists).
  std::cerr << "F=" << dict.size() << " 3F=" << dim << " K=" << K
            << " D=" << D << " ell=" << ell << "\n";
  const auto t0 = Clock::now();
  std::vector<Signature> sigs;
  for (const auto& path : corpora) {
    const auto tokens = core::parts::tokenize_stream(slurp(path));
    for (const auto& sr : core::signatures::split_sentences(tokens))
      for (const auto& s : core::signatures::encode(dict, tokens, sr.start, sr.end))
        sigs.push_back(to_positions(s));
  }
  const auto t1 = Clock::now();
  std::cerr << "signatures: " << sigs.size() << "  (" << ms(t0, t1) << " ms)\n";

  // Frequent Directions factor.
  core::sketch::FrequentDirections fd(dim, ell);
  {
    binarycore::binary_vec::BigSparseBinaryVecDynamic v(dim);
    for (const auto& sig : sigs) {
      for (auto& c : v.chunks) c.data.clear();
      constexpr std::size_t cs =
          binarycore::binary_vec::BigSparseBinaryVecDynamic::chunk_size;
      for (std::uint32_t g : sig)
        v.chunks[g / cs].data.push_back(static_cast<std::uint16_t>(g % cs));
      fd.add(v);
    }
    fd.finalize();
  }
  const auto t2 = Clock::now();
  std::cerr << "FD rank=" << fd.rank() << "  (" << ms(t1, t2) << " ms)\n";

  Pass1Learner::Config cfg;
  cfg.K = K;
  cfg.D = D;
  cfg.lambda = lambda;
  cfg.c_band_boost = cboost;
  cfg.pursuit.alpha_num = alpha_num;
  cfg.pursuit.alpha_den = alpha_den;
  cfg.pursuit.max_fired = max_fired;
  std::cerr << "alpha=" << (double(alpha_num) / alpha_den)
            << " maxfired=" << max_fired << " lambda=" << lambda
            << " strip=" << strip << "\n";
  Pass1Learner learner(dim, weights, cfg);  // keep `weights` for BER reporting
  if (strip > 0.0) learner.strip_hubs(fd, strip);
  learner.seed(fd, sigs);
  const auto t3 = Clock::now();
  std::cerr << "seed  (" << ms(t2, t3) << " ms)\n";

  for (const auto& sig : sigs) learner.observe(sig);
  learner.finalize();
  const auto t4 = Clock::now();
  std::cerr << "train (" << ms(t3, t4) << " ms)\n";

  std::ofstream out(out_path, std::ios::binary);
  if (!out.is_open()) { std::cerr << "cannot open " << out_path << "\n"; return 1; }
  learner.codebook().save(out);

  // Block-wise reconstruction BER (§7.5.2): identity band C should reconstruct
  // near-losslessly; L/R (OR-pooled context) are graded.
  const auto& cb = learner.codebook();
  const std::uint32_t Fw = static_cast<std::uint32_t>(dict.size());
  double rL = 0, rC = 0, rR = 0, pC = 0;
  std::size_t n = 0;
  for (const auto& sig : sigs) {
    if (sig.empty()) continue;
    const auto fired = core::codebook::pursuit_encode(cb, sig, weights, cfg.pursuit);
    const auto recon = cb.decode(fired);
    const auto m = core::codebook::recon_metrics(recon, sig, weights, Fw);
    rL += m.recall_L; rC += m.recall_C; rR += m.recall_R; pC += m.precision_C;
    ++n;
  }
  if (n) { rL /= n; rC /= n; rR /= n; pC /= n; }

  std::cerr << "utilization: " << learner.utilization()
            << "  mean_recall: " << learner.mean_recall(sigs) << "\n";
  std::cerr << "block recall  C=" << rC << " L=" << rL << " R=" << rR
            << "   C precision=" << pC << "\n";
  std::cerr << "wrote " << out_path << "\n";
  return 0;
}
