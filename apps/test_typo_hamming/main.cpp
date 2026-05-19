// Stand-alone tool: measure encoder Hamming distance under typos.
// Place at apps/test_typo_hamming/main.cpp and add to apps/CMakeLists.txt.

#include "binarycore/encoding/token_encoder.hpp"
#include "binarycore/encoding/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using namespace binarycore;


static int hamming(TokenVec a, TokenVec b) {
  return __builtin_popcountll(a.data[0] ^ b.data[0]);
}

// Apply a typo of given type at a random position. Returns modified string.
static std::string apply_substitution(const std::string& w, std::mt19937& rng) {
  if (w.empty()) return w;
  std::string out = w;
  std::uniform_int_distribution<size_t> pos_d(0, w.size() - 1);
  std::uniform_int_distribution<int> ch_d(0, 25);
  size_t p = pos_d(rng);
  char old = out[p];
  char nw;
  do {
    nw = static_cast<char>('a' + ch_d(rng));
  } while (nw == old);
  out[p] = nw;
  return out;
}

static std::string apply_transposition(const std::string& w, std::mt19937& rng) {
  if (w.size() < 2) return w;
  std::string out = w;
  std::uniform_int_distribution<size_t> pos_d(0, w.size() - 2);
  size_t p = pos_d(rng);
  std::swap(out[p], out[p + 1]);
  return out;
}

static std::string apply_insertion(const std::string& w, std::mt19937& rng) {
  std::uniform_int_distribution<size_t> pos_d(0, w.size());
  std::uniform_int_distribution<int> ch_d(0, 25);
  size_t p = pos_d(rng);
  std::string out = w;
  out.insert(p, 1, static_cast<char>('a' + ch_d(rng)));
  return out;
}

static std::string apply_deletion(const std::string& w, std::mt19937& rng) {
  if (w.empty()) return w;
  std::uniform_int_distribution<size_t> pos_d(0, w.size() - 1);
  std::string out = w;
  out.erase(pos_d(rng), 1);
  return out;
}

struct Stats {
  std::vector<int> distances;
  void add(int d) { distances.push_back(d); }
  void report(const char* name) const {
    if (distances.empty()) {
      std::printf("  %-15s no samples\n", name);
      return;
    }
    auto sorted = distances;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0;
    for (int d : sorted) sum += d;
    double mean = sum / sorted.size();
    int p50 = sorted[sorted.size() / 2];
    int p90 = sorted[sorted.size() * 9 / 10];
    int p99 = sorted[sorted.size() * 99 / 100];
    int mx = sorted.back();
    std::printf("  %-15s n=%5zu  mean=%5.2f  p50=%2d  p90=%2d  p99=%2d  max=%2d\n",
                name, sorted.size(), mean, p50, p90, p99, mx);
  }
};

static std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss; ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s corpus.txt [n_samples=2000]\n", argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  const int n_samples = (argc >= 3) ? std::atoi(argv[2]) : 2000;
  const std::string text = slurp(path);

  // Collect distinct lowercase alphabetic words of length >= 3.
  auto toks = tokenize(text);
  std::unordered_set<std::string> distinct;
  std::vector<std::string> words;
  for (auto& t : toks) {
    if (!t.is_word()) continue;
    std::string lc;
    lc.reserve(t.text.size());
    bool all_alpha = true;
    for (char c : t.text) {
      if (c >= 'a' && c <= 'z') lc.push_back(c);
      else if (c >= 'A' && c <= 'Z') lc.push_back(c - 'A' + 'a');
      else { all_alpha = false; break; }
    }
    if (!all_alpha || lc.size() < 3) continue;
    if (distinct.insert(lc).second) words.push_back(lc);
  }
  std::fprintf(stderr, "Loaded %zu distinct alphabetic words from %s\n",
               words.size(), path.c_str());

  std::mt19937 rng(42);
  std::shuffle(words.begin(), words.end(), rng);
  int n = std::min<int>(n_samples, words.size());

  Stats sub, trans, ins, del;
  for (int i = 0; i < n; ++i) {
    const auto& w = words[i];
    TokenVec enc = encode_word(w);

    {
      auto v = apply_substitution(w, rng);
      if (v != w) sub.add(hamming(enc, encode_word(v)));
    }
    {
      auto v = apply_transposition(w, rng);
      if (v != w) trans.add(hamming(enc, encode_word(v)));
    }
    {
      auto v = apply_insertion(w, rng);
      if (v != w) ins.add(hamming(enc, encode_word(v)));
    }
    {
      auto v = apply_deletion(w, rng);
      if (v != w) del.add(hamming(enc, encode_word(v)));
    }
  }

  std::printf("Typo Hamming distance (64-bit tokens):\n");
  sub.report("substitution");
  trans.report("transposition");
  ins.report("insertion");
  del.report("deletion");
  std::printf("\nFor reference, random 64-bit pairs would have mean=32.\n");
  return 0;
}
