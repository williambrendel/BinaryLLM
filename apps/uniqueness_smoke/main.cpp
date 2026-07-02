// ============================================================================
// apps/uniqueness_smoke/main.cpp
//
// Smoke test for Option B's uniqueness contract on a real vocabulary,
// with strategy comparison via dict_reorder side functions and a
// trained-only dict file format (load auto-augments singletons +
// special characters).
//
// Workflow:
//   1. Load dictionary (auto-detect text vs binary by .bin extension).
//      Load augments with singletons + connector specials so the
//      runtime dict has full coverage.
//   2. If loaded from text, also write a binary cache (.bin) for fast
//      reload next time.
//   3. For each requested strategy:
//        a. Reorder the dict (or use as-is for "discovery").
//        b. Encode every word.
//        c. Report parts-per-word distribution, collisions, top-N
//           collision groups, per-phase timing.
//
// Strategies (selected via --algorithm):
//   discovery     — original dict (no reorder)
//   type          — kind first (S, E, M), original sequence within
//   type-level    — kind first, longest L first within
//   level-type    — longest L first, kind within (S, E, M)
//   all (default) — runs all four with a side-by-side summary
//
// Usage:
//   uniqueness_smoke <dict_path> <words_file>
//                    [--top N]
//                    [--algorithm=discovery|type|type-level|level-type|all]
// ============================================================================

#include "dict_io.hpp"
#include "dict_reorder.hpp"
#include "dictionary.hpp"
#include "tokenize.hpp"      // ascii_lowercase
#include "word_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void usage(const char* prog) {
  std::cerr << "usage: " << prog
            << " <dict_path> <words_file>"
            << " [--top N]"
            << " [--algorithm=discovery|type|type-level|level-type|all]\n";
}

bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string with_extension(const std::string& path, const std::string& ext) {
  const auto dot = path.find_last_of('.');
  if (dot == std::string::npos) return path + ext;
  return path.substr(0, dot) + ext;
}

bool file_exists(const std::string& path) {
  std::ifstream f(path);
  return f.good();
}

std::uint64_t hash_bag(const std::vector<std::uint32_t>& bag) {
  std::uint64_t h = 14695981039346656037ULL;
  for (std::uint32_t id : bag) {
    const std::uint32_t v = id;
    h ^= static_cast<std::uint64_t>(v & 0xff);         h *= 1099511628211ULL;
    h ^= static_cast<std::uint64_t>((v >>  8) & 0xff); h *= 1099511628211ULL;
    h ^= static_cast<std::uint64_t>((v >> 16) & 0xff); h *= 1099511628211ULL;
    h ^= static_cast<std::uint64_t>((v >> 24) & 0xff); h *= 1099511628211ULL;
  }
  return h;
}

std::vector<std::string> load_words(const std::string& path) {
  std::vector<std::string> out;
  std::ifstream in(path);
  if (!in.is_open()) {
    std::cerr << "error: cannot open words file '" << path << "'\n";
    std::exit(1);
  }
  std::string line;
  while (std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
      line.pop_back();
    }
    if (!line.empty()) out.push_back(std::move(line));
  }
  return out;
}

core::parts::PartDictionary load_dict_auto(const std::string& path) {
  const bool binary = ends_with(path, ".bin");
  if (binary) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "error: cannot open dict '" << path << "'\n";
      std::exit(1);
    }
    return core::parts::load_dict_binary(in);
  }
  std::ifstream in(path);
  if (!in.is_open()) {
    std::cerr << "error: cannot open dict '" << path << "'\n";
    std::exit(1);
  }
  return core::parts::load_dict_text(in);
}

void save_binary_cache(const core::parts::PartDictionary& dict,
                       const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "warning: cannot write binary cache '" << path << "'\n";
    return;
  }
  core::parts::save_dict_binary(dict, out);
}

struct StrategyResult {
  std::string name;
  std::int64_t reorder_us = 0;
  std::int64_t encode_us = 0;
  std::int64_t analyze_us = 0;
  std::size_t encoded = 0;
  std::size_t empty_bags = 0;
  std::size_t distinct_bags = 0;
  std::size_t collision_keys = 0;
  std::size_t colliding_words = 0;
  double parts_min = 0, parts_avg = 0, parts_p50 = 0;
  double parts_p90 = 0, parts_p99 = 0, parts_max = 0;
  std::vector<std::vector<std::string>> top_groups;
};

using Clock = std::chrono::steady_clock;
auto now() { return Clock::now(); }
std::int64_t us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

core::parts::PartDictionary apply_strategy(
    const core::parts::PartDictionary& original,
    const std::string& strategy) {
  if (strategy == "discovery")  return original;
  if (strategy == "type")       return core::parts::reorder_dict_by_type(original);
  if (strategy == "type-level") return core::parts::reorder_dict_by_type_then_level(original);
  if (strategy == "level-type") return core::parts::reorder_dict_by_level_then_type(original);
  std::cerr << "error: unknown strategy '" << strategy << "'\n";
  std::exit(2);
}

StrategyResult run_strategy(
    const core::parts::PartDictionary& original_dict,
    const std::vector<std::string>& words,
    const std::string& strategy,
    std::size_t top_n) {
  StrategyResult r;
  r.name = strategy;

  const auto t0 = now();
  const auto dict = (strategy == "discovery")
      ? original_dict
      : apply_strategy(original_dict, strategy);
  const auto t1 = now();
  r.reorder_us = us(t0, t1);

  std::unordered_map<std::uint64_t, std::vector<std::string>> by_key;
  by_key.reserve(words.size());
  std::vector<std::size_t> bag_sizes;
  bag_sizes.reserve(words.size());
  std::size_t total_parts = 0;

  const auto t2 = now();
  for (const auto& w : words) {
    const std::string w_lower = core::parts::ascii_lowercase(w);
    auto bag = core::signatures::encode_word(dict, w_lower);
    if (bag.empty()) {
      ++r.empty_bags;
      continue;
    }
    bag_sizes.push_back(bag.size());
    total_parts += bag.size();
    by_key[hash_bag(bag)].push_back(w);
  }
  const auto t3 = now();
  r.encode_us = us(t2, t3);
  r.encoded = words.size() - r.empty_bags;

  if (!bag_sizes.empty()) {
    std::sort(bag_sizes.begin(), bag_sizes.end());
    auto pct = [&](double p) -> double {
      std::size_t i = static_cast<std::size_t>(p * bag_sizes.size());
      if (i >= bag_sizes.size()) i = bag_sizes.size() - 1;
      return static_cast<double>(bag_sizes[i]);
    };
    r.parts_min = static_cast<double>(bag_sizes.front());
    r.parts_avg = static_cast<double>(total_parts) /
                  static_cast<double>(bag_sizes.size());
    r.parts_p50 = pct(0.50);
    r.parts_p90 = pct(0.90);
    r.parts_p99 = pct(0.99);
    r.parts_max = static_cast<double>(bag_sizes.back());
  }

  const auto t4 = now();
  r.distinct_bags = by_key.size();
  std::vector<std::pair<std::size_t, std::vector<std::string>*>> groups;
  groups.reserve(by_key.size());
  for (auto& [key, ws] : by_key) {
    if (ws.size() > 1) {
      ++r.collision_keys;
      r.colliding_words += ws.size();
      groups.emplace_back(ws.size(), &ws);
    }
  }
  std::sort(groups.begin(), groups.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  const std::size_t shown = std::min(top_n, groups.size());
  for (std::size_t i = 0; i < shown; ++i) {
    r.top_groups.push_back(*groups[i].second);
  }
  const auto t5 = now();
  r.analyze_us = us(t4, t5);

  return r;
}

void print_strategy_report(const StrategyResult& r) {
  std::cout << "\n========== strategy: " << r.name << " ==========\n";
  std::cout << "  encoded:       " << r.encoded << "\n";
  std::cout << "  empty bag:     " << r.empty_bags << "\n";
  std::cout << "  parts/word:    min=" << r.parts_min
            << " avg=" << std::fixed << std::setprecision(2) << r.parts_avg
            << " p50=" << r.parts_p50
            << " p90=" << r.parts_p90
            << " p99=" << r.parts_p99
            << " max=" << r.parts_max << "\n";
  std::cout << std::defaultfloat;
  std::cout << "  distinct bags: " << r.distinct_bags << "\n";
  std::cout << "  collisions:    " << r.collision_keys
            << " keys (" << r.colliding_words << " words)\n";
  if (r.encoded > 0) {
    const double rate = 100.0 * static_cast<double>(r.colliding_words) /
                        static_cast<double>(r.encoded);
    std::cout << "  collision rate: " << std::fixed << std::setprecision(4)
              << rate << "%\n";
    std::cout << std::defaultfloat;
  }
  std::cout << "  timing (ms):   reorder=" << (r.reorder_us / 1000)
            << "  encode=" << (r.encode_us / 1000)
            << "  analyze=" << (r.analyze_us / 1000) << "\n";

  if (!r.top_groups.empty()) {
    std::cout << "\n  Top " << r.top_groups.size()
              << " collision groups (largest first):\n";
    for (const auto& ws : r.top_groups) {
      std::cout << "    [" << ws.size() << "] ";
      const std::size_t cap = 8;
      for (std::size_t j = 0; j < std::min(cap, ws.size()); ++j) {
        if (j > 0) std::cout << "  ";
        std::cout << ws[j];
      }
      if (ws.size() > cap) std::cout << ", ...";
      std::cout << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 2;
  }
  const std::string dict_path = argv[1];
  const std::string words_path = argv[2];
  std::size_t top_n = 20;
  std::string algorithm = "all";

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--top" && i + 1 < argc) {
      top_n = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (arg.rfind("--top=", 0) == 0) {
      top_n = static_cast<std::size_t>(std::stoul(arg.substr(6)));
    } else if (arg.rfind("--algorithm=", 0) == 0) {
      algorithm = arg.substr(12);
    } else {
      std::cerr << "unknown arg: " << arg << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  // Phase: load dict (augments singletons + specials internally).
  const auto t_load_begin = now();
  auto dict = load_dict_auto(dict_path);
  const auto t_load_end = now();
  const auto load_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          t_load_end - t_load_begin).count();

  // Phase: write binary cache if loaded from text and no .bin exists.
  std::int64_t save_ms = 0;
  std::string bin_cache_path;
  if (!ends_with(dict_path, ".bin")) {
    bin_cache_path = with_extension(dict_path, ".bin");
    if (!file_exists(bin_cache_path)) {
      const auto t_save_begin = now();
      save_binary_cache(dict, bin_cache_path);
      const auto t_save_end = now();
      save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    t_save_end - t_save_begin).count();
    }
  }

  // Phase: load words.
  const auto t_words_begin = now();
  auto words = load_words(words_path);
  const auto t_words_end = now();
  const auto words_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          t_words_end - t_words_begin).count();

  std::cout << "dictionary: " << dict_path
            << " (F = " << dict.size() << ", augmented)\n";
  std::cout << "words:      " << words_path
            << " (" << words.size() << " entries)\n";
  std::cout << "timing (ms): load_dict=" << load_ms
            << "  load_words=" << words_ms;
  if (save_ms > 0) {
    std::cout << "  save_bin_cache=" << save_ms
              << " (" << bin_cache_path << ")";
  }
  std::cout << "\n";

  std::vector<std::string> strategies;
  if (algorithm == "all") {
    strategies = {"discovery", "type", "type-level", "level-type"};
  } else {
    strategies = {algorithm};
  }

  std::vector<StrategyResult> results;
  for (const auto& s : strategies) {
    results.push_back(run_strategy(dict, words, s, top_n));
  }
  for (const auto& r : results) print_strategy_report(r);

  if (results.size() > 1) {
    std::cout << "\n========== summary ==========\n";
    std::cout << std::left;
    std::cout << std::setw(14) << "strategy"
              << std::setw(12) << "F-collide%"
              << std::setw(10) << "avg-parts"
              << std::setw(10) << "p99-parts"
              << std::setw(12) << "encode(ms)" << "\n";
    for (const auto& r : results) {
      const double rate = (r.encoded > 0)
          ? 100.0 * static_cast<double>(r.colliding_words) /
            static_cast<double>(r.encoded)
          : 0.0;
      std::cout << std::setw(14) << r.name
                << std::setw(12) << std::fixed << std::setprecision(4) << rate
                << std::setw(10) << std::setprecision(2) << r.parts_avg
                << std::setw(10) << std::setprecision(0) << r.parts_p99
                << std::setw(12) << (r.encode_us / 1000) << "\n";
    }
    std::cout << std::defaultfloat;
  }

  return 0;
}
