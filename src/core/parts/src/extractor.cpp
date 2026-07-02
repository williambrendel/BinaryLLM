// ============================================================================
// core/parts/src/extractor.cpp
// ============================================================================

#include "extractor.hpp"
#include "decomposer.hpp"

#include "adaptive_threshold.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::parts {

void PartExtractor::add_word(std::string_view word) {
  ++word_count_;
  if (word.empty()) return;
  ++all_word_freq_[std::string(word)];
}

void PartExtractor::add_delimiter(std::string_view delim) {
  if (delim.empty()) return;
  ++delim_freq_[std::string(delim)];
}

namespace {

// Connector characters that the tokenizer may absorb into words. Each
// is seeded into the dictionary in three places:
//
//   (1) As a single-character Whole atom — every char the tokenizer
//       can emit inside a word must have at least one atom
//       representation for the decomposer to use as a last resort.
//   (2) As three positional Letter atoms ("x##", "##x##", "##x") —
//       symmetric with the letter/digit Letter atoms, so any
//       character position can be encoded with position information.
//   (3) As a Delimiter atom — so the decoder can emit the character
//       as a standalone delimiter for character-level reconstruction
//       of unseen tokens.
//
// This list must stay in sync with tokenize.cpp's connector rules
// and number-prefix rule:
//   '-', '\'', '&', ',', '.'  → in-word connectors
//   '$'                       → leading currency for numbers
const char* const kConnectorChars[] = {"-", "'", "&", ",", ".", "$"};

std::vector<std::pair<std::string, std::uint64_t>> sort_by_freq_desc(
    const std::unordered_map<std::string, std::uint64_t>& freq) {
  std::vector<std::pair<std::string, std::uint64_t>> v;
  v.reserve(freq.size());
  for (const auto& [k, f] : freq) v.emplace_back(k, f);
  std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) {
              if (a.second != b.second) return a.second > b.second;
              return a.first < b.first;
            });
  return v;
}

std::size_t pruned_count(
    const std::vector<std::pair<std::string, std::uint64_t>>& sorted_freq) {
  if (sorted_freq.empty()) return 0;
  std::vector<double> values;
  values.reserve(sorted_freq.size());
  for (const auto& [_, f] : sorted_freq) {
    values.push_back(static_cast<double>(f));
  }
  return core::adaptive_threshold::adaptive_prune_count(values);
}

std::size_t add_top_k(
    PartDictionary& dict, Kind kind,
    const std::vector<std::pair<std::string, std::uint64_t>>& sorted,
    std::size_t k) {
  std::size_t added = 0;
  for (std::size_t i = 0; i < k && i < sorted.size(); ++i) {
    if (dict.lookup(kind, sorted[i].first) == kInvalidPartId) {
      dict.add(kind, sorted[i].first);
      ++added;
    }
  }
  return added;
}

// Add the three positional Letter atoms for a single-character value
// using the "##" WordPiece-style position marker.
void add_letter_positions(PartDictionary& dict, std::string s) {
  dict.add(Kind::Letter, s + "##");        // start of word
  dict.add(Kind::Letter, "##" + s + "##"); // middle
  dict.add(Kind::Letter, "##" + s);        // end of word
}

std::size_t run_peel_iteration(
    PartDictionary& dict,
    const std::unordered_map<std::string, std::uint64_t>& all_word_freq,
    const ExtractorConfig& cfg,
    std::size_t& total_runs_out) {
  total_runs_out = 0;
  std::size_t parts_added = 0;

  struct PoolEntry {
    std::string substring;
    bool touches_start;
    bool touches_end;
    std::uint64_t weight;
  };
  std::vector<PoolEntry> pool;
  pool.reserve(all_word_freq.size());

  for (const auto& [word, freq] : all_word_freq) {
    auto runs = find_singleton_runs(dict, word);
    for (const auto& r : runs) {
      const std::size_t len = r.end - r.start;
      if (len < 2) continue;
      PoolEntry e;
      e.substring = word.substr(r.start, len);
      e.touches_start = r.touches_word_start;
      e.touches_end = r.touches_word_end;
      e.weight = freq;
      pool.push_back(std::move(e));
    }
  }
  total_runs_out = pool.size();
  if (pool.empty()) return 0;

  // L sweep matches the decomposer's L range (see kMinPartLength /
  // kMaxPartLength in part.hpp). These MUST stay in sync — a part
  // produced here but never queried by the decomposer is dead weight.
  for (std::size_t L = kMaxPartLength; L >= kMinPartLength; --L) {
    std::unordered_map<std::string, std::uint64_t> start_bin;
    std::unordered_map<std::string, std::uint64_t> mid_bin;
    std::unordered_map<std::string, std::uint64_t> end_bin;

    for (const auto& e : pool) {
      const std::size_t n = e.substring.size();
      if (L > n) continue;

      for (std::size_t i = 0; i + L <= n; ++i) {
        mid_bin[e.substring.substr(i, L)] += e.weight;
      }
      if (e.touches_start) {
        start_bin[e.substring.substr(0, L)] += e.weight;
      }
      if (e.touches_end) {
        end_bin[e.substring.substr(n - L, L)] += e.weight;
      }
    }

    {
      auto sorted = sort_by_freq_desc(start_bin);
      const std::size_t k = pruned_count(sorted);
      parts_added += add_top_k(dict, Kind::Start, sorted, k);
    }
    {
      auto sorted = sort_by_freq_desc(mid_bin);
      const std::size_t k = pruned_count(sorted);
      parts_added += add_top_k(dict, Kind::Mid, sorted, k);
    }
    {
      auto sorted = sort_by_freq_desc(end_bin);
      const std::size_t k = pruned_count(sorted);
      parts_added += add_top_k(dict, Kind::End, sorted, k);
    }
  }

  (void)cfg;
  return parts_added;
}

}  // namespace

PartDictionary PartExtractor::finalize() {
  PartDictionary dict;
  last_peel_iters_ = 0;

  // === Seed phase ============================================== //

  // Whole atoms: length-bound + cascade prune on frequency.
  {
    std::vector<std::pair<std::string, std::uint64_t>> wholes;
    wholes.reserve(all_word_freq_.size());
    for (const auto& [v, f] : all_word_freq_) {
      if (v.size() >= 2 && v.size() <= cfg_.whole_max_len) {
        wholes.emplace_back(v, f);
      }
    }
    std::sort(wholes.begin(), wholes.end(),
              [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
              });

    std::vector<double> freqs;
    freqs.reserve(wholes.size());
    for (const auto& [_, f] : wholes) freqs.push_back(static_cast<double>(f));
    const std::size_t k = core::adaptive_threshold::adaptive_prune_count(freqs);

    for (std::size_t i = 0; i < k && i < wholes.size(); ++i) {
      dict.add(Kind::Whole, wholes[i].first);
    }
  }

  // Single-character Whole atoms (coverage backstop):
  //   26 letters + 10 digits + 6 connector chars = 42.
  if (cfg_.add_letter_singletons) {
    for (char c = 'a'; c <= 'z'; ++c) {
      dict.add(Kind::Whole, std::string(1, c));
    }
    for (char c = '0'; c <= '9'; ++c) {
      dict.add(Kind::Whole, std::string(1, c));
    }
    for (const char* c : kConnectorChars) {
      dict.add(Kind::Whole, c);
    }
  }

  // Positional Letter atoms with "##" WordPiece-style encoding:
  //   26 letters × 3 positions = 78
  //   10 digits  × 3 positions = 30
  //   6 connectors × 3 positions = 18
  // Total: 126.
  //
  // The "##" separator is collision-free across our character set
  // since '#' is in neither alnum nor any connector class. Every
  // (character, position) pair maps to a unique string.
  if (cfg_.add_letter_singletons) {
    for (char c = 'a'; c <= 'z'; ++c) {
      add_letter_positions(dict, std::string(1, c));
    }
    for (char c = '0'; c <= '9'; ++c) {
      add_letter_positions(dict, std::string(1, c));
    }
    for (const char* c : kConnectorChars) {
      add_letter_positions(dict, std::string(c));
    }
  }

  // Observed delimiters in frequency order.
  {
    auto sorted = sort_by_freq_desc(delim_freq_);
    for (const auto& [v, _] : sorted) dict.add(Kind::Delimiter, v);
  }

  // Unconditional connector delimiter atoms (completeness backstop).
  for (const char* c : kConnectorChars) {
    dict.add(Kind::Delimiter, c);
  }

  // === Peel phase ============================================== //

  for (std::size_t iter = 0; iter < cfg_.max_peel_iterations; ++iter) {
    std::size_t total_runs = 0;
    const std::size_t added = run_peel_iteration(
        dict, all_word_freq_, cfg_, total_runs);
    ++last_peel_iters_;
    if (total_runs == 0) break;
    if (added == 0) break;
  }

  return dict;
}

}  // namespace core::parts
