// encode_corpus
// -------------
// Reads a UTF-8 text file, tokenizes it, encodes every word token, and reports:
//   - total word tokens
//   - unique encodings
//   - collision count and rate
//   - top-N collision groups (which distinct words share an encoding)
//
// Usage:
//   encode_corpus <path>                       Default: top-20 collisions
//   encode_corpus <path> --top N               Show top-N collisions
//   encode_corpus <path> --top N --quiet       Suppress per-group printout
//
// We treat upper/lowercase as separate words for collision counting because the
// encoder's capitalization bits preserve the distinction (HELLO vs hello vs Hello
// have different encodings — that's a feature, not a collision).

#include "binarycore/token_encoder.hpp"
#include "binarycore/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Args {
  std::string path;
  int top_n = 20;
  bool quiet = false;
};

bool parse_args(int argc, char** argv, Args& out) {
  if (argc < 2) {
    std::cerr << "Usage: encode_corpus <path> [--top N] [--quiet]\n";
    return false;
  }
  out.path = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--top" && i + 1 < argc) {
      out.top_n = std::atoi(argv[++i]);
      if (out.top_n < 0)
        out.top_n = 0;
    } else if (a == "--quiet") {
      out.quiet = true;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      return false;
    }
  }
  return true;
}

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "Failed to open: " << path << "\n";
    std::exit(2);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

} // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args))
    return 1;

  const std::string text = slurp(args.path);

  // Tokenize the entire file at once.
  auto tokens = binarycore::tokenize(text);

  // Bucket distinct word *texts* by their encoded value.
  // map[encoding] -> set of distinct strings that produced it
  std::unordered_map<uint64_t, std::unordered_set<std::string>> by_encoding;
  by_encoding.reserve(tokens.size() / 4);

  std::size_t total_word_tokens = 0;
  std::size_t total_symbol_tokens = 0;

  for (const auto& t : tokens) {
    if (t.is_word()) {
      ++total_word_tokens;
      by_encoding[t.encoded].insert(t.text);
    } else {
      ++total_symbol_tokens;
    }
  }

  // Count collisions: an encoding is "colliding" iff it covers more than one
  // distinct source string.
  std::size_t colliding_encodings = 0;
  std::size_t colliding_words = 0; // total distinct strings inside collisions
  for (const auto& [enc, strings] : by_encoding) {
    if (strings.size() > 1) {
      ++colliding_encodings;
      colliding_words += strings.size();
    }
  }

  const std::size_t unique_encodings = by_encoding.size();
  const std::size_t unique_strings = [&] {
    std::size_t s = 0;
    for (const auto& [_, strings] : by_encoding)
      s += strings.size();
    return s;
  }();

  // ---- Report ----
  std::cout << "Corpus:                " << args.path << "\n";
  std::cout << "Bytes read:            " << text.size() << "\n";
  std::cout << "\n";
  std::cout << "Total tokens:          " << tokens.size() << "\n";
  std::cout << "  word tokens:         " << total_word_tokens << "\n";
  std::cout << "  symbol tokens:       " << total_symbol_tokens << "\n";
  std::cout << "\n";
  std::cout << "Distinct word strings: " << unique_strings << "\n";
  std::cout << "Distinct encodings:    " << unique_encodings << "\n";

  if (unique_strings > 0) {
    const double collision_rate = 100.0 * static_cast<double>(unique_strings - unique_encodings) /
                                  static_cast<double>(unique_strings);
    std::cout << "Colliding encodings:   " << colliding_encodings << "  (" << colliding_words
              << " distinct strings affected)\n";
    std::cout << "Collision rate:        " << collision_rate << "%  "
              << "  (= (distinct_strings - distinct_encodings) / distinct_strings)\n";
  }

  // ---- Top-N collision groups ----
  if (args.top_n > 0 && !args.quiet) {
    // Build a list of (group_size, sample_strings) for groups with >1 member.
    std::vector<std::pair<std::size_t, std::vector<std::string>>> groups;
    groups.reserve(colliding_encodings);
    for (const auto& [enc, strings] : by_encoding) {
      if (strings.size() <= 1)
        continue;
      std::vector<std::string> v(strings.begin(), strings.end());
      std::sort(v.begin(), v.end());
      groups.emplace_back(v.size(), std::move(v));
    }
    // Largest groups first.
    std::sort(groups.begin(), groups.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    const std::size_t to_show =
        std::min<std::size_t>(static_cast<std::size_t>(args.top_n), groups.size());

    if (to_show > 0) {
      std::cout << "\nTop " << to_show << " collision groups (largest first):\n";
      for (std::size_t i = 0; i < to_show; ++i) {
        std::cout << "  [" << groups[i].first << "]";
        std::size_t shown = 0;
        for (const auto& s : groups[i].second) {
          if (shown++ >= 8) {
            std::cout << ", ...";
            break;
          }
          std::cout << "  " << s;
        }
        std::cout << "\n";
      }
    }
  }

  return 0;
}
