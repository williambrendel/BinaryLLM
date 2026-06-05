// ============================================================================
// apps/phase1_extract/main.cpp
//
// Reads raw text from one or more input files, tokenizes each, runs the
// part extractor over the combined token streams (frequencies accumulate
// across files), and writes the resulting part dictionary to the final
// output path.
//
// Usage:
//   phase1_extract <input1.txt> [input2.txt] ... <output_dict[.bin]|->
//   phase1_extract -                              <output_dict[.bin]|->     # stdin as only input
//
// Last arg is the output; everything before is input.
// Use "-" to mean stdin (only valid as the lone input) or stdout (text only).
//
// Output format is chosen by extension:
//   .bin   →  compact binary (BDIC header + per-kind atom blocks)
//   else   →  human-readable text (id\tkind\tvalue\n per line)
//
// Per-file stats are printed to stderr, followed by an aggregate
// summary and the dictionary breakdown.
// ============================================================================

#include "binarytrain/parts/dictionary.hpp"
#include "binarytrain/parts/extractor.hpp"
#include "binarytrain/parts/tokenize.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace binarytrain::parts;

namespace {

void usage(const char* prog) {
  std::cerr << "usage: " << prog
            << " <input1.txt|-> [input2.txt] ... <output_dict[.bin]|->\n";
}

std::string slurp(std::istream& in) {
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct FileStats {
  std::size_t words = 0;
  std::size_t delims = 0;
  std::size_t total() const { return words + delims; }
};

FileStats ingest(const std::string& path, PartExtractor& ex) {
  std::string raw;
  if (path == "-") {
    raw = slurp(std::cin);
  } else {
    std::ifstream fin(path);
    if (!fin.is_open()) {
      std::cerr << "error: cannot open '" << path << "'\n";
      std::exit(1);
    }
    raw = slurp(fin);
  }
  const auto tokens = tokenize_stream(raw);
  FileStats stats;
  for (const auto& t : tokens) {
    if (t.type == StreamToken::Type::Word) {
      ex.add_word(t.value);
      ++stats.words;
    } else {
      ex.add_delimiter(t.value);
      ++stats.delims;
    }
  }
  return stats;
}

bool path_ends_with(const std::string& s, std::string_view suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(),
                   suffix.size(), suffix) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 2;
  }

  std::vector<std::string> input_paths;
  for (int i = 1; i < argc - 1; ++i) input_paths.emplace_back(argv[i]);
  const std::string output_path = argv[argc - 1];

  if (input_paths.size() > 1) {
    for (const auto& p : input_paths) {
      if (p == "-") {
        std::cerr << "error: '-' (stdin) is only valid as the lone input\n";
        return 2;
      }
    }
  }

  ExtractorConfig cfg;
  cfg.whole_max_len = 16;
  PartExtractor ex(cfg);

  FileStats total;
  for (std::size_t i = 0; i < input_paths.size(); ++i) {
    const auto stats = ingest(input_paths[i], ex);
    std::cerr << "input " << (i + 1) << ": " << input_paths[i] << "\n"
              << "  tokens: " << stats.total()
              << " (words=" << stats.words
              << ", delims=" << stats.delims << ")\n";
    total.words += stats.words;
    total.delims += stats.delims;
  }
  if (input_paths.size() > 1) {
    std::cerr << "total tokens: " << total.total()
              << " (words=" << total.words
              << ", delims=" << total.delims << ")\n";
  }

  PartDictionary dict = ex.finalize();

  // Choose output format from the path extension.
  if (output_path == "-") {
    // stdout always gets text.
    dict.save(std::cout);
  } else if (path_ends_with(output_path, ".bin")) {
    std::ofstream fout(output_path, std::ios::binary);
    if (!fout.is_open()) {
      std::cerr << "error: cannot open '" << output_path << "' for write\n";
      return 1;
    }
    dict.save_binary(fout);
  } else {
    std::ofstream fout(output_path);
    if (!fout.is_open()) {
      std::cerr << "error: cannot open '" << output_path << "' for write\n";
      return 1;
    }
    dict.save(fout);
  }

  std::cerr << "peel iterations: " << ex.last_peel_iterations() << "\n";
  std::cerr << "dictionary size F = " << dict.size() << "\n";
  std::cerr << "  whole:  " << dict.count_of_kind(Kind::Whole)     << "\n";
  std::cerr << "  start:  " << dict.count_of_kind(Kind::Start)     << "\n";
  std::cerr << "  mid:    " << dict.count_of_kind(Kind::Mid)       << "\n";
  std::cerr << "  end:    " << dict.count_of_kind(Kind::End)       << "\n";
  std::cerr << "  letter: " << dict.count_of_kind(Kind::Letter)    << "\n";
  std::cerr << "  delim:  " << dict.count_of_kind(Kind::Delimiter) << "\n";

  return 0;
}
