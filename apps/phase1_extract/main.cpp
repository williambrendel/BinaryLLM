// ============================================================================
// apps/phase1_extract/main.cpp
//
// Reads raw text from one or more files (or stdin), tokenizes each,
// feeds them into a single PartExtractor, and writes the resulting
// part dictionary in the trained-only format (singletons + connector
// specials are NOT written — they're regenerated on load by
// augment_with_atoms).
//
// Output format is chosen by file extension:
//   .bin                → binary (BDI2 format, uint8 kind flag per part)
//   anything else / -   → text   ("##" convention, [delim] section)
//
// Usage:
//   phase1_extract <output_dict.txt|.bin|-> <input.txt|-> [<input> ...]
// ============================================================================

#include "dict_io.hpp"
#include "dictionary.hpp"
#include "extractor.hpp"
#include "tokenize.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace core::parts;

namespace {

void usage(const char* prog) {
  std::cerr << "usage: " << prog
            << " <output_dict.txt|.bin|-> <input.txt|-> [<input.txt> ...]\n";
}

std::string slurp(std::istream& in) {
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool read_input(const std::string& path, std::string& out) {
  if (path == "-") {
    out = slurp(std::cin);
    return true;
  }
  std::ifstream fin(path);
  if (!fin.is_open()) {
    std::cerr << "error: cannot open '" << path << "'\n";
    return false;
  }
  out = slurp(fin);
  return true;
}

bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 2;
  }
  const std::string out_path = argv[1];
  std::vector<std::string> in_paths;
  for (int i = 2; i < argc; ++i) in_paths.emplace_back(argv[i]);

  PartExtractor ex;
  std::size_t total_tokens = 0;
  std::size_t total_words = 0;
  std::size_t total_delims = 0;

  for (const auto& in_path : in_paths) {
    std::string raw;
    if (!read_input(in_path, raw)) return 1;
    auto tokens = tokenize_stream(raw);
    std::size_t words = 0, delims = 0;
    for (const auto& t : tokens) {
      if (t.type == StreamToken::Type::Word) {
        ex.add_word(t.value);
        ++words;
      } else {
        ex.add_delimiter(t.value);
        ++delims;
      }
    }
    std::cerr << "  " << in_path << ": "
              << tokens.size() << " tokens ("
              << words << " words, " << delims << " delims)\n";
    total_tokens += tokens.size();
    total_words += words;
    total_delims += delims;
  }

  PartDictionary dict = ex.finalize();

  const bool binary = ends_with(out_path, ".bin");
  if (out_path == "-") {
    save_dict_text(dict, std::cout);
  } else if (binary) {
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout.is_open()) {
      std::cerr << "error: cannot open '" << out_path << "' for write\n";
      return 1;
    }
    save_dict_binary(dict, fout);
  } else {
    std::ofstream fout(out_path);
    if (!fout.is_open()) {
      std::cerr << "error: cannot open '" << out_path << "' for write\n";
      return 1;
    }
    save_dict_text(dict, fout);
  }

  std::cerr << "inputs: " << in_paths.size() << "\n";
  std::cerr << "total tokens: " << total_tokens
            << " (words=" << total_words
            << ", delims=" << total_delims << ")\n";
  std::cerr << "dictionary size F = " << dict.size()
            << " (saved as " << (binary ? "binary" : "text")
            << ", singletons/specials excluded from file)\n";
  std::cerr << "  whole:  " << dict.count_of_kind(Kind::Whole) << "\n";
  std::cerr << "  start:  " << dict.count_of_kind(Kind::Start) << "\n";
  std::cerr << "  mid:    " << dict.count_of_kind(Kind::Mid) << "\n";
  std::cerr << "  end:    " << dict.count_of_kind(Kind::End) << "\n";
  std::cerr << "  letter: " << dict.count_of_kind(Kind::Letter) << "\n";
  std::cerr << "  delim:  " << dict.count_of_kind(Kind::Delimiter) << "\n";

  return 0;
}
