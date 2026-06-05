// ============================================================================
// apps/phase2_signatures/main.cpp
//
// Reads a corpus and a pre-trained part dictionary, walks the corpus
// sentence by sentence, decomposes each token using the dictionary,
// accumulates the asymmetric between-word co-occurrence matrix and the
// unconditional part-frequency vector, and writes both to disk.
//
// Usage:
//   phase2_signatures <corpus.txt> <input_dict.txt> <output_prefix>
//
// Writes:
//   <output_prefix>_C_between.tsv   row\tcol\tvalue   per non-zero entry
//   <output_prefix>_freqs.tsv       part_id\tcount    per part
//
// Stderr summary: dictionary size, token counts, sentence counts, nnz.
//
// The C_within matrix is intentionally NOT built here. Its builder exists
// (CoocWithinBuilder) but the current attention pipeline does not consume
// it, so phase 2 skips it to save time and disk on the typical run.
// ============================================================================

#include "binarytrain/parts/decomposer.hpp"
#include "binarytrain/parts/dictionary.hpp"
#include "binarytrain/parts/tokenize.hpp"
#include "binarytrain/signatures/cooc_between_builder.hpp"
#include "binarytrain/signatures/sentence_split.hpp"

#include "binarycore/sparse/sparse_matrix.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using binarycore::sparse::SparseMatrix;
using namespace binarytrain;

namespace {

void usage(const char* prog) {
  std::cerr << "usage: " << prog
            << " <corpus.txt|-> <input_dict.txt> <output_prefix>\n";
}

std::string slurp(std::istream& in) {
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Write a sparse matrix as TSV triplets: row\tcol\tvalue per non-zero.
// Rows iterated in order; within each row, columns are already sorted
// ascending by the CSR invariant.
bool write_matrix_tsv(const SparseMatrix<double>& m,
                      const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  for (std::size_t i = 0; i < m.rows(); ++i) {
    const auto r = m.row(i);
    for (std::size_t k = 0; k < r.nnz; ++k) {
      out << i << '\t' << r.indices[k] << '\t' << r.values[k] << '\n';
    }
  }
  return static_cast<bool>(out);
}

// Write frequency vector: part_id\tcount per part. Includes zero-count
// parts so the file has exactly F lines (the dictionary size).
bool write_freqs_tsv(const std::vector<std::uint64_t>& f,
                     const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) return false;
  for (std::size_t i = 0; i < f.size(); ++i) {
    out << i << '\t' << f[i] << '\n';
  }
  return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    usage(argv[0]);
    return 2;
  }
  const std::string in_path = argv[1];
  const std::string dict_path = argv[2];
  const std::string out_prefix = argv[3];

  const auto t_start = std::chrono::steady_clock::now();

  // --- Read corpus ---
  std::string raw;
  if (in_path == "-") {
    raw = slurp(std::cin);
  } else {
    std::ifstream fin(in_path);
    if (!fin.is_open()) {
      std::cerr << "error: cannot open corpus '" << in_path << "'\n";
      return 1;
    }
    raw = slurp(fin);
  }

  // --- Load dictionary ---
  parts::PartDictionary dict;
  {
    std::ifstream din(dict_path);
    if (!din.is_open()) {
      std::cerr << "error: cannot open dictionary '" << dict_path << "'\n";
      return 1;
    }
    dict = parts::PartDictionary::load(din);
  }
  dict.build_indices();
  const std::size_t F = dict.size();

  // --- Tokenize ---
  auto tokens = parts::tokenize_stream(raw);

  // --- Split into sentences ---
  auto sentence_ranges = signatures::split_sentences(tokens);

  // --- Decompose + accumulate ---
  signatures::CoocBetweenBuilder cb(F);  // default: include_delimiters=true

  std::size_t skipped_delims = 0;
  for (const auto& range : sentence_ranges) {
    std::vector<std::vector<std::uint32_t>> window;
    window.reserve(range.end - range.start);
    for (std::size_t i = range.start; i < range.end; ++i) {
      const auto& tok = tokens[i];
      if (tok.type == parts::StreamToken::Type::Word) {
        window.push_back(parts::decompose_word(dict, tok.value));
      } else {
        if (cb.config().include_delimiters) {
          window.push_back(parts::decompose_delimiter(dict, tok.value));
        } else {
          ++skipped_delims;
        }
      }
    }
    cb.add_window(window);
  }

  auto out = std::move(cb).finalize();

  const auto t_end = std::chrono::steady_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_end - t_start).count();

  // --- Write outputs ---
  const std::string mat_path = out_prefix + "_C_between.tsv";
  const std::string freq_path = out_prefix + "_freqs.tsv";

  if (!write_matrix_tsv(out.c_between, mat_path)) {
    std::cerr << "error: cannot write matrix to '" << mat_path << "'\n";
    return 1;
  }
  if (!write_freqs_tsv(out.frequencies, freq_path)) {
    std::cerr << "error: cannot write freqs to '" << freq_path << "'\n";
    return 1;
  }

  // --- Summary to stderr ---
  std::cerr << "corpus: " << in_path << "\n";
  std::cerr << "dictionary: " << dict_path << " (F = " << F << ")\n";
  std::cerr << "tokens: " << tokens.size() << "\n";
  std::cerr << "sentences: " << sentence_ranges.size() << "\n";
  std::cerr << "windows added: " << out.window_count << "\n";
  std::cerr << "total words+delims: " << out.total_word_count << "\n";
  std::cerr << "total part occurrences: " << out.total_part_count << "\n";
  if (skipped_delims) {
    std::cerr << "skipped delimiters: " << skipped_delims << "\n";
  }
  std::cerr << "C_between nnz: " << out.c_between.nnz()
            << " (matrix " << out.c_between.rows()
            << "x" << out.c_between.cols() << ")\n";
  std::cerr << "elapsed: " << elapsed_ms << " ms\n";
  std::cerr << "wrote: " << mat_path << "\n";
  std::cerr << "wrote: " << freq_path << "\n";

  return 0;
}
