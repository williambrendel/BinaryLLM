// ============================================================================
// apps/build_candidates/main.cpp
//
// Build the binary 3F candidate file from a dictionary + corpora.
//
// For each input file, tokenize once, then encode candidates at BOTH
// segmentation scales and concatenate them into one stream:
//
//     stream = (paragraph-scale candidates) ++ (sentence-scale candidates)
//
//   - paragraph scale: encode() over each paragraph's token span;
//                      anchors see the whole paragraph as context.
//   - sentence scale:  encode() over each sentence's token span;
//                      anchors see only their own sentence.
//
// Each candidate is a flat 3F BinaryVecDynamic (before|current|after,
// global-indexed). The file stores them in stream order. Multi-scale
// teaches the clustering stage co-occurrence at both granularities from
// the same corpus; the clusterer consumes them as one stream.
//
// Two passes: pass 1 counts candidates (exact record_count for the
// header), pass 2 encodes + writes. Corpora are re-read from disk in
// pass 2 (encode dominates cost, re-tokenizing is cheap).
//
// Usage:
//   build_candidates <dict> <out.cnd3> <corpus1> [corpus2 ...]
// ============================================================================

#include "binary_vec.hpp"
#include "candidate_file.hpp"
#include "dict_io.hpp"
#include "dictionary.hpp"
#include "encoder.hpp"
#include "paragraph_split.hpp"
#include "sentence_split.hpp"
#include "tokenize.hpp"

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
using core::signatures::split_paragraphs;
using core::signatures::split_sentences;

namespace {

void usage(const char* p) {
  std::cerr << "usage: " << p << " <dict> <out.cnd3> <corpus1> [corpus2 ...]\n";
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

std::size_t count_words(const std::vector<StreamToken>& tokens,
                        std::size_t start, std::size_t end) {
  std::size_t n = 0;
  for (std::size_t i = start; i < end && i < tokens.size(); ++i)
    if (tokens[i].type == StreamToken::Type::Word) ++n;
  return n;
}

// Multi-scale candidate count: one per Word per paragraph span +
// one per Word per sentence span = 2 * (Word token count).
std::size_t count_candidates(const std::vector<StreamToken>& tokens) {
  std::size_t total = 0;
  for (const auto& pr : split_paragraphs(tokens))
    total += count_words(tokens, pr.start, pr.end);
  for (const auto& sr : split_sentences(tokens))
    total += count_words(tokens, sr.start, sr.end);
  return total;
}

void emit_candidates(const PartDictionary& dict,
                     const std::vector<StreamToken>& tokens,
                     core::signatures::CandidateFileWriter& w) {
  for (const auto& pr : split_paragraphs(tokens)) {
    auto sigs = encode(dict, tokens, pr.start, pr.end);
    for (const auto& s : sigs) w.write(s);
  }
  for (const auto& sr : split_sentences(tokens)) {
    auto sigs = encode(dict, tokens, sr.start, sr.end);
    for (const auto& s : sigs) w.write(s);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) { usage(argv[0]); return 2; }

  const std::string dict_path = argv[1];
  const std::string out_path = argv[2];
  std::vector<std::string> corpora;
  for (int i = 3; i < argc; ++i) corpora.emplace_back(argv[i]);

  auto dict = load_dict(dict_path);
  const std::uint32_t F = static_cast<std::uint32_t>(dict.size());
  const std::uint32_t dim = 3 * F;
  std::cerr << "dict F = " << F << "  (3F = " << dim << ")\n";

  std::uint64_t total = 0;
  for (const auto& path : corpora) {
    const auto tokens = tokenize_stream(slurp(path));
    const std::size_t n = count_candidates(tokens);
    std::cerr << "  " << path << ": " << n << " candidates (multi-scale)\n";
    total += n;
  }
  std::cerr << "total candidates: " << total << "\n";

  std::ofstream out(out_path, std::ios::binary);
  if (!out.is_open()) { std::cerr << "cannot open " << out_path << "\n"; return 1; }
  core::signatures::CandidateFileWriter writer(out, dim, total);

  for (const auto& path : corpora) {
    const auto tokens = tokenize_stream(slurp(path));
    emit_candidates(dict, tokens, writer);
  }

  std::cerr << "wrote " << writer.written() << " candidates to " << out_path << "\n";
  if (writer.written() != total) {
    std::cerr << "WARNING: written != counted (" << writer.written()
              << " != " << total << ")\n";
    return 1;
  }
  return 0;
}
