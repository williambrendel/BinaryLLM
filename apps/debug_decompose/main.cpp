// ============================================================================
// apps/debug_decompose/main.cpp
//
// Inspection tool. Given a dictionary and one or more sentences, prints:
//   1. The token stream (words + delimiters) per sentence.
//   2. Per-word decomposition as {value(kind)=id} lists, grouped
//      {{...},{...}} across the words of each sentence.
//   3. The part IDs for each part (shown inline in step 2).
//   4. The full 3F signature for --show=WORD words: set bits as global
//      indices, jumping across [before|current|after] bands at 0/F/2F.
//   5. (optional) --jaccard=WORD: pull WORD's 3F signature from each
//      sentence and report pairwise Jaccard similarity.
//
// Usage:
//   debug_decompose <dict> "<sentence>" [more positional sentences...]
//                   [--sentence="..."] [--show=WORD] [--jaccard=WORD]
//
// Sentences may be given positionally (after the dict) or via
// --sentence=. They accumulate. Each is encoded as its own scope.
// ============================================================================

#include "binary_vec.hpp"
#include "decomposer.hpp"
#include "dict_io.hpp"
#include "dictionary.hpp"
#include "encoder.hpp"
#include "tokenize.hpp"
#include "word_encoder.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using core::parts::Kind;
using core::parts::PartDictionary;
using core::parts::PartKey;
using core::parts::StreamToken;
using core::parts::decompose_word;
using core::parts::tokenize_stream;
using core::parts::ascii_lowercase;
using BVD = binarycore::binary_vec::BigSparseBinaryVecDynamic;

namespace {

constexpr std::size_t kChunkSize = BVD::chunk_size;

void usage(const char* prog) {
  std::cerr << "usage: " << prog
            << " <dict> \"<sentence>\" [more sentences...]"
            << " [--sentence=\"...\"] [--show=WORD] [--jaccard=WORD]\n";
}

bool ends_with(std::string_view s, std::string_view suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
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

std::string render_part(const PartKey& k) {
  switch (k.kind) {
    case Kind::Start:     return k.value + "##";
    case Kind::Mid:       return "##" + k.value + "##";
    case Kind::End:       return "##" + k.value;
    case Kind::Whole:     return k.value;
    case Kind::Letter:    return k.value;
    case Kind::Delimiter: return "[delim:" + k.value + "]";
  }
  return k.value;
}

std::vector<std::size_t> set_bits(const BVD& v) {
  std::vector<std::size_t> out;
  for (std::size_t k = 0; k < v.chunks.size(); ++k) {
    for (std::uint16_t local : v.chunks[k].data) {
      out.push_back(k * kChunkSize + local);
    }
  }
  return out;
}

// Find the sig index of the first Word in `tokens` equal (lowercased)
// to `target`. Returns sigs.size() if not found. `sigs` is parallel to
// the Word tokens of `tokens` in stream order.
std::size_t find_word_sig(const std::vector<StreamToken>& tokens,
                          const std::string& target_lower,
                          std::size_t sig_count) {
  std::size_t w = 0;
  for (const auto& t : tokens) {
    if (t.type != StreamToken::Type::Word) continue;
    if (ascii_lowercase(t.value) == target_lower) return w;
    ++w;
  }
  return sig_count;  // not found
}

void print_token_stream(const std::vector<StreamToken>& tokens) {
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto& t = tokens[i];
    const bool is_word = (t.type == StreamToken::Type::Word);
    std::cout << "  [" << i << "] " << (is_word ? "word " : "delim")
              << "  \"" << t.value << "\"\n";
  }
}

void print_decomposition(const PartDictionary& dict,
                         const std::vector<StreamToken>& tokens) {
  std::cout << "{";
  bool first = true;
  for (const auto& t : tokens) {
    if (t.type != StreamToken::Type::Word) continue;
    if (!first) std::cout << ",\n ";
    first = false;
    auto ids = decompose_word(dict, ascii_lowercase(t.value));
    std::cout << "{";
    for (std::size_t j = 0; j < ids.size(); ++j) {
      if (j > 0) std::cout << ", ";
      std::cout << render_part(dict.at(ids[j])) << "=" << ids[j];
    }
    std::cout << "}  // \"" << t.value << "\"";
  }
  std::cout << "}\n";
}

void print_3f(const PartDictionary& dict, std::size_t F, const BVD& sig) {
  const auto bits = set_bits(sig);
  std::cout << "    popcount = " << bits.size() << "\n    [";
  bool prev_current = false, prev_after = false;
  for (std::size_t j = 0; j < bits.size(); ++j) {
    const std::size_t b = bits[j];
    const bool in_current = (b >= F && b < 2 * F);
    const bool in_after   = (b >= 2 * F);
    if (j > 0) {
      if ((in_current && !prev_current) || (in_after && !prev_after))
        std::cout << "  |  ";
      else
        std::cout << ", ";
    }
    std::cout << b;
    prev_current = in_current;
    prev_after = in_after;
  }
  std::cout << "]\n      current band (local id = global - F):\n";
  for (std::size_t b : bits) {
    if (b >= F && b < 2 * F) {
      const std::uint32_t id = static_cast<std::uint32_t>(b - F);
      std::cout << "        " << b << " -> id " << id
                << "  " << render_part(dict.at(id)) << "\n";
    }
  }
}

// Jaccard over two 3F vectors via the binary_vec helper.
float jaccard_3f(const BVD& a, const BVD& b) {
  return binarycore::binary_vec::jaccard(a, b);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { usage(argv[0]); return 2; }

  const std::string dict_path = argv[1];
  std::vector<std::string> sentences;
  std::vector<std::string> show_words;
  std::vector<std::string> jaccard_words;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--sentence=", 0) == 0)      sentences.push_back(a.substr(11));
    else if (a.rfind("--show=", 0) == 0)     show_words.push_back(a.substr(7));
    else if (a.rfind("--jaccard=", 0) == 0)  jaccard_words.push_back(a.substr(10));
    else if (a.rfind("--", 0) == 0) {
      std::cerr << "unknown arg: " << a << "\n"; usage(argv[0]); return 2;
    } else {
      sentences.push_back(a);  // positional sentence
    }
  }
  if (sentences.empty()) { usage(argv[0]); return 2; }

  auto dict = load_dict(dict_path);
  const std::size_t F = dict.size();
  std::cout << "dictionary: " << dict_path << "  (F = " << F
            << ", 3F = " << (3 * F) << ")\n";

  // Tokenize + encode each sentence as its own scope.
  std::vector<std::vector<StreamToken>> toks(sentences.size());
  std::vector<std::vector<BVD>> sigs(sentences.size());
  for (std::size_t s = 0; s < sentences.size(); ++s) {
    toks[s] = tokenize_stream(sentences[s]);
    sigs[s] = core::signatures::encode(dict, toks[s]);
  }

  // Per-sentence report.
  for (std::size_t s = 0; s < sentences.size(); ++s) {
    std::cout << "\n############ sentence[" << s << "]: \""
              << sentences[s] << "\" ############\n";
    std::cout << "\n=== 1. token stream ===\n";
    print_token_stream(toks[s]);
    std::cout << "\n=== 2/3. decomposition  {value(kind)=id} ===\n";
    print_decomposition(dict, toks[s]);

    if (!show_words.empty()) {
      std::cout << "\n=== 4. 3F signatures (before=[0,F) current=[F,2F) after=[2F,3F)) ===\n";
      for (const auto& target : show_words) {
        const std::string tl = ascii_lowercase(target);
        const std::size_t w = find_word_sig(toks[s], tl, sigs[s].size());
        std::cout << "\n  \"" << target << "\":\n";
        if (w >= sigs[s].size()) { std::cout << "    (not in this sentence)\n"; continue; }
        print_3f(dict, F, sigs[s][w]);
      }
    }
  }

  // Cross-sentence Jaccard.
  if (!jaccard_words.empty() && sentences.size() >= 2) {
    std::cout << "\n############ Jaccard (cross-sentence 3F) ############\n";
    for (const auto& target : jaccard_words) {
      const std::string tl = ascii_lowercase(target);
      std::cout << "\n  word \"" << target << "\":\n";
      // Gather (sentence index, sig) for every sentence containing it.
      std::vector<std::pair<std::size_t, const BVD*>> hits;
      for (std::size_t s = 0; s < sentences.size(); ++s) {
        const std::size_t w = find_word_sig(toks[s], tl, sigs[s].size());
        if (w < sigs[s].size()) hits.emplace_back(s, &sigs[s][w]);
      }
      if (hits.size() < 2) {
        std::cout << "    found in " << hits.size()
                  << " sentence(s); need >= 2 to compare\n";
        continue;
      }
      for (std::size_t a = 0; a < hits.size(); ++a) {
        for (std::size_t b = a + 1; b < hits.size(); ++b) {
          const float j = jaccard_3f(*hits[a].second, *hits[b].second);
          std::cout << "    s[" << hits[a].first << "] vs s["
                    << hits[b].first << "]  jaccard = " << j << "\n";
        }
      }
    }
  } else if (!jaccard_words.empty()) {
    std::cout << "\n(note: --jaccard needs >= 2 sentences)\n";
  }

  return 0;
}
