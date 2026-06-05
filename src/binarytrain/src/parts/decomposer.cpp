// ============================================================================
// binarytrain/src/parts/decomposer.cpp
// ============================================================================

#include "binarytrain/parts/decomposer.hpp"

#include <algorithm>
#include <string>

namespace binarytrain::parts {

namespace {

// Internal: run the greedy claiming algorithm. Output: a `claimed`
// vector of length word.size() (true if covered by a non-singleton
// part), plus a list of `claims` describing what part covered which
// position range.
struct Claim {
  std::size_t start;
  std::size_t end;
  Kind kind;
  std::string value;
};

struct Layout {
  std::vector<Claim> claims;
  bool has_start_claim = false;
  bool has_end_claim = false;
};

Layout build_layout(const PartDictionary& dict, std::string_view word) {
  Layout out;
  const std::size_t n = word.size();
  if (n == 0) return out;

  // Step 1: Whole-word fast path.
  if (dict.has_whole(word)) {
    if (dict.lookup(Kind::Whole, word) != kInvalidPartId) {
      out.claims.push_back({0, n, Kind::Whole, std::string(word)});
      out.has_start_claim = true;
      out.has_end_claim = true;
      return out;
    }
  }

  // Step 2: Start phase.
  std::size_t start_consumed = 0;
  for (std::size_t L = 4; L >= 2; --L) {
    if (L > n) continue;
    std::string_view cand = word.substr(0, L);
    if (dict.has_start(cand) &&
        dict.lookup(Kind::Start, cand) != kInvalidPartId) {
      out.claims.push_back({0, L, Kind::Start, std::string(cand)});
      out.has_start_claim = true;
      start_consumed = L;
      break;
    }
  }

  // Step 3: End phase.
  std::size_t end_consumed = 0;
  for (std::size_t L = 4; L >= 2; --L) {
    if (L > n - start_consumed) continue;
    std::string_view cand = word.substr(n - L, L);
    if (dict.has_end(cand) &&
        dict.lookup(Kind::End, cand) != kInvalidPartId) {
      out.claims.push_back({n - L, n, Kind::End, std::string(cand)});
      out.has_end_claim = true;
      end_consumed = L;
      break;
    }
  }

  // Step 4: Mid phase. Scan left-to-right in the uncovered range.
  std::size_t i = start_consumed;
  const std::size_t mid_end = n - end_consumed;
  while (i < mid_end) {
    bool found = false;
    for (std::size_t L = 5; L >= 2; --L) {
      if (i + L > mid_end) continue;
      std::string_view cand = word.substr(i, L);
      if (dict.has_mid(cand) &&
          dict.lookup(Kind::Mid, cand) != kInvalidPartId) {
        out.claims.push_back({i, i + L, Kind::Mid, std::string(cand)});
        i += L;
        found = true;
        break;
      }
    }
    if (!found) ++i;
  }

  return out;
}

// Emit a Letter singleton part with the position-tagged value, using
// the WordPiece-style "##" continuation marker:
//   start of word (no preceding part)  → "x##"
//   end of word   (no following part)  → "##x"
//   middle of word                     → "##x##"
//   single-char word (both start+end)  → "x##" (defaults to start)
//
// "##" is collision-free for our character set since '#' isn't in
// any letter/digit/connector class.
std::uint32_t emit_letter(const PartDictionary& dict, char c,
                          bool at_start, bool at_end) {
  std::string s;
  if (at_end && !at_start) {
    s = "##" + std::string(1, c);
  } else if (!at_start && !at_end) {
    s = "##" + std::string(1, c) + "##";
  } else {
    // at_start (with or without at_end) — defaults to start encoding.
    s = std::string(1, c) + "##";
  }
  return dict.lookup(Kind::Letter, s);
}

}  // namespace

std::vector<std::uint32_t> decompose_word(
    const PartDictionary& dict,
    std::string_view word) {
  std::vector<std::uint32_t> out;
  if (word.empty()) return out;

  Layout layout = build_layout(dict, word);
  const std::size_t n = word.size();

  std::vector<bool> claimed(n, false);
  for (const auto& c : layout.claims) {
    for (std::size_t p = c.start; p < c.end; ++p) claimed[p] = true;
  }

  std::vector<std::pair<std::size_t, std::uint32_t>> emitted;
  for (const auto& c : layout.claims) {
    const std::uint32_t id = dict.lookup(c.kind, c.value);
    if (id != kInvalidPartId) emitted.emplace_back(c.start, id);
  }

  for (std::size_t p = 0; p < n; ++p) {
    if (claimed[p]) continue;
    const bool at_start = (p == 0) && !layout.has_start_claim;
    const bool at_end = (p == n - 1) && !layout.has_end_claim;
    const std::uint32_t id = emit_letter(dict, word[p], at_start, at_end);
    if (id != kInvalidPartId) emitted.emplace_back(p, id);
  }

  std::sort(emitted.begin(), emitted.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  out.reserve(emitted.size());
  for (const auto& [_, id] : emitted) out.push_back(id);
  return out;
}

std::vector<SingletonRun> find_singleton_runs(
    const PartDictionary& dict,
    std::string_view word) {
  std::vector<SingletonRun> runs;
  const std::size_t n = word.size();
  if (n == 0) return runs;

  Layout layout = build_layout(dict, word);

  std::vector<bool> claimed(n, false);
  for (const auto& c : layout.claims) {
    for (std::size_t p = c.start; p < c.end; ++p) claimed[p] = true;
  }

  std::size_t p = 0;
  while (p < n) {
    if (claimed[p]) { ++p; continue; }
    const std::size_t run_start = p;
    while (p < n && !claimed[p]) ++p;
    const std::size_t run_end = p;
    SingletonRun r;
    r.start = run_start;
    r.end = run_end;
    r.touches_word_start = (run_start == 0);
    r.touches_word_end   = (run_end == n);
    runs.push_back(r);
  }

  return runs;
}

std::vector<std::uint32_t> decompose_delimiter(
    const PartDictionary& dict,
    std::string_view delim) {
  std::vector<std::uint32_t> out;
  if (delim.empty()) return out;

  if (dict.has_delimiter(delim)) {
    const std::uint32_t id = dict.lookup(Kind::Delimiter, delim);
    if (id != kInvalidPartId) {
      out.push_back(id);
      return out;
    }
  }

  for (char c : delim) {
    const std::string s(1, c);
    if (dict.has_delimiter(s)) {
      const std::uint32_t id = dict.lookup(Kind::Delimiter, s);
      if (id != kInvalidPartId) out.push_back(id);
    }
  }
  return out;
}

}  // namespace binarytrain::parts
