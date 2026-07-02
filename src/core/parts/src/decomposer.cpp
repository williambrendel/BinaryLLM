// ============================================================================
// core/parts/src/decomposer.cpp
//
// Single peel algorithm (Whole fast path + dict-ID-order peel with
// multi-Mid + letter singleton fill). Implementation uses a candidate
// enumeration to avoid the O(F) per-word scan: for each (L, pos) in
// the word we ask the dict whether the substring is registered,
// rather than asking every part whether it matches the word.
//
// O(n * kMaxPartLength) hash lookups per word, then sort, then linear
// peel. For n=10 and kMax=7 that's ~50 lookups.
// ============================================================================

#include "decomposer.hpp"

#include <algorithm>
#include <string>
#include <utility>

namespace core::parts {

namespace {

// ---- Letter singleton helpers (## position marker) ----------------

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

// ---- Range helpers -------------------------------------------------

bool range_is_free(const std::vector<bool>& claimed,
                   std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    if (claimed[i]) return false;
  }
  return true;
}

void claim_range(std::vector<bool>& claimed,
                 std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    claimed[i] = true;
  }
}

// ---- Candidate enumeration ----------------------------------------

struct Candidate {
  std::size_t pos;
  std::size_t L;
  Kind kind;
  std::uint32_t id;
};

// For every L in [kMin, kMax] and every valid position, ask the dict
// whether the substring is registered as Start/End/Mid. Collect hits.
std::vector<Candidate> enumerate_candidates(
    const PartDictionary& dict, std::string_view word) {
  const std::size_t n = word.size();
  std::vector<Candidate> out;
  out.reserve(32);

  for (std::size_t L = kMinPartLength; L <= kMaxPartLength; ++L) {
    if (L > n) break;

    // Start at pos 0.
    {
      const std::string_view sv = word.substr(0, L);
      const std::uint32_t id = dict.lookup(Kind::Start, sv);
      if (id != kInvalidPartId) out.push_back({0, L, Kind::Start, id});
    }
    // End at pos n-L.
    {
      const std::string_view sv = word.substr(n - L, L);
      const std::uint32_t id = dict.lookup(Kind::End, sv);
      if (id != kInvalidPartId) out.push_back({n - L, L, Kind::End, id});
    }
    // Mid at every valid position.
    for (std::size_t pos = 0; pos + L <= n; ++pos) {
      const std::string_view sv = word.substr(pos, L);
      const std::uint32_t id = dict.lookup(Kind::Mid, sv);
      if (id != kInvalidPartId) out.push_back({pos, L, Kind::Mid, id});
    }
  }
  return out;
}

// ---- Greedy peel ---------------------------------------------------

// Apply the peel algorithm to a pre-sorted candidate list.
// Output: the (pos, id) pairs that fired, plus the final claimed
// bitmap and the has_start/has_end flags (needed for letter fill).
struct PeelResult {
  std::vector<std::pair<std::size_t, std::uint32_t>> emitted;
  std::vector<bool> claimed;
  bool has_start_claim = false;
  bool has_end_claim = false;
};

PeelResult peel(std::size_t n,
                const std::vector<Candidate>& candidates) {
  PeelResult r;
  r.claimed.assign(n, false);
  r.emitted.reserve(candidates.size());

  for (const Candidate& c : candidates) {
    if (c.kind == Kind::Start && r.has_start_claim) continue;
    if (c.kind == Kind::End && r.has_end_claim) continue;
    if (!range_is_free(r.claimed, c.pos, c.pos + c.L)) continue;

    claim_range(r.claimed, c.pos, c.pos + c.L);
    r.emitted.emplace_back(c.pos, c.id);
    if (c.kind == Kind::Start) r.has_start_claim = true;
    if (c.kind == Kind::End) r.has_end_claim = true;
  }
  return r;
}

// Sort candidates by (id asc, pos asc) — dict ID order = the dict's
// insertion order, which is whatever the dict's user shaped it to be
// (either the extractor's original order, or a strategy-reordered
// dict from dict_reorder.hpp).
void sort_candidates_by_id(std::vector<Candidate>& cands) {
  std::sort(cands.begin(), cands.end(),
            [](const Candidate& a, const Candidate& b) {
              if (a.id != b.id) return a.id < b.id;
              return a.pos < b.pos;
            });
}

}  // namespace

// ---- Public API ---------------------------------------------------

std::vector<std::uint32_t> decompose_word(
    const PartDictionary& dict,
    std::string_view word) {
  std::vector<std::uint32_t> out;
  const std::size_t n = word.size();
  if (n == 0) return out;

  // Whole-word fast path.
  if (dict.has_whole(word)) {
    const std::uint32_t id = dict.lookup(Kind::Whole, word);
    if (id != kInvalidPartId) {
      out.push_back(id);
      return out;
    }
  }

  auto cands = enumerate_candidates(dict, word);
  sort_candidates_by_id(cands);
  auto pr = peel(n, cands);

  // Letter singleton fill.
  for (std::size_t p = 0; p < n; ++p) {
    if (pr.claimed[p]) continue;
    const bool at_start = (p == 0) && !pr.has_start_claim;
    const bool at_end = (p == n - 1) && !pr.has_end_claim;
    const std::uint32_t id = emit_letter(dict, word[p], at_start, at_end);
    if (id != kInvalidPartId) pr.emitted.emplace_back(p, id);
  }

  std::sort(pr.emitted.begin(), pr.emitted.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  out.reserve(pr.emitted.size());
  for (const auto& [_, id] : pr.emitted) out.push_back(id);
  return out;
}

std::vector<SingletonRun> find_singleton_runs(
    const PartDictionary& dict,
    std::string_view word) {
  std::vector<SingletonRun> runs;
  const std::size_t n = word.size();
  if (n == 0) return runs;

  // If the word is a Whole atom, the entire word is "covered" — no
  // singleton runs.
  if (dict.has_whole(word)) {
    const std::uint32_t id = dict.lookup(Kind::Whole, word);
    if (id != kInvalidPartId) return runs;
  }

  auto cands = enumerate_candidates(dict, word);
  sort_candidates_by_id(cands);
  auto pr = peel(n, cands);

  std::size_t p = 0;
  while (p < n) {
    if (pr.claimed[p]) { ++p; continue; }
    const std::size_t run_start = p;
    while (p < n && !pr.claimed[p]) ++p;
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

}  // namespace core::parts
