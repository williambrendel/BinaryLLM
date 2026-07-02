#pragma once

// ============================================================================
// core/parts/part.hpp
//
// Core types for the part-based token vocabulary.
//
// A Part is one basis element in the F-dim part space. Each part has a
// position role (Kind) and a literal value (lowercase string for words,
// raw string for delimiters).
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>

namespace core::parts {

// Position role of a part. The same character sequence in different
// positions yields different parts — `-ing` and `ing-` and `-ing-` are
// three distinct basis elements with three distinct IDs.
enum class Kind : std::uint8_t {
  Start      = 0,  // anchored to word start: "un", "re", "trans"
  End        = 1,  // anchored to word end: "ing", "ed", "tion"
  Mid        = 2,  // appears in word interior: "port", "graph", "rate"
  Letter     = 3,  // positional letter singleton: "a##", "##a##", "##a"
                   // (the Letter kind is further specialized by its
                   // 1-character value plus a position via the start/
                   // end/mid pattern; encoded into the value field.
                   // See LetterPos below for the layout convention.)
  Whole      = 4,  // short whole-word atom: "the", "am", "I" (lowercased)
  Delimiter  = 5,  // punctuation or whitespace token, kept literally
};

// Length bounds for Start/Mid/End enumeration and matching.
//
// The extractor generates candidate parts at every L in
// [kMinPartLength, kMaxPartLength] during each peel iteration, and the
// decomposer searches the same range at match time. These MUST stay in
// sync — a part the extractor produces but the decomposer doesn't
// search for is dead weight in the dictionary (occupying an ID and
// inflating F without ever participating in a decomposition).
//
// Length 1 is reserved for the Letter singleton path (positional, "##"
// encoded) which serves as the universal coverage backstop.
inline constexpr std::size_t kMinPartLength = 2;
inline constexpr std::size_t kMaxPartLength = 7;

// Stringify Kind for serialization and debug.
inline std::string_view to_string(Kind k) {
  switch (k) {
    case Kind::Start:     return "start";
    case Kind::End:       return "end";
    case Kind::Mid:       return "mid";
    case Kind::Letter:    return "letter";
    case Kind::Whole:     return "whole";
    case Kind::Delimiter: return "delim";
  }
  return "?";
}

// Parse Kind from string. Returns true on success.
inline bool parse_kind(std::string_view s, Kind& out) {
  if (s == "start")     { out = Kind::Start;     return true; }
  if (s == "end")       { out = Kind::End;       return true; }
  if (s == "mid")       { out = Kind::Mid;       return true; }
  if (s == "letter")    { out = Kind::Letter;    return true; }
  if (s == "whole")     { out = Kind::Whole;     return true; }
  if (s == "delim")     { out = Kind::Delimiter; return true; }
  return false;
}

// Part identity = (kind, value).
struct PartKey {
  Kind kind;
  std::string value;

  bool operator==(const PartKey&) const = default;
};

// Hash for PartKey — combines kind and value hash.
struct PartKeyHash {
  std::size_t operator()(const PartKey& k) const noexcept {
    std::size_t h = std::hash<std::string>{}(k.value);
    // Mix in kind. The shift constants here are arbitrary mixing
    // designed to keep distinct kinds with the same value at distinct
    // hash buckets.
    return h ^ (std::size_t{static_cast<std::uint8_t>(k.kind)} * 0x9E3779B97F4A7C15ULL);
  }
};

// Sentinel ID for "not in dictionary."
inline constexpr std::uint32_t kInvalidPartId = static_cast<std::uint32_t>(-1);

}  // namespace core::parts
