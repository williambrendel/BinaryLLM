#pragma once

// ============================================================================
// binarytrain/parts/dictionary.hpp
//
// PartDictionary stores the trained part vocabulary. Built once by
// PartExtractor, then queried (read-only) by the decomposer at
// inference time.
//
// Storage:
//   - id_to_key:  vector<PartKey>          — index by ID
//   - key_to_id:  unordered_map<PartKey, uint32_t>
//
// Plus per-kind length-indexed lookup tables (built lazily on first
// query) for greedy longest-match decomposition.
//
// Serialization:
//   - save() / load(): text format, one part per line:
//       <id>\t<kind>\t<value>\n
//     with `value` escaped for special characters.
//   - save_binary() / load_binary(): compact 12-byte header + per-
//     kind atom blocks. See dictionary.cpp for the byte layout.
//   Both formats reassign IDs in insertion order on load; the saved
//   IDs are informational only.
// ============================================================================

#include "binarytrain/parts/part.hpp"

#include <istream>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace binarytrain::parts {

class PartDictionary {
public:
  PartDictionary() = default;

  std::uint32_t add(Kind kind, std::string value);

  std::uint32_t lookup(Kind kind, std::string_view value) const noexcept;

  const PartKey& at(std::uint32_t id) const { return id_to_key_[id]; }

  std::size_t size() const noexcept { return id_to_key_.size(); }

  std::size_t count_of_kind(Kind kind) const noexcept;

  const std::vector<PartKey>& all_parts() const noexcept {
    return id_to_key_;
  }

  // Text serialization: human-readable, one part per line.
  // Format: id\tkind\tvalue\n   (value is escape-encoded).
  void save(std::ostream& out) const;
  static PartDictionary load(std::istream& in);

  // Binary serialization: compact 12-byte header ("BDIC" + version +
  // flags + num_kinds) followed by per-kind blocks. Atoms within each
  // kind are written in their original ID order. On load, IDs are
  // reassigned in the order atoms appear in the file (consistent with
  // the text format's behavior).
  //
  // The stream MUST be opened in binary mode (e.g. std::ofstream with
  // std::ios::binary) on platforms that do newline translation
  // (Windows).
  void save_binary(std::ostream& out) const;
  static PartDictionary load_binary(std::istream& in);

  void build_indices() const;

  bool has_start(std::string_view value) const;
  bool has_end(std::string_view value) const;
  bool has_mid(std::string_view value) const;
  bool has_whole(std::string_view value) const;
  bool has_delimiter(std::string_view value) const;

private:
  std::vector<PartKey> id_to_key_;
  std::unordered_map<PartKey, std::uint32_t, PartKeyHash> key_to_id_;

  mutable bool indices_built_ = false;
  mutable std::unordered_map<std::string, bool> starts_set_;
  mutable std::unordered_map<std::string, bool> ends_set_;
  mutable std::unordered_map<std::string, bool> mids_set_;
  mutable std::unordered_map<std::string, bool> wholes_set_;
  mutable std::unordered_map<std::string, bool> delims_set_;

  void ensure_indices_() const;
};

}  // namespace binarytrain::parts
