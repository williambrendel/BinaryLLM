// ============================================================================
// core/parts/src/dictionary.cpp
// ============================================================================

#include "dictionary.hpp"

#include <array>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace core::parts {

namespace {

// ----- Text format helpers ----- //

std::string escape_value(std::string_view v) {
  std::string out;
  out.reserve(v.size() + 4);
  for (char c : v) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\t': out += "\\t";  break;
      case '\r': out += "\\r";  break;
      default:   out.push_back(c);
    }
  }
  return out;
}

std::string unescape_value(std::string_view v) {
  std::string out;
  out.reserve(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] == '\\' && i + 1 < v.size()) {
      switch (v[i + 1]) {
        case '\\': out.push_back('\\'); ++i; break;
        case 'n':  out.push_back('\n'); ++i; break;
        case 't':  out.push_back('\t'); ++i; break;
        case 'r':  out.push_back('\r'); ++i; break;
        default:   out.push_back(v[i]);
      }
    } else {
      out.push_back(v[i]);
    }
  }
  return out;
}

// ----- Binary format helpers ----- //
//
// File layout (all integers little-endian):
//
//   HEADER (12 bytes):
//     magic[4]     = "BDIC"
//     version u16  = kBinaryVersion (1)
//     flags u16    = 0 (reserved)
//     num_kinds u32 = 6
//
//   PER KIND (repeated num_kinds times, in fixed order
//             Whole, Start, Mid, End, Letter, Delimiter):
//     kind u8      (1..6, see kind_to_byte)
//     pad[3]       = 0 (alignment)
//     count u32    = atoms in this kind
//     REPEAT count atoms:
//       length u16
//       bytes[length]   (raw atom string, no terminator)
//
// Atoms within a kind are written in original ID order. On load, IDs
// are reassigned in the order atoms appear in the stream — matching
// the behavior of the text format.

constexpr std::uint16_t kBinaryVersion = 1;
constexpr std::uint32_t kNumKinds = 6;

std::uint8_t kind_to_byte(Kind k) {
  switch (k) {
    case Kind::Whole:     return 1;
    case Kind::Start:     return 2;
    case Kind::Mid:       return 3;
    case Kind::End:       return 4;
    case Kind::Letter:    return 5;
    case Kind::Delimiter: return 6;
  }
  return 0;
}

bool byte_to_kind(std::uint8_t b, Kind& out) {
  switch (b) {
    case 1: out = Kind::Whole;     return true;
    case 2: out = Kind::Start;     return true;
    case 3: out = Kind::Mid;       return true;
    case 4: out = Kind::End;       return true;
    case 5: out = Kind::Letter;    return true;
    case 6: out = Kind::Delimiter; return true;
    default: return false;
  }
}

void write_u16_le(std::ostream& out, std::uint16_t v) {
  unsigned char b[2] = {
    static_cast<unsigned char>(v & 0xFFu),
    static_cast<unsigned char>((v >> 8) & 0xFFu)
  };
  out.write(reinterpret_cast<const char*>(b), 2);
}

void write_u32_le(std::ostream& out, std::uint32_t v) {
  unsigned char b[4] = {
    static_cast<unsigned char>(v & 0xFFu),
    static_cast<unsigned char>((v >> 8) & 0xFFu),
    static_cast<unsigned char>((v >> 16) & 0xFFu),
    static_cast<unsigned char>((v >> 24) & 0xFFu)
  };
  out.write(reinterpret_cast<const char*>(b), 4);
}

std::uint16_t read_u16_le(std::istream& in) {
  unsigned char b[2];
  in.read(reinterpret_cast<char*>(b), 2);
  if (in.gcount() != 2) {
    throw std::runtime_error("PartDictionary::load_binary: truncated u16");
  }
  return static_cast<std::uint16_t>(b[0])
       | (static_cast<std::uint16_t>(b[1]) << 8);
}

std::uint32_t read_u32_le(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  if (in.gcount() != 4) {
    throw std::runtime_error("PartDictionary::load_binary: truncated u32");
  }
  return static_cast<std::uint32_t>(b[0])
       | (static_cast<std::uint32_t>(b[1]) << 8)
       | (static_cast<std::uint32_t>(b[2]) << 16)
       | (static_cast<std::uint32_t>(b[3]) << 24);
}

}  // namespace

// ============================================================================
// PartDictionary: add / lookup / counts
// ============================================================================

std::uint32_t PartDictionary::add(Kind kind, std::string value) {
  PartKey key{kind, std::move(value)};
  if (auto it = key_to_id_.find(key); it != key_to_id_.end()) {
    return it->second;
  }
  const std::uint32_t id = static_cast<std::uint32_t>(id_to_key_.size());
  id_to_key_.push_back(key);
  key_to_id_.emplace(std::move(key), id);
  indices_built_ = false;
  return id;
}

std::uint32_t PartDictionary::lookup(
    Kind kind, std::string_view value) const noexcept {
  PartKey key{kind, std::string(value)};
  auto it = key_to_id_.find(key);
  if (it == key_to_id_.end()) return kInvalidPartId;
  return it->second;
}

std::size_t PartDictionary::count_of_kind(Kind kind) const noexcept {
  std::size_t n = 0;
  for (const auto& k : id_to_key_) {
    if (k.kind == kind) ++n;
  }
  return n;
}

// ============================================================================
// Text save / load
// ============================================================================

void PartDictionary::save(std::ostream& out) const {
  for (std::size_t i = 0; i < id_to_key_.size(); ++i) {
    const auto& k = id_to_key_[i];
    out << i << '\t' << to_string(k.kind) << '\t'
        << escape_value(k.value) << '\n';
  }
}

PartDictionary PartDictionary::load(std::istream& in) {
  PartDictionary dict;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto t1 = line.find('\t');
    if (t1 == std::string::npos) continue;
    auto t2 = line.find('\t', t1 + 1);
    if (t2 == std::string::npos) continue;

    std::string_view id_str(line.data(), t1);
    std::string_view kind_str(line.data() + t1 + 1, t2 - t1 - 1);
    std::string_view val_str(line.data() + t2 + 1, line.size() - t2 - 1);

    Kind kind;
    if (!parse_kind(kind_str, kind)) {
      throw std::runtime_error("PartDictionary::load: unknown kind '" +
                               std::string(kind_str) + "'");
    }

    std::string value = unescape_value(val_str);

    (void)id_str;

    dict.add(kind, std::move(value));
  }
  return dict;
}

// ============================================================================
// Binary save / load
// ============================================================================

void PartDictionary::save_binary(std::ostream& out) const {
  // Header.
  out.write("BDIC", 4);
  write_u16_le(out, kBinaryVersion);
  write_u16_le(out, 0);                 // flags (reserved)
  write_u32_le(out, kNumKinds);

  // Group atoms by kind, preserving ID order within each kind.
  std::array<std::vector<const std::string*>, kNumKinds> by_kind;
  for (const auto& k : id_to_key_) {
    const std::uint8_t b = kind_to_byte(k.kind);
    if (b >= 1 && b <= kNumKinds) {
      by_kind[b - 1].push_back(&k.value);
    }
  }

  // Write each kind block in fixed order (1..6).
  for (std::uint8_t kb = 1; kb <= kNumKinds; ++kb) {
    const char kind_tag = static_cast<char>(kb);
    out.write(&kind_tag, 1);
    const char zeros[3] = {0, 0, 0};
    out.write(zeros, 3);                // padding

    const auto& atoms = by_kind[kb - 1];
    write_u32_le(out, static_cast<std::uint32_t>(atoms.size()));

    for (const std::string* val : atoms) {
      const std::size_t len = val->size();
      if (len > 0xFFFFu) {
        // u16 length cap. Atoms above 65535 bytes shouldn't exist in
        // practice — whole_max_len is 16, Letter atoms are ≤5 chars
        // with the "##" markers, observed delimiters are ≤2 bytes.
        // Throwing prevents silent corruption.
        throw std::runtime_error(
            "PartDictionary::save_binary: atom length exceeds u16");
      }
      write_u16_le(out, static_cast<std::uint16_t>(len));
      out.write(val->data(), static_cast<std::streamsize>(len));
    }
  }
}

PartDictionary PartDictionary::load_binary(std::istream& in) {
  PartDictionary dict;

  // Header.
  char magic[4];
  in.read(magic, 4);
  if (in.gcount() != 4 ||
      magic[0] != 'B' || magic[1] != 'D' ||
      magic[2] != 'I' || magic[3] != 'C') {
    throw std::runtime_error("PartDictionary::load_binary: bad magic");
  }

  const std::uint16_t version = read_u16_le(in);
  if (version != kBinaryVersion) {
    throw std::runtime_error(
        "PartDictionary::load_binary: unsupported version " +
        std::to_string(version));
  }

  (void)read_u16_le(in);  // flags — reserved, ignored

  const std::uint32_t num_kinds = read_u32_le(in);
  if (num_kinds != kNumKinds) {
    throw std::runtime_error(
        "PartDictionary::load_binary: expected " +
        std::to_string(kNumKinds) + " kinds, got " +
        std::to_string(num_kinds));
  }

  for (std::uint32_t k = 0; k < num_kinds; ++k) {
    unsigned char kind_hdr[4];
    in.read(reinterpret_cast<char*>(kind_hdr), 4);
    if (in.gcount() != 4) {
      throw std::runtime_error(
          "PartDictionary::load_binary: truncated kind header");
    }
    Kind kind;
    if (!byte_to_kind(kind_hdr[0], kind)) {
      throw std::runtime_error(
          "PartDictionary::load_binary: unknown kind byte " +
          std::to_string(kind_hdr[0]));
    }
    // kind_hdr[1..3] are padding; ignored.

    const std::uint32_t count = read_u32_le(in);
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint16_t len = read_u16_le(in);
      std::string value(len, '\0');
      if (len > 0) {
        in.read(value.data(), len);
        if (in.gcount() != static_cast<std::streamsize>(len)) {
          throw std::runtime_error(
              "PartDictionary::load_binary: truncated atom bytes");
        }
      }
      dict.add(kind, std::move(value));
    }
  }

  return dict;
}

// ============================================================================
// Lookup indices
// ============================================================================

void PartDictionary::build_indices() const {
  ensure_indices_();
}

void PartDictionary::ensure_indices_() const {
  if (indices_built_) return;
  starts_set_.clear();
  ends_set_.clear();
  mids_set_.clear();
  wholes_set_.clear();
  delims_set_.clear();
  for (const auto& k : id_to_key_) {
    switch (k.kind) {
      case Kind::Start:     starts_set_[k.value] = true; break;
      case Kind::End:       ends_set_[k.value]   = true; break;
      case Kind::Mid:       mids_set_[k.value]   = true; break;
      case Kind::Whole:     wholes_set_[k.value] = true; break;
      case Kind::Delimiter: delims_set_[k.value] = true; break;
      case Kind::Letter:    break;
    }
  }
  indices_built_ = true;
}

bool PartDictionary::has_start(std::string_view value) const {
  ensure_indices_();
  return starts_set_.find(std::string(value)) != starts_set_.end();
}
bool PartDictionary::has_end(std::string_view value) const {
  ensure_indices_();
  return ends_set_.find(std::string(value)) != ends_set_.end();
}
bool PartDictionary::has_mid(std::string_view value) const {
  ensure_indices_();
  return mids_set_.find(std::string(value)) != mids_set_.end();
}
bool PartDictionary::has_whole(std::string_view value) const {
  ensure_indices_();
  return wholes_set_.find(std::string(value)) != wholes_set_.end();
}
bool PartDictionary::has_delimiter(std::string_view value) const {
  ensure_indices_();
  return delims_set_.find(std::string(value)) != delims_set_.end();
}

}  // namespace core::parts
