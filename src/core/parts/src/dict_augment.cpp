// ============================================================================
// core/parts/src/dict_augment.cpp
// ============================================================================

#include "dict_augment.hpp"

#include <string>

namespace core::parts {

namespace {

const char* const kConnectors[] = {"-", "'", "&", ",", ".", "$"};

void add_letter_positions(PartDictionary& dict, const std::string& s) {
  dict.add(Kind::Letter, s + "##");
  dict.add(Kind::Letter, "##" + s + "##");
  dict.add(Kind::Letter, "##" + s);
}

}  // namespace

void augment_with_atoms(PartDictionary& dict) {
  // Single-char Whole atoms.
  for (char c = 'a'; c <= 'z'; ++c) {
    dict.add(Kind::Whole, std::string(1, c));
  }
  for (char c = '0'; c <= '9'; ++c) {
    dict.add(Kind::Whole, std::string(1, c));
  }
  for (const char* c : kConnectors) {
    dict.add(Kind::Whole, c);
  }

  // Positional Letter atoms.
  for (char c = 'a'; c <= 'z'; ++c) {
    add_letter_positions(dict, std::string(1, c));
  }
  for (char c = '0'; c <= '9'; ++c) {
    add_letter_positions(dict, std::string(1, c));
  }
  for (const char* c : kConnectors) {
    add_letter_positions(dict, std::string(c));
  }

  // Connector Delimiter atoms.
  for (const char* c : kConnectors) {
    dict.add(Kind::Delimiter, c);
  }
}

}  // namespace core::parts
