#pragma once

// ============================================================================
// core/parts/dict_augment.hpp
//
// Append the standard set of singletons + special characters to a
// PartDictionary. Idempotent: relies on PartDictionary::add() returning
// the existing ID for duplicate (kind, value).
//
// Adds (in this order):
//   42  single-char Whole atoms      (a-z, 0-9, 6 connectors)
//   126 positional Letter atoms      (a-z, 0-9, 6 connectors × 3 positions)
//   6   connector Delimiter atoms    (-, ', &, ,, ., $)
//
// Connector set: -  '  &  ,  .  $
//
// Used by:
//   - load_dict_text() and load_dict_binary() — augment after reading
//     the trained-only file content
//   - PartExtractor::finalize() — append after the peel loop if you
//     want runtime dict layout to match loaded layout (optional)
// ============================================================================

#include "dictionary.hpp"

namespace core::parts {

void augment_with_atoms(PartDictionary& dict);

}  // namespace core::parts
