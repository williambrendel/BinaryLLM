#pragma once

// ============================================================================
// core/parts/dict_reorder.hpp
//
// Strategy side functions: produce a NEW PartDictionary containing the
// same parts as the source, but with Start/Mid/End atoms in a
// different insertion order. Since the decomposer iterates parts in
// dict ID order, reordering the dict changes which parts get peel
// priority — without touching the decomposer itself.
//
// Non-matchable atoms (Whole, Letter, Delimiter) are copied first in
// their original order; only Start/Mid/End are reshuffled.
//
// Available strategies:
//
//   reorder_dict_by_type
//     Sort key (kind[S<E<M] asc, original_id asc).
//     "All Starts first (in original order), then all Ends, then all
//     Mids."
//
//   reorder_dict_by_type_then_level
//     Sort key (kind[S<E<M] asc, L desc, original_id asc).
//     "All Starts longest-first, then all Ends longest-first, then
//     all Mids longest-first."
//
//   reorder_dict_by_level_then_type
//     Sort key (L desc, kind[S<E<M] asc, original_id asc).
//     "L=kMax: Start, End, Mid; then L=kMax-1: Start, End, Mid; ..."
//
// The "discovery" / "original" strategy is just the source dict
// (no reordering function call needed).
// ============================================================================

#include "dictionary.hpp"

namespace core::parts {

PartDictionary reorder_dict_by_type(const PartDictionary& src);
PartDictionary reorder_dict_by_type_then_level(const PartDictionary& src);
PartDictionary reorder_dict_by_level_then_type(const PartDictionary& src);

}  // namespace core::parts
