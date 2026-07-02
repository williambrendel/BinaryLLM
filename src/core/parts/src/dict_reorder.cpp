// ============================================================================
// core/parts/src/dict_reorder.cpp
// ============================================================================

#include "dict_reorder.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace core::parts {

namespace {

// Priority for "Start before End before Mid" orderings.
constexpr int kind_priority(Kind k) {
  switch (k) {
    case Kind::Start: return 0;
    case Kind::End:   return 1;
    case Kind::Mid:   return 2;
    default:          return 99;
  }
}

// Reorder item: a copy of one S/M/E atom with its original ID
// retained as a tie-breaker.
struct Item {
  std::uint32_t original_id;
  Kind kind;
  std::string value;
};

// Common shape of all reorderings: copy non-matchable atoms (Whole,
// Letter, Delimiter) in original order, then add Start/Mid/End atoms
// in the strategy-specific order.
template <class Cmp>
PartDictionary reorder_impl(const PartDictionary& src, Cmp cmp) {
  PartDictionary dst;

  // Phase 1: copy non-matchable atoms first, in original order. Their
  // exact ID positions don't affect the decomposer (it skips them in
  // the iteration loop), but we keep them clustered at the front for
  // tidiness.
  for (std::uint32_t id = 0; id < src.size(); ++id) {
    const PartKey& k = src.at(id);
    if (k.kind == Kind::Whole || k.kind == Kind::Letter ||
        k.kind == Kind::Delimiter) {
      dst.add(k.kind, k.value);
    }
  }

  // Phase 2: collect Start/Mid/End atoms with their original IDs.
  std::vector<Item> items;
  items.reserve(src.size());
  for (std::uint32_t id = 0; id < src.size(); ++id) {
    const PartKey& k = src.at(id);
    if (k.kind == Kind::Start || k.kind == Kind::Mid ||
        k.kind == Kind::End) {
      items.push_back({id, k.kind, k.value});
    }
  }

  // Sort by the strategy comparator.
  std::sort(items.begin(), items.end(), cmp);

  // Add to new dict in the new order.
  for (const auto& item : items) {
    dst.add(item.kind, item.value);
  }
  return dst;
}

}  // namespace

PartDictionary reorder_dict_by_type(const PartDictionary& src) {
  return reorder_impl(src, [](const Item& a, const Item& b) {
    const int ka = kind_priority(a.kind);
    const int kb = kind_priority(b.kind);
    if (ka != kb) return ka < kb;
    return a.original_id < b.original_id;
  });
}

PartDictionary reorder_dict_by_type_then_level(const PartDictionary& src) {
  return reorder_impl(src, [](const Item& a, const Item& b) {
    const int ka = kind_priority(a.kind);
    const int kb = kind_priority(b.kind);
    if (ka != kb) return ka < kb;
    if (a.value.size() != b.value.size()) {
      return a.value.size() > b.value.size();  // longer first
    }
    return a.original_id < b.original_id;
  });
}

PartDictionary reorder_dict_by_level_then_type(const PartDictionary& src) {
  return reorder_impl(src, [](const Item& a, const Item& b) {
    if (a.value.size() != b.value.size()) {
      return a.value.size() > b.value.size();  // longer first
    }
    const int ka = kind_priority(a.kind);
    const int kb = kind_priority(b.kind);
    if (ka != kb) return ka < kb;
    return a.original_id < b.original_id;
  });
}

}  // namespace core::parts
