// ============================================================================
// binarytrain/signatures/cooc_within_builder.cpp
// ============================================================================

#include "binarytrain/signatures/cooc_within_builder.hpp"

#include <cassert>

namespace binarytrain::signatures {

CoocWithinBuilder::CoocWithinBuilder(std::size_t feature_count)
    : builder_(feature_count, feature_count),
      feature_count_(feature_count) {}

void CoocWithinBuilder::add_window(
    const std::vector<std::vector<std::uint32_t>>& window) {
  ++window_count_;

  for (const auto& item : window) {
    total_word_count_ += 1;
    const std::size_t n = item.size();
    // For each ordered pair (i < j) of positions in the item, look at
    // the two parts. Skip self-pairs (a == b). Canonicalize to upper
    // triangle (min, max). Sum into the builder.
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t a = item[i];
      assert(a < feature_count_ && "part id out of feature_count range");
      for (std::size_t j = i + 1; j < n; ++j) {
        const std::uint32_t b = item[j];
        assert(b < feature_count_ && "part id out of feature_count range");
        if (a == b) continue;  // self-pair: no within-pair contribution
        const std::uint32_t lo = a < b ? a : b;
        const std::uint32_t hi = a < b ? b : a;
        builder_.add(lo, hi, 1.0);
      }
    }
  }
}

CoocWithinBuilder::Output CoocWithinBuilder::finalize() && {
  return Output{
      std::move(builder_).finalize(),
      window_count_,
      total_word_count_,
  };
}

}  // namespace binarytrain::signatures
