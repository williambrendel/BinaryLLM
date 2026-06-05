// ============================================================================
// binarytrain/signatures/cooc_between_builder.cpp
// ============================================================================

#include "binarytrain/signatures/cooc_between_builder.hpp"

#include <cassert>

namespace binarytrain::signatures {

CoocBetweenBuilder::CoocBetweenBuilder(std::size_t feature_count, Config cfg)
    : builder_(feature_count, feature_count),
      f_(feature_count, 0),
      config_(cfg) {}

void CoocBetweenBuilder::add_window(
    const std::vector<std::vector<std::uint32_t>>& window) {
  ++window_count_;
  if (window.empty()) return;

  const std::size_t F = f_.size();

  // Frequencies: count every part occurrence in the window.
  for (const auto& item : window) {
    total_word_count_ += 1;
    for (std::uint32_t p : item) {
      assert(p < F && "part id out of feature_count range");
      (void)F;  // silence unused-variable warning in release builds
      ++f_[p];
      ++total_part_count_;
    }
  }

  // C_between: for each i < j position pair, for each part a in item_i
  // and part b in item_j, add 1 to C_between[a, b].
  //
  // Self-pairs (a == b at different positions) are allowed and contribute
  // to the diagonal — they represent the same part appearing at two
  // positions in the same window, which is structurally meaningful.
  const std::size_t W = window.size();
  for (std::size_t i = 0; i < W; ++i) {
    const auto& word_i = window[i];
    if (word_i.empty()) continue;
    for (std::size_t j = i + 1; j < W; ++j) {
      const auto& word_j = window[j];
      if (word_j.empty()) continue;
      for (std::uint32_t a : word_i) {
        for (std::uint32_t b : word_j) {
          builder_.add(a, b, 1.0);
        }
      }
    }
  }
}

CoocBetweenBuilder::Output CoocBetweenBuilder::finalize() && {
  return Output{
      std::move(builder_).finalize(),
      std::move(f_),
      window_count_,
      total_word_count_,
      total_part_count_,
  };
}

}  // namespace binarytrain::signatures
