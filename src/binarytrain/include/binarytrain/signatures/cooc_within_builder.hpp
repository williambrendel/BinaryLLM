#pragma once

// ============================================================================
// binarytrain/signatures/cooc_within_builder.hpp
//
// Builds the symmetric within-item co-occurrence matrix C_within.
//
// Per-window accumulation:
//
//   For each window (a sentence), for each item (word) in the window,
//   for each unordered pair (a, b) of distinct parts in that item's
//   decomposition with a < b:
//     C_within[a, b] += 1
//
//   Storage is upper-triangle only (a < b). Self-pairs (a == b) are
//   skipped: an item with a repeated part counts that repetition in
//   pairs *with other distinct parts*, not as a pair with itself.
//
//   Multiple occurrences of distinct parts within one item count
//   multiplicatively. An item [a, a, b] contributes
//     C_within[min(a, b), max(a, b)] += 2  (two a's paired with one b)
//   following the same "every position pair counts" rule as the
//   between matrix.
//
// Not currently consumed by the downstream attention pipeline — kept
// available for future use (intra-word structural signatures, parts
// affinity within morphology). Build it only when needed; the phase 2
// app does not invoke this builder by default.
//
// Delimiters:
//
//   Delimiters typically decompose to a single part, so they contribute
//   no within-item pairs and the within matrix is unaffected by
//   delimiter inclusion. The builder takes no Config — feed it whatever
//   item set you want.
// ============================================================================

#include "binarycore/sparse/sparse_matrix.hpp"
#include "binarycore/sparse/sparse_matrix_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace binarytrain::signatures {

class CoocWithinBuilder {
public:
  // Construct with a fixed feature count F. Part IDs in added windows
  // must be in [0, F).
  explicit CoocWithinBuilder(std::size_t feature_count);

  // Add one window. Each element is one item's decomposition. Within
  // accumulation runs per-item only — window position order does not
  // affect the within matrix.
  void add_window(const std::vector<std::vector<std::uint32_t>>& window);

  // Stats during accumulation.
  std::size_t feature_count() const noexcept { return feature_count_; }
  std::size_t window_count() const noexcept { return window_count_; }
  std::size_t total_word_count() const noexcept { return total_word_count_; }

  // Finalize and return outputs. Consumes the builder.
  struct Output {
    binarycore::sparse::SparseMatrix<double> c_within;
    std::size_t window_count;
    std::size_t total_word_count;
  };
  Output finalize() &&;

private:
  binarycore::sparse::SparseMatrixBuilder<double> builder_;
  std::size_t feature_count_;
  std::size_t window_count_ = 0;
  std::size_t total_word_count_ = 0;
};

}  // namespace binarytrain::signatures
