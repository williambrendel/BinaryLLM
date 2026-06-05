#pragma once

// ============================================================================
// binarytrain/signatures/cooc_between_builder.hpp
//
// Builds the asymmetric between-word co-occurrence matrix C_between
// over a stream of sentence windows.
//
// Per-window accumulation:
//
//   For each window (a sentence), for each pair of word positions
//   (i, j) with i < j, for each part a in word_i and each part b in
//   word_j:
//     C_between[a, b] += 1
//
//   The i < j convention bakes positional order into the matrix:
//   C_between[a, b] counts how many times part a appears in some word
//   strictly before part b's word in the same sentence. C_between is
//   asymmetric — C[a, b] and C[b, a] both exist (in general) and have
//   different values.
//
// Frequency vector:
//
//   f[a] += 1 for every occurrence of part a across all windows and
//   word positions. This is the unconditional part frequency and is
//   used downstream to normalize C_between into the conditional
//   probability matrices A_forward[i, j] = C[i, j] / f[j] and
//   A_backward[i, j] = C[j, i] / f[j].
//
// Delimiters:
//
//   Config::include_delimiters (default true) controls whether
//   delimiter tokens contribute to C_between and f. When true, each
//   delimiter token is treated as a 1-part "word" in the window — its
//   decomposed part list (typically a single delimiter-part ID)
//   participates in cross-word pair accumulation exactly like a real
//   word. When false, delimiters are stripped from the window before
//   accumulation.
//
//   The user is responsible for performing the decomposition; the
//   builder consumes ready-made windows of part-ID lists.
//
// Storage:
//
//   Internal accumulation uses binarycore::sparse::SparseMatrixBuilder,
//   which keeps a vector of hash maps until finalize() converts to CSR.
//
// Output:
//
//   finalize() && consumes the builder and returns an Output struct
//   bundling the finalized matrix, frequencies, and window/word/part
//   counts.
// ============================================================================

#include "binarycore/sparse/sparse_matrix.hpp"
#include "binarycore/sparse/sparse_matrix_builder.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace binarytrain::signatures {

class CoocBetweenBuilder {
public:
  struct Config {
    // When true (default), delimiter tokens in the window contribute
    // to C_between and f exactly like word tokens.
    bool include_delimiters = true;
  };

  // Construct with a fixed feature count F (number of parts in the
  // dictionary). The matrix is F x F; the frequency vector has F slots.
  // Part IDs in added windows must be in [0, F).
  explicit CoocBetweenBuilder(std::size_t feature_count, Config cfg);
  
  // Convenience overload: default-constructed Config.
  explicit CoocBetweenBuilder(std::size_t feature_count)
      : CoocBetweenBuilder(feature_count, Config()) {}

  // Add one window. Each element is one tokenized item's decomposition
  // (a vector of part IDs). The order of items in `window` is the
  // sequence order — earlier items get the "i" position in the i < j
  // accumulation.
  //
  // The builder cannot tell words from delimiters once they've been
  // decomposed; the caller is responsible for filtering delimiters
  // out beforehand if config.include_delimiters is false.
  void add_window(const std::vector<std::vector<std::uint32_t>>& window);

  // Stats during accumulation (read-only).
  std::size_t feature_count() const noexcept { return f_.size(); }
  std::size_t window_count() const noexcept { return window_count_; }
  std::size_t total_word_count() const noexcept { return total_word_count_; }
  std::size_t total_part_count() const noexcept { return total_part_count_; }
  const std::vector<std::uint64_t>& frequencies() const noexcept { return f_; }
  const Config& config() const noexcept { return config_; }

  // Finalize and return outputs. Consumes the builder (rvalue-qualified
  // to match the underlying SparseMatrixBuilder's finalize()).
  struct Output {
    binarycore::sparse::SparseMatrix<double> c_between;
    std::vector<std::uint64_t> frequencies;
    std::size_t window_count;
    std::size_t total_word_count;
    std::size_t total_part_count;
  };
  Output finalize() &&;

private:
  binarycore::sparse::SparseMatrixBuilder<double> builder_;
  std::vector<std::uint64_t> f_;
  Config config_;
  std::size_t window_count_ = 0;
  std::size_t total_word_count_ = 0;
  std::size_t total_part_count_ = 0;
};

}  // namespace binarytrain::signatures
