# ============================================================================
# src/core/parts2f/CMakeLists.txt
#
# core::parts2f — Pass-1 context-part discovery over the 2F=[L|R] block.
# Two variants (A: boosted-threshold, B: greedy submodular set-cover) behind a
# shared part-growth + per-part threshold. Surprisal-weighted, matrix-free.
# ============================================================================

add_library(parts2f STATIC
  src/parts2f.cpp
)

target_include_directories(parts2f PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(parts2f PUBLIC
  sparse
)

target_compile_features(parts2f PUBLIC cxx_std_20)
