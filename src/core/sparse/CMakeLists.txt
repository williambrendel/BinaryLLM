# ============================================================================
# src/core/sparse/CMakeLists.txt
#
# Header-only INTERFACE library. Surprisal-weighted similarity primitives
# (info_jaccard.hpp) — the pipeline's native metric (spec §7.5.0).
#
# Use:
#   target_link_libraries(<your_target> PRIVATE sparse)
# ============================================================================

add_library(sparse INTERFACE)

target_include_directories(sparse INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(sparse INTERFACE cxx_std_20)
