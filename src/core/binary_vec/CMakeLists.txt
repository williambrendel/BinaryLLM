# ============================================================================
# src/binarycore/binary_vec/CMakeLists.txt
#
# Header-only INTERFACE library. No source files, no objects to link.
# Exposes the include/ directory as a public include path; callers
# include "binary_vec.hpp" (the only public entry).
#
# Use:
#   target_link_libraries(<your_target> PRIVATE binary_vec)
# ============================================================================

add_library(binary_vec INTERFACE)

target_include_directories(binary_vec INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(binary_vec INTERFACE cxx_std_20)
