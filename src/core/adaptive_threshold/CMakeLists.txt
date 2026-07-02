# ============================================================================
# src/core/adaptive_threshold/CMakeLists.txt
#
# STATIC library (has a .cpp). Exposes the include/ directory as a
# public include path; callers include "adaptive_threshold.hpp".
#
# Use:
#   target_link_libraries(<your_target> PRIVATE adaptive_threshold)
# ============================================================================

add_library(adaptive_threshold STATIC
  src/adaptive_threshold.cpp
)

target_include_directories(adaptive_threshold PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(adaptive_threshold PUBLIC cxx_std_20)
