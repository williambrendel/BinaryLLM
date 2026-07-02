# ============================================================================
# src/core/sketch/CMakeLists.txt
#
# Frequent Directions sketch over the signature stream. Depends on the
# core::math dense kernels and the binary_vec signature type.
# ============================================================================

add_library(sketch STATIC
  src/frequent_directions.cpp
)

target_include_directories(sketch PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sketch PUBLIC
  math
  binary_vec
)

target_compile_features(sketch PUBLIC cxx_std_20)
