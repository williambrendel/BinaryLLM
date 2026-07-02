# ============================================================================
# src/core/pass1/CMakeLists.txt
#
# Pass-1 dictionary learner (Online Dictionary Learning). Orchestrates the
# codebook, the FD factor (seeding), and the surprisal metric.
# ============================================================================

add_library(pass1 STATIC
  src/pass1.cpp
)

target_include_directories(pass1 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pass1 PUBLIC
  codebook
  sketch
  sparse
)

target_compile_features(pass1 PUBLIC cxx_std_20)
