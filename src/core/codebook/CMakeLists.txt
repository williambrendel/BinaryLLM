# ============================================================================
# src/core/codebook/CMakeLists.txt
#
# The learned sparse binary codebook: container (Codebook) + α-threshold
# pursuit encoder. Learning (Pass1Learner) is added in a later stage.
# ============================================================================

add_library(codebook STATIC
  src/codebook.cpp
  src/pursuit.cpp
  src/recon_metrics.cpp
)

target_include_directories(codebook PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(codebook PUBLIC
  sparse
)

target_compile_features(codebook PUBLIC cxx_std_20)
