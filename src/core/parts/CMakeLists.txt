# ============================================================================
# src/core/parts/CMakeLists.txt
#
# core::parts library — tokenizer, dictionary, extractor, decomposer,
# dict_reorder side functions, dict_augment, and the trained-only
# dict_io save/load functions.
# ============================================================================

add_library(parts STATIC
  src/tokenize.cpp
  src/dictionary.cpp
  src/decomposer.cpp
  src/extractor.cpp
  src/dict_reorder.cpp
  src/dict_augment.cpp
  src/dict_io.cpp
)

target_include_directories(parts PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(parts PUBLIC
  adaptive_threshold
)

target_compile_features(parts PUBLIC cxx_std_20)
