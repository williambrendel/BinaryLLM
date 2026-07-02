add_library(signatures STATIC
  src/sentence_split.cpp
  src/paragraph_split.cpp
  src/word_encoder.cpp
  src/encoder.cpp
  src/candidate_file.cpp
  src/weight_table.cpp
)

target_include_directories(signatures PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(signatures PUBLIC
  parts
  binary_vec
)

target_compile_features(signatures PUBLIC cxx_std_20)