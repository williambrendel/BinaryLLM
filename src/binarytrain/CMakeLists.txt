add_library(binarytrain STATIC
  src/parts/tokenize.cpp
  src/parts/dictionary.cpp
  src/parts/extractor.cpp
  src/parts/decomposer.cpp
  src/signatures/sentence_split.cpp
  src/signatures/cooc_between_builder.cpp
  src/signatures/cooc_within_builder.cpp
)

target_include_directories(binarytrain PUBLIC include)

target_link_libraries(binarytrain PUBLIC
  binaryoptim
  binarycore
)

target_compile_features(binarytrain PUBLIC cxx_std_20)
