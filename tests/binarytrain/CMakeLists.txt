add_executable(binarytrain_tests
  main.cpp
  parts/test_tokenize.cpp
  parts/test_tokenize_connectors.cpp
  parts/test_dictionary.cpp
  parts/test_dictionary_binary.cpp
  parts/test_extractor.cpp
  parts/test_decomposer.cpp
  signatures/test_sentence_split.cpp
  signatures/test_cooc_between.cpp
  signatures/test_cooc_within.cpp
)

target_include_directories(binarytrain_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/external/doctest
)

target_link_libraries(binarytrain_tests PRIVATE binarytrain)

add_test(NAME binarytrain_tests COMMAND binarytrain_tests)
