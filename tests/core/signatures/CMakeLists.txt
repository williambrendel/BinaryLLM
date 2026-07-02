add_executable(signatures_tests
  test_encoder.cpp
  test_sentence_split.cpp
  test_paragraph_split.cpp
  test_word_encoder.cpp
  test_candidate_file.cpp
  test_weight_table.cpp
)

target_link_libraries(signatures_tests PRIVATE
  signatures
  doctest_main
)

target_compile_features(signatures_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(signatures_tests PRIVATE /W4 /WX)
else()
  target_compile_options(signatures_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME signatures_tests COMMAND signatures_tests)