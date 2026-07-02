add_executable(parts_tests
  test_tokenize.cpp
  test_tokenize_connectors.cpp
  test_dictionary.cpp
  test_dictionary_binary.cpp
  test_decomposer.cpp
  test_dict_reorder.cpp
  test_dict_io.cpp
  test_extractor.cpp
)

target_link_libraries(parts_tests PRIVATE
  parts
  doctest_main
)

target_compile_features(parts_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(parts_tests PRIVATE /W4 /WX)
else()
  target_compile_options(parts_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME parts_tests COMMAND parts_tests)
