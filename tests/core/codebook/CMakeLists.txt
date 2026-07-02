add_executable(codebook_tests
  test_codebook.cpp
  test_pursuit.cpp
)

target_link_libraries(codebook_tests PRIVATE
  codebook
  doctest_main
)

target_compile_features(codebook_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(codebook_tests PRIVATE /W4 /WX)
else()
  target_compile_options(codebook_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME codebook_tests COMMAND codebook_tests)
