add_executable(math_tests
  test_dense.cpp
)

target_link_libraries(math_tests PRIVATE
  math
  doctest_main
)

target_compile_features(math_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(math_tests PRIVATE /W4 /WX)
else()
  target_compile_options(math_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME math_tests COMMAND math_tests)
