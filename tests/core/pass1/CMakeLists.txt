add_executable(pass1_tests
  test_pass1.cpp
)

target_link_libraries(pass1_tests PRIVATE
  pass1
  doctest_main
)

target_compile_features(pass1_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(pass1_tests PRIVATE /W4 /WX)
else()
  target_compile_options(pass1_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME pass1_tests COMMAND pass1_tests)
