add_executable(parts2f_tests
  test_parts2f.cpp
)

target_link_libraries(parts2f_tests PRIVATE
  parts2f
  doctest_main
)

target_compile_features(parts2f_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(parts2f_tests PRIVATE /W4 /WX)
else()
  target_compile_options(parts2f_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME parts2f_tests COMMAND parts2f_tests)
