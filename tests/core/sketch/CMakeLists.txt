add_executable(sketch_tests
  test_frequent_directions.cpp
)

target_link_libraries(sketch_tests PRIVATE
  sketch
  doctest_main
)

target_compile_features(sketch_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(sketch_tests PRIVATE /W4 /WX)
else()
  target_compile_options(sketch_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME sketch_tests COMMAND sketch_tests)
