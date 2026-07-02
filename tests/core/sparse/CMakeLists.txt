# ============================================================================
# tests/core/sparse/CMakeLists.txt
#
# Test executable for the sparse submodule (surprisal-weighted similarity).
# ============================================================================

add_executable(sparse_tests
  test_info_jaccard.cpp
)

target_link_libraries(sparse_tests PRIVATE
  sparse
  doctest_main
)

target_compile_features(sparse_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(sparse_tests PRIVATE /W4 /WX)
else()
  target_compile_options(sparse_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME sparse_tests COMMAND sparse_tests)
