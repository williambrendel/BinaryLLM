# ============================================================================
# tests/binarycore/binary_vec/CMakeLists.txt
#
# Standalone test executable for the binary_vec submodule. One test
# file per source header, all linked into a single binary_vec_tests
# executable for easy invocation.
#
# Expects:
#   - The binary_vec INTERFACE library is defined (added via
#     add_subdirectory(src/binarycore/binary_vec) from a parent
#     CMakeLists).
#   - A doctest_main target exists providing doctest's main().
#
# Two integration options for the root tests/binarycore/CMakeLists.txt:
#
# Option A (preferred — one executable per submodule, this file):
#   In tests/binarycore/CMakeLists.txt add:
#     add_subdirectory(binary_vec)
#
# Option B (single binarycore_tests executable):
#   Skip this CMakeLists. In tests/binarycore/CMakeLists.txt add the
#   five sources to the existing target:
#     target_sources(binarycore_tests PRIVATE
#       binary_vec/test_binary_vec.cpp
#       binary_vec/test_dense_binary_vec.cpp
#       binary_vec/test_sparse_binary_vec.cpp
#       binary_vec/test_big_sparse_binary_vec.cpp
#       binary_vec/test_big_sparse_binary_vec_dynamic.cpp
#     )
#     target_link_libraries(binarycore_tests PRIVATE binary_vec)
#
# Going with Option A here.
# ============================================================================

add_executable(binary_vec_tests
  test_binary_vec.cpp
  test_dense_binary_vec.cpp
  test_sparse_binary_vec.cpp
  test_big_sparse_binary_vec.cpp
  test_big_sparse_binary_vec_dynamic.cpp
)

target_link_libraries(binary_vec_tests PRIVATE
  binary_vec
  doctest_main         # existing target providing doctest's main()
)

target_compile_features(binary_vec_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(binary_vec_tests PRIVATE /W4 /WX)
else()
  target_compile_options(binary_vec_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME binary_vec_tests COMMAND binary_vec_tests)
