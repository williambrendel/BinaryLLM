# ============================================================================
# tests/core/adaptive_threshold/CMakeLists.txt
#
# Standalone test executable for the adaptive_threshold submodule. Two
# test source files linked into one executable for easy invocation:
#   - test_thresholds.cpp     — per-utility (entropy, gap_ratio) + spot checks
#   - test_adaptive_prune.cpp — elbow, scaled_mm, and composite cascade
#
# Expects:
#   - The adaptive_threshold STATIC library is defined (added via
#     add_subdirectory(src/core/adaptive_threshold) from a parent
#     CMakeLists).
#   - A doctest_main target exists providing doctest's main().
#
# Two integration options for the root tests/core/CMakeLists.txt:
#
# Option A (preferred — one executable per submodule, this file):
#   In tests/core/CMakeLists.txt add:
#     add_subdirectory(adaptive_threshold)
#
# Option B (single core_tests executable):
#   Skip this CMakeLists. In tests/core/CMakeLists.txt add the two
#   sources to the existing target:
#     target_sources(core_tests PRIVATE
#       adaptive_threshold/test_thresholds.cpp
#       adaptive_threshold/test_adaptive_prune.cpp
#     )
#     target_link_libraries(core_tests PRIVATE adaptive_threshold)
#
# Going with Option A here.
# ============================================================================

add_executable(adaptive_threshold_tests
  test_thresholds.cpp
  test_adaptive_prune.cpp
)

target_link_libraries(adaptive_threshold_tests PRIVATE
  adaptive_threshold
  doctest_main         # existing target providing doctest's main()
)

target_compile_features(adaptive_threshold_tests PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(adaptive_threshold_tests PRIVATE /W4 /WX)
else()
  target_compile_options(adaptive_threshold_tests PRIVATE -Wall -Wextra -Werror)
endif()

add_test(NAME adaptive_threshold_tests COMMAND adaptive_threshold_tests)
