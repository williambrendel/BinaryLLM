# ============================================================================
# src/core/math/CMakeLists.txt
#
# Tier-1 dense float32 kernels (matmul / gram / symmetric eigensolver).
# fast-math is scoped to THIS target only — it enables the reduction
# reassociation the contiguous inner loops need to vectorize, and must not
# leak to code that relies on strict IEEE (e.g. surprisal's isinf).
# ============================================================================

find_package(Threads REQUIRED)

add_library(math STATIC
  src/dense.cpp
)

target_include_directories(math PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(math PUBLIC Threads::Threads)

target_compile_features(math PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(math PRIVATE -ffast-math)
elseif(MSVC)
  target_compile_options(math PRIVATE /fp:fast)
endif()
