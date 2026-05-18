cmake_minimum_required(VERSION 3.20)
project(BinaryLLM
    VERSION 0.1.0
    DESCRIPTION "Binary-native LLM architecture for edge devices"
    LANGUAGES CXX
)

# C++20 for std::popcount, std::bit_cast
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always emit compile_commands.json (clangd uses this for IntelliSense)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BINARYLLM_BUILD_TESTS "Build tests" ON)
option(BINARYLLM_BUILD_APPS  "Build demo apps" ON)

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        add_compile_options(-O3)
        # Don't use -march=native here — it makes binaries non-portable.
        # We rely on std::popcount which the compiler maps to hardware POPCNT.
    endif()
endif()

# Doctest header-only test framework
add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE
    ${CMAKE_SOURCE_DIR}/external/doctest
)

# Project libraries
add_subdirectory(src/binarycore)

# Tests
if(BINARYLLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/binarycore)
endif()

# Apps
if(BINARYLLM_BUILD_APPS)
    add_subdirectory(apps)
endif()
