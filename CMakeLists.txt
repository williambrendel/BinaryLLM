cmake_minimum_required(VERSION 3.20)
project(BinaryLLM
    VERSION 0.1.0
    DESCRIPTION "Binary-native LLM architecture for edge devices"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BINARYLLM_BUILD_TESTS "Build tests" ON)
option(BINARYLLM_BUILD_APPS  "Build demo apps" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        add_compile_options(-O3)
    endif()
endif()

# Doctest header-only test framework
add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE
    ${CMAKE_SOURCE_DIR}/external/doctest
)

# Pre-compiled doctest main() — link this into a test executable to
# get the include path AND main() in one step.
add_library(doctest_main STATIC ${CMAKE_SOURCE_DIR}/tests/doctest_main.cpp)
target_link_libraries(doctest_main PUBLIC doctest)

add_subdirectory(src)

if(BINARYLLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BINARYLLM_BUILD_APPS)
    add_subdirectory(apps)
endif()