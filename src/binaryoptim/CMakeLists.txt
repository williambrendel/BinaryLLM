add_library(binaryoptim STATIC
  src/dominant_set.cpp
  src/thresholds.cpp
)

target_include_directories(binaryoptim PUBLIC include)

target_compile_features(binaryoptim PUBLIC cxx_std_20)
