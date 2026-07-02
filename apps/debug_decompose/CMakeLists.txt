add_executable(debug_decompose main.cpp)

target_link_libraries(debug_decompose PRIVATE
  parts
  signatures
  binary_vec
)

target_compile_features(debug_decompose PRIVATE cxx_std_20)
