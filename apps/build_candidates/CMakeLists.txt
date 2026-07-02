add_executable(build_candidates main.cpp)

target_link_libraries(build_candidates PRIVATE
  parts
  signatures
  binary_vec
)

target_compile_features(build_candidates PRIVATE cxx_std_20)
