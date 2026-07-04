add_executable(context_parts_eval main.cpp)

target_link_libraries(context_parts_eval PRIVATE
  parts2f
  parts
  signatures
  binary_vec
)

target_compile_features(context_parts_eval PRIVATE cxx_std_20)
