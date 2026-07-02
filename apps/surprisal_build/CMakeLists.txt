add_executable(surprisal_build main.cpp)

target_link_libraries(surprisal_build PRIVATE
  parts
  signatures
  binary_vec
)

target_compile_features(surprisal_build PRIVATE cxx_std_20)
