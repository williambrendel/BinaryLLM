add_executable(pass1_train main.cpp)

target_link_libraries(pass1_train PRIVATE
  parts
  signatures
  sketch
  pass1
  codebook
  binary_vec
)

target_compile_features(pass1_train PRIVATE cxx_std_20)
