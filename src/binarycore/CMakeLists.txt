add_library(binarycore STATIC
    src/encoding/token_encoder.cpp
    src/encoding/tokenizer.cpp
)

target_include_directories(binarycore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(binarycore PUBLIC cxx_std_20)
