add_executable(binarycore_tests
    main.cpp
    test_binary_vec.cpp
    test_popcount.cpp
    test_hamming.cpp
    test_dot.cpp
    test_similarity.cpp
    test_token_encoder.cpp
    test_tokenizer.cpp
    sparse/test_sparse.cpp
)

target_link_libraries(binarycore_tests PRIVATE
    binarycore
    doctest
)

add_test(NAME binarycore_tests COMMAND binarycore_tests)
