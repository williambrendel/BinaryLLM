add_executable(binarycore_tests
    main.cpp
    test_binary_vec.cpp
    test_token_encoder.cpp
    test_tokenizer.cpp
)

target_link_libraries(binarycore_tests PRIVATE
    binarycore
    doctest
)

# Register with CTest
add_test(NAME binarycore_tests COMMAND binarycore_tests)
