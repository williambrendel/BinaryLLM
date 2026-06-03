add_executable(binaryoptim_tests
    main.cpp
    test_dominant_set.cpp
)

target_link_libraries(binaryoptim_tests PRIVATE
    binaryoptim
    doctest
)

add_test(NAME binaryoptim_tests COMMAND binaryoptim_tests)
