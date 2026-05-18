add_executable(encode_corpus
    main.cpp
)

target_link_libraries(encode_corpus PRIVATE
    binarycore
)
