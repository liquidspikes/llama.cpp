set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_executable(ggml-rpc-cache-dump ggml-rpc-cache-dump.cpp)
target_link_libraries(ggml-rpc-cache-dump PRIVATE ggml)
target_compile_features(ggml-rpc-cache-dump PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ggml-rpc-cache-dump RUNTIME)
endif()
