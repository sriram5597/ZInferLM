add_library(zinferlm ../include/zinferlm/models.h ../include/zinferlm/tokenizer.h
    ./model_loader/gguf/gguf.h ./model_loader/gguf/gguf.cpp 
    ./model_loader/gguf/metadata.cpp ./model_loader/gguf/metadata.h
    ./model_loader/gguf/tensors.cpp ./model_loader/gguf/tensors.h
    ./tokenizer/tokenizer.cpp ./tokenizer/tokenizer.h
    ./engine/graph.cpp ./engine/graph.h
    ./engine/tensors.cpp ./engine/tensors.h
    ./layers/layers.h ./layers/token_emb.cpp ./layers/token_unemb.cpp
    ./models/qwen/model.cpp ./models/qwen/model.h
    ./models/models.cpp
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 QUIET pcre2-8)

if (NOT PCRE2_FOUND)
    find_path(PCRE2_INCLUDE_DIRS pcre2.h)
    find_library(PCRE2_LIBRARIES NAMES pcre2-8 pcre2-8d)
    if (PCRE2_INCLUDE_DIRS AND PCRE2_LIBRARIES)
        set(PCRE2_FOUND TRUE)
    endif()
endif()

if (NOT PCRE2_FOUND)
    message(FATAL_ERROR "PCRE2 (libpcre2-8) not found! Please install it via your package manager:\n"
                        "  Ubuntu/Debian: sudo apt install libpcre2-dev\n"
                        "  macOS:        brew install pcre2\n"
                        "  Windows:      vcpkg install pcre2")
endif()

target_link_libraries(zinferlm PRIVATE ${PCRE2_LIBRARIES} ggml)

target_include_directories(zinferlm PRIVATE . ${PCRE2_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/ggml/include)
target_include_directories(zinferlm PUBLIC ../include)
