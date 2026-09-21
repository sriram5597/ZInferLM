set(TARGET zinferlm-tokenize)

add_executable(${TARGET} tokenize.cpp)
target_link_libraries(${TARGET} PUBLIC zinferlm)
