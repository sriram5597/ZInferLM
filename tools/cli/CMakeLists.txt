set(TARGET zinferlm-cli)

add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PUBLIC zinferlm)

