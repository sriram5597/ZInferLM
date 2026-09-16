set(TARGET zinferlm-debug)

add_executable(${TARGET} debug.cpp)
target_link_libraries(${TARGET} PUBLIC zinferlm)
