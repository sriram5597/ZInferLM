set(TARGET zinferlm-memplan)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PUBLIC zinferlm)
