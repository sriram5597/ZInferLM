set(TARGET zinferlm-serve)

cmake_policy(SET CMP0167 NEW)

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)

find_package(Boost REQUIRED COMPONENTS json)

add_executable(${TARGET} main.cpp server.cpp server.hpp handlers.cpp handlers.hpp)
target_link_libraries(${TARGET} PRIVATE zinferlm Boost::headers Boost::json Threads::Threads)
