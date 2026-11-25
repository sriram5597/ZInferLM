set(TARGET zinferlm-serve)

# Prefer BoostConfig.cmake over the removed FindBoost module.
cmake_policy(SET CMP0167 NEW)

# Third-party libraries (nlohmann/json, minja) come in through the zinferlm
# target; they are acquired once in cmake/deps.cmake, not here.
find_package(Boost REQUIRED COMPONENTS json)
find_package(Threads REQUIRED)

add_executable(${TARGET} main.cpp server.cpp server.hpp handlers.cpp handlers.hpp)
target_link_libraries(${TARGET} PRIVATE zinferlm Boost::headers Boost::json Threads::Threads)
