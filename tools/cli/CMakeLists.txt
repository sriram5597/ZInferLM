set(TARGET zinferlm-cli)

add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PUBLIC zinferlm)

add_custom_command(TARGET ${TARGET} POST_BUILD 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${CMAKE_SOURCE_DIR}/models/qwen2.5-1.5b-instruct-q5_k_m.gguf
  ${CMAKE_BINARY_DIR}/models/qwen2.5-1.5b-instruct-q5_k_m.gguf
)

