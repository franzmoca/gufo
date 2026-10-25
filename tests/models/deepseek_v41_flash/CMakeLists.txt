add_executable(ds41_chat_template_test
  chat_template_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/deepseek_v41_flash/chat_template.cpp)
target_link_libraries(ds41_chat_template_test PRIVATE gufo_core)
target_compile_definitions(ds41_chat_template_test PRIVATE
  GUFO_DS41_CHAT_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/encoding-1017.json")
add_test(NAME ds41.template COMMAND ds41_chat_template_test)
set_tests_properties(ds41.template PROPERTIES LABELS "cpu;ds41;tokenization")

if(ENGINE_ENABLE_HIP)
  add_executable(ds41_load_probe EXCLUDE_FROM_ALL load_probe.cpp)
  target_link_libraries(ds41_load_probe PRIVATE gufo_deepseek_v41_flash)
endif()
