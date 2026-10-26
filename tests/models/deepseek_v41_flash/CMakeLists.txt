add_executable(ds41_chat_template_test
  chat_template_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/deepseek_v41_flash/chat_template.cpp)
target_link_libraries(ds41_chat_template_test PRIVATE gufo_core)
target_compile_definitions(ds41_chat_template_test PRIVATE
  GUFO_DS41_CHAT_GOLDENS="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/encoding-1017.json")
add_test(NAME ds41.template COMMAND ds41_chat_template_test)
set_tests_properties(ds41.template PROPERTIES LABELS "cpu;ds41;tokenization")

add_executable(ds41_vision_prompt_test
  vision_prompt_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/deepseek_v41_flash/vision/prompt.cpp)
target_link_libraries(ds41_vision_prompt_test PRIVATE gufo_core)
add_test(NAME ds41.vision_prompt COMMAND ds41_vision_prompt_test)
set_tests_properties(ds41.vision_prompt PROPERTIES LABELS "cpu;ds41;vision")

if(ENGINE_ENABLE_HIP)
  add_executable(ds41_mmq_typed_test mmq_typed_test.cpp)
  target_link_libraries(ds41_mmq_typed_test PRIVATE gufo_deepseek_v41_flash)
  add_test(NAME ds41.mmq_typed COMMAND ds41_mmq_typed_test)
  set_tests_properties(ds41.mmq_typed PROPERTIES
    LABELS "gpu;hip;gfx1151;ds41;quality"
    SKIP_RETURN_CODE 77)

  add_executable(ds41_attention_test attention_test.cpp)
  target_link_libraries(ds41_attention_test PRIVATE gufo_deepseek_v41_flash)
  add_test(NAME ds41.attention COMMAND ds41_attention_test)
  set_tests_properties(ds41.attention PROPERTIES
    LABELS "gpu;hip;gfx1151;ds41;quality"
    SKIP_RETURN_CODE 77)

  add_executable(ds41_hc_state_pass_test hc_state_pass_test.cpp)
  target_link_libraries(ds41_hc_state_pass_test PRIVATE gufo_deepseek_v41_flash)
  add_test(NAME ds41.hc_state_pass COMMAND ds41_hc_state_pass_test)
  set_tests_properties(ds41.hc_state_pass PROPERTIES
    LABELS "gpu;hip;gfx1151;ds41;quality"
    SKIP_RETURN_CODE 77)

  add_executable(ds41_candidate_mask_test candidate_mask_test.cpp)
  target_link_libraries(ds41_candidate_mask_test PRIVATE gufo_deepseek_v41_flash)
  add_test(NAME ds41.candidate_mask COMMAND ds41_candidate_mask_test)
  set_tests_properties(ds41.candidate_mask PROPERTIES
    LABELS "gpu;hip;gfx1151;ds41;quality"
    SKIP_RETURN_CODE 77)

  add_executable(ds41_gemv_bench EXCLUDE_FROM_ALL gemv_bench.cpp)
  target_link_libraries(ds41_gemv_bench PRIVATE gufo_deepseek_v41_flash)

  add_executable(ds41_mmq_bench EXCLUDE_FROM_ALL mmq_bench.cpp)
  target_link_libraries(ds41_mmq_bench PRIVATE gufo_deepseek_v41_flash)

  add_executable(ds41_vision_probe EXCLUDE_FROM_ALL vision_probe.cpp)
  target_link_libraries(ds41_vision_probe PRIVATE gufo_deepseek_v41_flash)

  add_executable(ds41_load_probe EXCLUDE_FROM_ALL load_probe.cpp)
  target_link_libraries(ds41_load_probe PRIVATE gufo_deepseek_v41_flash)
endif()
