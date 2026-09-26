# Host contracts run in the hosted PR suite. Model-backed checks read the
# GGUF named by GUFO_GEMMA4_MODEL and skip (77) without it.
add_custom_target(gemma4_tests)

function(add_gemma4_test name library labels)
  set(target gemma4_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  target_link_libraries(${target} PRIVATE ${library})
  add_test(NAME gemma4.${name} COMMAND ${target})
  set_tests_properties(gemma4.${name} PROPERTIES
    LABELS "gemma4;${labels}" SKIP_RETURN_CODE 77)
  add_dependencies(gemma4_tests ${target})
endfunction()

add_gemma4_test(config gufo_gemma4 "cpu")
add_gemma4_test(tokenizer_contract gufo_gemma4 "cpu;tokenization")
add_gemma4_test(template gufo_gemma4 "cpu;chat_template")
add_gemma4_test(tool_syntax gufo_gemma4 "cpu;tools")
add_gemma4_test(artifact gufo_gemma4 "cpu;tokenization;chat_template;external-model")
foreach(target gemma4_template_test gemma4_artifact_test)
  target_compile_definitions(${target} PRIVATE
    GUFO_CHAT_TEMPLATE_HF_GOLDENS="${PROJECT_SOURCE_DIR}/tests/fixtures/chat_template_hf_goldens.json"
    GUFO_GEMMA4_FIXTURES="${PROJECT_SOURCE_DIR}/tests/models/gemma4/fixtures")
endforeach()

add_executable(gemma4_reference_probe EXCLUDE_FROM_ALL reference_probe.cpp)
target_link_libraries(gemma4_reference_probe PRIVATE gufo_gemma4_reference)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  foreach(name IN ITEMS attention_ops kernel_ops projection_ops)
    add_gemma4_test(${name} gufo_gemma4 "hip;gpu;gfx1151")
  endforeach()
  add_executable(gemma4_gpu_probe EXCLUDE_FROM_ALL gpu_probe.cpp)
  target_link_libraries(gemma4_gpu_probe PRIVATE gufo_gemma4)
endif()
