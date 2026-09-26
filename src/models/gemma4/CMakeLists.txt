# Gemma 4 dense text model package (gemma4 target, gemma4-assistant MTP
# drafter). The scalar oracle is built only by explicit validation targets.
add_library(gufo_gemma4 STATIC
  chat_template.cpp
  config.cpp
  tokenizer.cpp
  tool_syntax.cpp
  weights.cpp)

target_include_directories(gufo_gemma4 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_gemma4 PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  set(GEMMA4_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  target_sources(gufo_gemma4 PRIVATE
    ${GEMMA4_ROCM_DIR}/kernels.hip.cpp
    ${GEMMA4_ROCM_DIR}/device_model.cpp
    ${GEMMA4_ROCM_DIR}/executor.cpp
    engine.cpp)
  set_source_files_properties(${GEMMA4_ROCM_DIR}/kernels.hip.cpp
    PROPERTIES LANGUAGE HIP)
  target_compile_options(gufo_gemma4 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_gemma4 PUBLIC hip::host roc::hipblas)
endif()

add_library(gufo_gemma4_reference STATIC EXCLUDE_FROM_ALL reference.cpp)
target_link_libraries(gufo_gemma4_reference PUBLIC gufo_gemma4)
# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(reference.cpp PROPERTIES COMPILE_OPTIONS "-O2")
