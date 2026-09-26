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

add_library(gufo_gemma4_reference STATIC EXCLUDE_FROM_ALL reference.cpp)
target_link_libraries(gufo_gemma4_reference PUBLIC gufo_gemma4)
# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(reference.cpp PROPERTIES COMPILE_OPTIONS "-O2")
