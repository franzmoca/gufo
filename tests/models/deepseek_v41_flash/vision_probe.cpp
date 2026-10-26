// DeepSeek V4.1 Flash vision probe.
//
//   ds41_vision_probe --image IMAGE [--pixels OUT.rgb]
//     decodes, plans and pads like the prompt builder; prints the grid and
//     writes the padded RGB8 pixels.
//   ds41_vision_probe --image IMAGE --mmproj SIDECAR --output SPAN.f32
//                     [--repeat N]
//     also encodes the image on the GPU and writes the span's FP32 rows
//     (tools/ds41/vision_reference.py --compare reads them).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/image.hpp"
#include "src/models/deepseek_v41_flash/vision/encoder.hpp"
#include "src/models/deepseek_v41_flash/vision/prompt.hpp"

namespace vision = gufo::models::deepseek_v41_flash::vision;

int main(int argc, char** argv) {
  std::string image_path, pixels_path, mmproj, output;
  int repeat = 1;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--image")
      image_path = argv[i + 1];
    else if (flag == "--pixels")
      pixels_path = argv[i + 1];
    else if (flag == "--mmproj")
      mmproj = argv[i + 1];
    else if (flag == "--output")
      output = argv[i + 1];
    else if (flag == "--repeat")
      repeat = std::stoi(argv[i + 1]);
    else {
      std::cerr << "unknown flag " << flag << '\n';
      return 2;
    }
  }
  if (image_path.empty()) {
    std::cerr << "usage: ds41_vision_probe --image IMAGE [--pixels OUT.rgb] "
                 "[--mmproj SIDECAR --output SPAN.f32 [--repeat N]]\n";
    return 2;
  }
  try {
    const auto decoded =
        gufo::core::DecodeImage(gufo::core::ReadImageFile(image_path));
    const auto grid = vision::PlanGrid(decoded.width, decoded.height);
    const auto padded = vision::PadImage(decoded, grid);
    std::printf(
        "{\"width\": %u, \"height\": %u, \"vit_width\": %u, \"vit_height\": "
        "%u, \"llm_width\": %u, \"llm_height\": %u, \"tokens\": %u}\n",
        grid.width, grid.height, grid.vit_width, grid.vit_height,
        grid.llm_width, grid.llm_height, grid.Tokens());
    if (!pixels_path.empty()) {
      std::ofstream(pixels_path, std::ios::binary)
          .write(reinterpret_cast<const char*>(padded.pixels.data()),
                 static_cast<std::streamsize>(padded.pixels.size()));
    }
    if (mmproj.empty())
      return 0;
    vision::Encoder encoder(mmproj);
    const auto upload = std::chrono::steady_clock::now();
    encoder.Upload();
    std::printf("upload %.0f ms, %.1f MiB resident\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - upload)
                    .count(),
                encoder.ResidentBytes() / 1048576.0);
    std::shared_ptr<const vision::Encoder::Embedding> span;
    for (int r = 0; r < repeat; ++r) {
      const auto start = std::chrono::steady_clock::now();
      span = encoder.Encode(padded, grid);
      std::printf("encode %.1f ms\n",
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count());
    }
    std::vector<float> rows(std::size_t{span->rows()} *
                            vision::kEmbeddingWidth);
    if (hipMemcpy(rows.data(), span->data(), rows.size() * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      std::cerr << "span copy failed\n";
      return 1;
    }
    if (!output.empty()) {
      std::ofstream(output, std::ios::binary)
          .write(reinterpret_cast<const char*>(rows.data()),
                 static_cast<std::streamsize>(rows.size() * sizeof(float)));
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
