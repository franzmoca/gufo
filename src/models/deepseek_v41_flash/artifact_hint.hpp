#ifndef GUFO_MODELS_DEEPSEEK_V41_FLASH_ARTIFACT_HINT_HPP_
#define GUFO_MODELS_DEEPSEEK_V41_FLASH_ARTIFACT_HINT_HPP_

#include <filesystem>
#include <regex>
#include <string>
#include <system_error>

namespace gufo::models::deepseek_v41_flash {

inline constexpr char kEngramFileName[] =
    "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";

/// A hint for a GSQ-RCO build that cannot open: every build's last shard
/// is a link to the shared Engram file, which downloads do not include.
/// Returns an empty string unless that link is what is missing.
inline std::string MissingEngramShardHint(
    const std::filesystem::path& first_shard) {
  static const std::regex kSplit{R"(^(.*)-00001-of-(\d{5})\.gguf$)"};
  std::smatch match;
  const std::string name = first_shard.filename().string();
  if (!std::regex_match(name, match, kSplit))
    return {};
  const std::string count = match[2].str();
  const auto folder = first_shard.parent_path();
  const auto link =
      folder / (match[1].str() + "-" + count + "-of-" + count + ".gguf");
  std::error_code error;
  if (std::filesystem::symlink_status(link, error).type() !=
          std::filesystem::file_type::not_found ||
      !std::filesystem::exists(folder / kEngramFileName, error))
    return {};
  const std::string command =
      "ln -s " + std::string(kEngramFileName) + " " + link.string();
  return "\nThe shared Engram file is the last shard of each DeepSeek V4.1 "
         "GSQ-RCO build; link it once per build (README_DEEPSEEK.md):\n  " +
         command;
}

}  // namespace gufo::models::deepseek_v41_flash

#endif  // GUFO_MODELS_DEEPSEEK_V41_FLASH_ARTIFACT_HINT_HPP_
