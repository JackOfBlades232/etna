#define _CRT_SECURE_NO_WARNINGS
#include "SlangShaderSerialization.hpp"
#include "Common.hpp"

#include <fmt/format.h>

#include <cstdio>

bool slang_shader_ser_write_to_file(
  const SlangShaderSerializationContext& data, std::filesystem::path dest, std::string& error)
{
  FILE* f = fopen(to_char_str(dest.string()).c_str(), "wb+");
  if (!f)
  {
    error = fmt::format("Failed to open {} shader file for writing", dest.string());
    return false;
  }

  for (const auto& chunk : data)
  {
    int res = fwrite(chunk.data(), chunk.size(), 1, f);
    if (res != 1)
    {
      error = fmt::format("Failed to write to {} shader file", dest.string());
      return false;
    }
  }

  fclose(f);
  return true;
}
