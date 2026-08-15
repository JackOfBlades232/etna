#include "Slang.hpp"

#include <spdlog/spdlog.h>

#include <cstring>
#include <memory>
#include <filesystem>

// @TODO: errors

int main(int argc, char** argv)
{
  std::filesystem::path source{};
  std::filesystem::path output{};
  std::filesystem::path depfile{};
  std::string target{};
  SlangCompiler::CreateInfo slangCi{};

  for (int i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], "-I") == 0)
    {
      ++i;
      if (i >= argc)
        assert(0);
      slangCi.commonIncludeDirs.emplace_back(argv[i]);
    }
    else if (strcmp(argv[i], "-g") == 0)
    {
      slangCi.emitSpirvDebugInfo = true;
    }
    else if (strcmp(argv[i], "-o") == 0)
    {
      ++i;
      if (i >= argc)
        assert(0);
      assert(output.empty());
      output = argv[i];
    }
    else if (strcmp(argv[i], "--depfile") == 0)
    {
      ++i;
      if (i >= argc)
        assert(0);
      assert(depfile.empty());
      depfile = argv[i];
    }
    else if (strcmp(argv[i], "-target") == 0)
    {
      ++i;
      if (i >= argc)
        assert(0);
      assert(target.empty());
      target = argv[i];
    }
    else
    {
      assert(source.empty());
      source = argv[i];
    }
  }

  assert(!source.empty());
  assert(!output.empty());
  assert(!target.empty());

  return SlangCompiler{slangCi}.compile(source, target, output, depfile);
}
