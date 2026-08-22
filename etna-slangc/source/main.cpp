#include "Slang.hpp"
#include "Commandline.hpp"

#include <spdlog/spdlog.h>

#include <cstring>
#include <memory>
#include <filesystem>
#include <span>

// @TODO: errors

int main(int argc, char** argv)
{
  std::filesystem::path source = {};
  std::filesystem::path output = {};
  std::filesystem::path depfile = {};
  std::string entryPoint = {};
  SlangCompiler::CreateInfo slangCi{};

  CommandlineParser cmdParser{CommandlineParser::CreateInfo{
    .args =
      {
        {.name = {},
         .kind = CommandlineArgumentKind::STRING,
         .desc = "names source file path to compile",
         .valueCb =
           [&](std::string_view path) {
             if (!source.empty())
             {
               spdlog::error(
                 "Invalid usage: can't specify more than one input file, trying '{}', already "
                 "specified '{}'",
                 path,
                 source.string());
               return false;
             }
             source = path;
             return true;
           }},
        {.name = {"-I"},
         .kind = CommandlineArgumentKind::STRING,
         .desc = "adds an include search path",
         .valueCb =
           [&](std::string_view dir) {
             slangCi.commonIncludeDirs.emplace_back(dir);
             return true;
           }},
        {.name = {"-g"},
         .kind = CommandlineArgumentKind::FLAG,
         .desc = "enables debug info embedding in spirv",
         .cb =
           [&] {
             slangCi.emitSpirvDebugInfo = true;
             return true;
           }},
        {.name = {"-o"},
         .kind = CommandlineArgumentKind::STRING,
         .desc = "specifies output spirv file path",
         .valueCb =
           [&](std::string_view path) {
             if (!output.empty())
             {
               spdlog::error(
                 "Invalid usage: can't specify more than one output, trying '{}', already "
                 "specified '{}'",
                 path,
                 depfile.string());
               return false;
             }
             output = path;
             return true;
           }},
        {.name = {"-df", "--depfile"},
         .kind = CommandlineArgumentKind::STRING,
         .desc = "specifies output path for (c)make dependency file",
         .valueCb =
           [&](std::string_view path) {
             if (!depfile.empty())
             {
               spdlog::error(
                 "Invalid usage: can't specify more than one depfile, trying '{}', already "
                 "specified '{}'",
                 path,
                 depfile.string());
               return false;
             }
             depfile = path;
             return true;
           }},
        {.name = {"-e", "--entry-point"},
         .kind = CommandlineArgumentKind::STRING,
         .desc = "specifies entry point name for compiling the shader",
         .valueCb =
           [&](std::string_view name) {
             if (!entryPoint.empty())
             {
               spdlog::error(
                 "Invalid usage: can't specify more than one entry point, trying '{}', already "
                 "specified '{}'",
                 name,
                 entryPoint);
               return false;
             }
             entryPoint = name;
             return true;
           }},
      },
    .helpArg = {"--help", "-h"},
    .programName = "etna-slangc"}};

  std::span<char*> options{argv + 1, size_t(argc - 1)};

  if (int errc = cmdParser.parse(options); errc != 0)
    return errc;

  if (source.empty())
  {
    cmdParser.reportUsageError("missing input file path");
    return 1;
  }
  else if (output.empty())
  {
    cmdParser.reportUsageError("missing output file path (-o)");
    return 1;
  }
  else if (entryPoint.empty())
  {
    cmdParser.reportUsageError("missing entry point (-e/--entry-point)");
    return 1;
  }

  return SlangCompiler{slangCi}.compile(source, entryPoint, output, depfile);
}
