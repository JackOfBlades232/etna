#pragma once
#ifndef ETNA_SLANG_RUNTIME_HPP_INCLUDED
#define ETNA_SLANG_RUNTIME_HPP_INCLUDED

#include <slang/slang.h>
#include <slang/slang-com-ptr.h>
#include <slang/slang-com-helper.h>

#include <string>
#include <filesystem>

// @TODO: conditional inclusion of the whole slang thing in the binary

class SlangCompiler
{
  Slang::ComPtr<slang::IGlobalSession> globalSession{};
  Slang::ComPtr<slang::ISession> session{};
  void* dynlibHandle = nullptr;

public:
  struct CreateInfo
  {
    std::vector<std::filesystem::path> commonIncludeDirs = {};
    bool emitSpirvDebugInfo = false;
  };

  explicit SlangCompiler(const CreateInfo& ci);
  // @TODO: semantics

  ~SlangCompiler();

  int compile(
    const std::filesystem::path& source,
    std::string_view target,
    const std::filesystem::path& dest,
    const std::filesystem::path& dest_depfile = {});
};

#endif
