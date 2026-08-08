#pragma once
#ifndef ETNA_SLANG_RUNTIME_HPP_INCLUDED
#define ETNA_SLANG_RUNTIME_HPP_INCLUDED

#include <slang/slang.h>
#include <slang/slang-com-ptr.h>

// @TODO: conditional inclusion of the whole slang thing in the binary

namespace etna
{

class SlangRuntime
{
  Slang::ComPtr<slang::IGlobalSession> globalSession{};
  Slang::ComPtr<slang::ISession> session{};
  void* dynlibHandle = nullptr;

public:
  struct CreateInfo
  {
  };

  explicit SlangRuntime(CreateInfo&& ci);
  // @TODO: semantics

  ~SlangRuntime();
};

} // namespace etna

#endif
