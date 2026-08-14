#include "Slang.hpp"

#include <spdlog/spdlog.h>

#include <memory>

int main(int argc, char** argv)
{
  (void)argc;
  (void)argv;

  auto slang = std::make_unique<SlangRuntime>(SlangRuntime::CreateInfo{});
  
  spdlog::info("Hello from etna-slangc!");

  return 0;
}
