#pragma once
#ifndef ETNA_ASSERT_HPP_INCLUDED
#define ETNA_ASSERT_HPP_INCLUDED

#include <stdexcept>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>


namespace etna
{

struct SourceLocation
{
  std::string_view file;
  uint32_t line;
};

[[noreturn]] inline void panic(SourceLocation loc, std::string message)
{
  spdlog::critical("Panicked at {}:{}, {}", loc.file, loc.line, message);
  std::terminate();
}
  
}

#define ETNA_CURRENT_LOCATION etna::SourceLocation{__FILE__, __LINE__}


#define ETNA_PANIC(fmtStr, ...) \
  etna::panic(ETNA_CURRENT_LOCATION, fmt::format(fmtStr, ##__VA_ARGS__))

#define ETNA_ASSERTF(expr, fmtStr, ...)           \
do                                                \
{                                                 \
  if (!static_cast<bool>((expr)))                 \
  {                                               \
    ETNA_PANIC("assertion '{}' failed: {}",       \
      #expr, fmt::format(fmtStr, ##__VA_ARGS__)); \
  }                                               \
}                                                 \
while (0)


#define ETNA_ASSERT(expr)                       \
do                                              \
{                                               \
  if (!static_cast<bool>((expr)))               \
  {                                             \
    ETNA_PANIC("assertion '{}' failed.", #expr);\
  }                                             \
}                                               \
while (0)

#define ETNA_VK_ASSERTF(expr, fmtStr, ...) ETNA_ASSERTF(expr == vk::Result::eSuccess, fmtStr, __VA_ARGS__)
#define ETNA_VK_ASSERT(expr) ETNA_ASSERT((vk::Result)expr == vk::Result::eSuccess)


namespace etna
{

template <typename T>
inline T validate_vk_result(vk::ResultValue<T> res)
{
  ETNA_VK_ASSERT(res.result);
  return res.value;
}

}

// NOTE: unsanitary macro for customizing vulkan.hpp
// Do NOT use in app code!
#define ETNA_VULKAN_HPP_ASSERT_ON_RESULT(expr) ETNA_ASSERTF(expr, "{}", message)

#endif // ETNA_ASSERT_HPP_INCLUDED
