#pragma once
#ifndef ETNA_COMMON_HPP_INCLUDED
#define ETNA_COMMON_HPP_INCLUDED

#include <string>

template <class TS>
  requires(std::same_as<TS, std::string> || std::same_as<TS, std::wstring>)
std::string to_char_str(const TS& s)
{
  if constexpr (std::same_as<TS, std::string>)
    return s;
  else
    return std::to_string(s);
}

template <class T>
inline T align_up_pot(T val, uint32_t alignment)
{
  return (val + T{alignment - 1u}) & ~T{alignment - 1u};
}

#endif

