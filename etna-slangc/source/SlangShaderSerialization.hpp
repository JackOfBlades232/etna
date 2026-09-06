#pragma once
#ifndef ETNA_SLANG_SHADER_SERIALIZATION_HPP_INCLUDED
#define ETNA_SLANG_SHADER_SERIALIZATION_HPP_INCLUDED

#include "Common.hpp"

#include <etna/SlangShaderFormat.hpp>

#include <cstdio>
#include <span>
#include <deque>
#include <filesystem>

class SlangShaderSerializationContext : public std::vector<std::vector<uint8_t>>
{
  uint64_t totalByteSize = 0;

public:
  template <class T>
  T* ref(auto r)
  {
    uint64_t off = r.byteOffset;
    for (auto& chunk : *this)
    {
      if (off < chunk.size())
        return reinterpret_cast<T*>(&chunk[off]);
      off -= chunk.size();
    }
    return nullptr;
  }

  uint64_t totalBytes() const { return totalByteSize; }
  auto& pushChunk(uint64_t size)
  {
    auto& chunk = emplace_back();
    chunk.resize(size);
    totalByteSize += size;
    return chunk;
  }
};

inline etna::SlangShaderBytecodeRef slang_shader_ser_write_bytecode(
  std::span<const uint8_t> bytecode, SlangShaderSerializationContext& data)
{
  etna::SlangShaderBytecodeRef ref = {data.totalBytes(), bytecode.size()};
  auto alignedSize = align_up_pot(bytecode.size(), sizeof(uint64_t));
  auto& chunk = data.pushChunk(alignedSize);
  auto dataEnd = std::copy(bytecode.begin(), bytecode.end(), chunk.begin());
  std::fill_n(dataEnd, alignedSize - bytecode.size(), 0);
  return ref;
}

template <class T>
  requires(sizeof(T) % sizeof(uint64_t) == 0 && alignof(T) <= sizeof(uint64_t))
etna::SlangShaderTypedArrayRef<T> slang_shader_ser_write_typed_array(
  std::span<const T> array, SlangShaderSerializationContext& data)
{
  etna::SlangShaderTypedArrayRef<T> ref = {{data.totalBytes()}, array.size()};
  auto& chunk = data.pushChunk(array.size() * sizeof(T));
  std::copy(array.begin(), array.end(), reinterpret_cast<T*>(chunk.data()));
  return ref;
}

template <class T>
  requires(sizeof(T) % sizeof(uint64_t) == 0 && alignof(T) <= sizeof(uint64_t))
etna::SlangShaderTypedRef<T> slang_shader_ser_write_type(
  const T& obj, SlangShaderSerializationContext& data)
{
  return slang_shader_ser_write_typed_array<T>({&obj, 1}, data);
}

bool slang_shader_ser_write_to_file(
  const SlangShaderSerializationContext& data, std::filesystem::path dest, std::string& error);

#endif
