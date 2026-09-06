#pragma once
#ifndef ETNA_SLANG_SHADER_FORMAT_HPP_INCLUDED
#define ETNA_SLANG_SHADER_FORMAT_HPP_INCLUDED

#include <bit>

namespace etna
{

static_assert(std::endian::native == std::endian::little);

// @TODO: more proper values
inline constexpr uint64_t SLANG_SHADER_MAGIC = 'SLNg';
inline constexpr uint64_t SLANG_SHADER_VERSION = '0001';
inline constexpr uint64_t SLANG_DEFAULT_ENTITY_SLOT = 0;

// @TODO: extend/contract to a more proper and useful desc
enum class SlangShaderResourceType : uint64_t
{
  COMBINED_TEXTURE_SAMPLER,
  RO_TEXTURE,
  SAMPLER,
  RW_TEXTURE,
  RO_BUFFER,
  RW_BUFFER,
};

struct SlangShaderResourceDesc
{
  SlangShaderResourceType type;
};

struct SlangShaderResourceBinding
{
  uint64_t binding;
  uint64_t arraySize; // UINT64_MAX -- unbounded
};

struct SlangShaderBytecodeRef
{
  uint64_t byteOffset;
  uint64_t byteCount;
};

template <class T>
struct SlangShaderTypedRef
{
  uint64_t byteOffset;
};

template <class T>
struct SlangShaderTypedArrayRef : SlangShaderTypedRef<T>
{
  uint64_t elemCount;
};

struct SlangShaderString
{
  uint64_t byteOffset;
  uint64_t charCount;
};

struct SlangShaderResolutionPath
{
  SlangShaderString leaf;
  SlangShaderTypedRef<SlangShaderResolutionPath> stem;
};

struct SlangShaderCbufMemberReflection
{
  SlangShaderResolutionPath name;
  uint64_t byteOffset;
  uint64_t elementByteSize;
  uint64_t arrayElementCount;
};

struct SlangShaderConstbufferReflection
{
  SlangShaderResolutionPath name;
  SlangShaderTypedArrayRef<SlangShaderCbufMemberReflection> members;
  SlangShaderResourceBinding binding;
};

struct SlangShaderResourceReflection
{
  SlangShaderResolutionPath name;
  SlangShaderResourceDesc desc;
  SlangShaderResourceBinding binding;
};

struct SlangShaderDescriptorSetReflection
{
  SlangShaderResolutionPath name;
  SlangShaderTypedArrayRef<SlangShaderConstbufferReflection> constBuffers;
  SlangShaderTypedArrayRef<SlangShaderResourceReflection> resources;
  uint64_t index;
};

struct SlangShaderReflectionHeader
{
  SlangShaderTypedArrayRef<SlangShaderDescriptorSetReflection> descriptorSets;
  uint64_t numThreadsX;
  uint64_t numThreadsY;
  uint64_t numThreadsZ;
};

struct SlangShaderHeader
{
  uint64_t magic;
  uint64_t version;
  SlangShaderReflectionHeader refl;
  SlangShaderBytecodeRef vert;
  SlangShaderBytecodeRef frag;
  SlangShaderBytecodeRef tesc;
  SlangShaderBytecodeRef tese;
  SlangShaderBytecodeRef geom;
  SlangShaderBytecodeRef comp;
};

inline bool slang_shader_ref_has_value(const auto& ref)
{
  // byte offset 0 always points to magic => is never a valid ref
  // so we can use ZII without having to shift the offset
  return ref.byteOffset != 0;
}

} // namespace etna

#endif // ETNA_SLANG_SHADER_FORMAT_HPP_INCLUDED
