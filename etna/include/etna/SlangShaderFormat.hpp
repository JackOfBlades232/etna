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
  TEXTURE,
  SAMPLER,
  BUFFER,
};

enum class SlangShaderResourceAccess : uint64_t
{
  RW,
  RO
};

struct SlangShaderResourceDesc
{
  SlangShaderResourceType type;
  SlangShaderResourceAccess access;

  bool operator==(const SlangShaderResourceDesc&) const = default;
};

struct SlangShaderResourceBinding
{
  uint64_t binding;
  uint64_t arraySize; // UINT64_MAX -- unbounded

  bool operator==(const SlangShaderResourceBinding&) const = default;
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

struct SlangShaderResolutionPath : SlangShaderString
{
};

struct SlangShaderConstbufferReflection
{
  SlangShaderResolutionPath name;
  SlangShaderResourceBinding binding;
  uint64_t totalByteCount;
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

struct SlangShaderPushConstantsReflection
{
  SlangShaderResolutionPath name;
  uint64_t totalByteCount;
};

struct SlangShaderReflectionHeader
{
  SlangShaderTypedArrayRef<SlangShaderDescriptorSetReflection> descriptorSets;
  SlangShaderPushConstantsReflection pushConstants;
  uint64_t numThreadsX;
  uint64_t numThreadsY;
  uint64_t numThreadsZ;
};

struct SlangShaderModuleRef
{
  SlangShaderBytecodeRef bytecode;
  SlangShaderString entryPoint;
  uint64_t declaresPushConsts;
};

struct SlangShaderHeader
{
  uint64_t magic;
  uint64_t version;
  SlangShaderReflectionHeader refl;
  SlangShaderModuleRef vert;
  SlangShaderModuleRef frag;
  SlangShaderModuleRef tesc;
  SlangShaderModuleRef tese;
  SlangShaderModuleRef geom;
  SlangShaderModuleRef comp;
};

inline bool slang_shader_ref_has_value(const auto& ref)
{
  // byte offset 0 always points to magic => is never a valid ref
  // so we can use ZII without having to shift the offset
  return ref.byteOffset != 0;
}

} // namespace etna

#endif // ETNA_SLANG_SHADER_FORMAT_HPP_INCLUDED
