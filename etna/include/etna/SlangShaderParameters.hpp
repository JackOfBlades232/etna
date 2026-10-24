#pragma once
#ifndef ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
#define ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED

#include "SlangShaderFormat.hpp"
#include "DescriptorSetLayout.hpp"

#include <unordered_map>
#include <vector>
#include <variant>
#include <memory>
#include <string>
#include <span>

namespace etna
{

// @TODO: better abstraction than unordered_map and vector for dims

enum class TypeId : uint32_t
{
  INVALID = 0,
};

enum class ParameterSetId : uint32_t
{
  INVALID = 0,
};

using ResourceOffset = uint32_t;

struct Field
{
  ResourceOffset resourceOffset = 0;
  TypeId subtypeId = TypeId::INVALID;
  std::vector<uint64_t> localArrayDims;

  bool operator==(const Field&) const = default;
};

struct Type
{
  std::unordered_map<std::string, Field> fields;

  bool operator==(const Type&) const = default;
};

struct ResourceDesc
{
  SlangShaderResourceDesc desc;
  SlangShaderResourceBinding binding;

  bool operator==(const ResourceDesc&) const = default;
};

struct ParameterSet
{
  TypeId typeId = TypeId::INVALID;
  DescriptorLayoutId dsetLayoutId = -1;
  std::vector<ResourceDesc> resources{};
  vk::ShaderStageFlags stageMask = {};
};

class ParameterSetCache
{
  std::vector<Type> types;
  std::vector<uint64_t> typeHashes;
  std::vector<ParameterSet> parameterSets;
  std::unordered_map<std::string, ParameterSetId> parameterSetMap;

public:
  ParameterSetCache()
  {
    types.emplace_back();
    parameterSets.emplace_back();
  }

  struct RootParameterSetParseResult
  {
    ParameterSetId parsetId;
    DescriptorSetInfo dsInfo;
  };

  RootParameterSetParseResult parseRootParameterSet(
    std::span<const SlangShaderResourceReflection> resources,
    std::string_view name,
    TypeId type_id,
    vk::ShaderStageFlags stage_mask,
    DescriptorSetLayoutCache& layout_cache);
  std::vector<TypeId> addTypes(const SlangShaderTypeDatabase& db, const char* blob);

  TypeId getRootTypeId(ParameterSetId parset_id) const
  {
    return parameterSets[size_t(parset_id)].typeId;
  }
  DescriptorLayoutId getRootDsetLayoutId(ParameterSetId parset_id) const
  {
    return parameterSets[size_t(parset_id)].dsetLayoutId;
  }
  vk::ShaderStageFlags getRootStageMask(ParameterSetId parset_id) const
  {
    return parameterSets[size_t(parset_id)].stageMask;
  }
};


} // namespace etna

#endif // ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
