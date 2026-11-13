#pragma once
#ifndef ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
#define ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED

#include "SlangShaderFormat.hpp"
#include "DescriptorSetLayout.hpp"
#include "DescriptorSet.hpp"

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
  uint64_t flatArraySize = 1;

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

class ParameterSetCache;
struct ShaderProgramInfo;

struct DsetBindingDesc
{
  ParameterSetId parsetId = ParameterSetId::INVALID;
  DescriptorLayoutId dsetLayoutId = -1;
  std::vector<Binding> bindings;
};

// @TODO: this is more properly named "binding builder" as it's valid for updates as well
class DsetBindingBuilder
{
  const ParameterSetCache* cache = nullptr;
  DsetBindingBuilder* parent = nullptr;
  ParameterSetId parsetId = ParameterSetId::INVALID;
  TypeId typeId = TypeId::INVALID;
  uint32_t offset = 0;
  uint32_t arrayOffset = 0;
  std::vector<Binding> builtBindings{};

public:
  DsetBindingBuilder& setTex(
    std::string_view name,
    ImageBinding&& value,
    std::initializer_list<uint32_t> local_array_indices = {});
  DsetBindingBuilder& setSmp(
    std::string_view name,
    SamplerBinding&& value,
    std::initializer_list<uint32_t> local_array_indices = {});
  DsetBindingBuilder& setBuf(
    std::string_view name,
    BufferBinding&& value,
    std::initializer_list<uint32_t> local_array_indices = {});
  DsetBindingBuilder& setCbuf(
    std::string_view name,
    BufferBinding&& value,
    std::initializer_list<uint32_t> local_array_indices = {});

  DsetBindingBuilder subtype(
    std::string_view name, std::initializer_list<uint32_t> local_array_indices = {});
  DsetBindingBuilder& end();

  DsetBindingDesc build();

private:
  friend struct ShaderProgramInfo;

  DsetBindingBuilder() = default;
  DsetBindingBuilder(const ParameterSetCache& cache, ParameterSetId parset_id);
  DsetBindingBuilder(
    DsetBindingBuilder& parent, TypeId type_id, uint32_t offset, uint32_t array_offset);

  uint32_t buildFlatArrayIndex(
    std::initializer_list<uint32_t> indices, const Field& field, std::string_view name) const;
};

class BindlessDsetBindingBuilder
{
  const ParameterSetCache* cache = nullptr;
  ParameterSetId parsetId = ParameterSetId::INVALID;
  TypeId typeId = TypeId::INVALID;
  std::vector<Binding> builtBindings{};

public:
  BindlessDsetBindingBuilder& setTex(uint32_t index, ImageBinding&& value);
  BindlessDsetBindingBuilder& setSmp(uint32_t index, SamplerBinding&& value);
  BindlessDsetBindingBuilder& setBuf(uint32_t index, BufferBinding&& value);

  DsetBindingDesc build();

private:
  friend class ParameterSetCache;

  BindlessDsetBindingBuilder() = default;
  BindlessDsetBindingBuilder(const ParameterSetCache& cache, ParameterSetId parset_id);
};

class ParameterSetCache
{
  std::vector<Type> types;
  std::vector<uint64_t> typeHashes;
  std::vector<ParameterSet> parameterSets;
  ParameterSetId bindlessParsetId = ParameterSetId::INVALID;
  TypeId bindlessTypeId = TypeId::INVALID;

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
  const ResourceDesc& getResourceDesc(ParameterSetId parset_id, uint32_t offset) const
  {
    return parameterSets[size_t(parset_id)].resources[offset];
  }

  const Field& getTypeField(TypeId type_id, std::string_view name) const
  {
    const auto& type = types[size_t(type_id)];
    auto it = type.fields.find(std::string{name});
    ETNA_VERIFYF(it != type.fields.end(), "{} is not a field of type {}", name, size_t(type_id));
    return it->second;
  }

  BindlessDsetBindingBuilder bindlessDsetBuilder() const;
};

} // namespace etna

#endif // ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
