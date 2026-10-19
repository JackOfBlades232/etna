#pragma once
#ifndef ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
#define ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED

#include "SlangShaderFormat.hpp"

#include <unordered_map>
#include <vector>
#include <variant>
#include <memory>
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

  bool operator==(const Field &) const = default;
};

struct Type
{
  std::unordered_map<std::string, Field> fields;

  bool operator==(const Type &) const = default;
};

struct ResourceDesc
{
  SlangShaderResourceDesc desc;
  SlangShaderResourceBinding binding;
};

struct ParameterSet
{
  TypeId typeId = TypeId::INVALID;
  std::vector<ResourceDesc> resources;
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

  ParameterSetId parseRootParameterSet(
    const SlangShaderDescriptorSetReflection& dset,
    const char* blob,
    std::span<const TypeId> typeid_remapping);
  std::vector<TypeId> addTypes(const SlangShaderTypeDatabase& db, const char* blob);
};


} // namespace etna

#endif // ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
