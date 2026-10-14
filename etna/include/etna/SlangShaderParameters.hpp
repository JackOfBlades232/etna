#pragma once
#ifndef ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
#define ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED

#include "SlangShaderFormat.hpp"

#include <unordered_map>
#include <vector>
#include <variant>
#include <memory>

namespace etna
{

struct ConstbufferDesc
{
  SlangShaderResourceBinding binding;
  uint64_t totalByteCount;
};

struct ResourceDesc
{
  SlangShaderResourceDesc desc;
  SlangShaderResourceBinding binding;
};

enum class ParameterSetId : uint32_t
{
  INVALID = 0,
};

struct Parameter : std::variant<ConstbufferDesc, ResourceDesc, ParameterSetId>
{
  uint32_t localArraySize = 0;
};

struct ParameterSet : std::unordered_map<std::string, Parameter>
{
};

struct ParameterSetCache
{
  std::vector<ParameterSet> parameterSets;
  std::unordered_map<std::string, ParameterSetId> rootParameterSets;

  ParameterSetId parseRootParameterSet(
    const SlangShaderDescriptorSetReflection& refl, const char* blob);
};


} // namespace etna

#endif // ETNA_SLANG_SHADER_PARAMETERS_HPP_INCLUDED
