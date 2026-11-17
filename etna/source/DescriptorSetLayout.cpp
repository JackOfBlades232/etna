#include <etna/DescriptorSetLayout.hpp>
#include <etna/DescriptorSet.hpp>

#include <spirv_reflect.h>

#include <etna/Assert.hpp>
#include <vulkan/vulkan_enums.hpp>


namespace etna
{

static bool is_dynamic_descriptor(vk::DescriptorType type)
{
  return type == vk::DescriptorType::eUniformBufferDynamic ||
    type == vk::DescriptorType::eStorageBufferDynamic;
}

void DescriptorSetInfo::addResource(
  const vk::DescriptorSetLayoutBinding& binding, vk::DescriptorBindingFlags flags)
{
  if (binding.binding > MAX_DESCRIPTOR_BINDINGS)
    ETNA_PANIC(
      "DescriptorSetInfo: Binding {} out of MAX_DESCRIPTOR_BINDINGS range", binding.binding);

  if (usedBindings.test(binding.binding))
  {
    auto& src = bindings[binding.binding];
    if (
      src.descriptorType != binding.descriptorType ||
      src.descriptorCount != binding.descriptorCount)
    {
      ETNA_PANIC("DescriptorSetInfo: incompatible bindings at index {}", binding.binding);
    }

    src.stageFlags |= binding.stageFlags;
    bindingFlags[binding.binding] |= flags;
    return;
  }

  usedBindings.set(binding.binding);
  bindings[binding.binding] = binding;
  bindingFlags[binding.binding] = flags;

  if (binding.binding + 1 > usedBindingsCap)
  {
    if (flags & vk::DescriptorBindingFlagBits::ePartiallyBound)
    {
      bindingFlags[binding.binding] |= vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
      if (usedBindingsCap > 0)
        bindingFlags[getMaxBinding()] &= ~vk::DescriptorBindingFlagBits::eVariableDescriptorCount;
      hasDynDescriptorArrayAsLastBinding = true;
    }
    else
    {
      hasDynDescriptorArrayAsLastBinding = false;
    }
    usedBindingsCap = binding.binding + 1;
  }

  if (is_dynamic_descriptor(binding.descriptorType))
    dynOffsets++;
}

void DescriptorSetInfo::clear()
{
  usedBindingsCap = 0;
  dynOffsets = 0;
  usedBindings.reset();
  for (auto& binding : bindings)
    binding = vk::DescriptorSetLayoutBinding{};
  for (auto& flags : bindingFlags)
    flags = vk::DescriptorBindingFlags{};
}

void DescriptorSetInfo::parseShader(
  vk::ShaderStageFlagBits stage, const SpvReflectDescriptorSet& spv)
{
  for (uint32_t i = 0u; i < spv.binding_count; i++)
  {
    const auto& spvBinding = *spv.bindings[i];

    vk::DescriptorSetLayoutBinding apiBinding{};
    vk::DescriptorBindingFlags apiFlags{};
    apiBinding.descriptorCount = 1;

    for (uint32_t j = 0; j < spvBinding.array.dims_count; j++)
    {
      apiBinding.descriptorCount *= spvBinding.array.dims[j];
    }

    apiBinding.descriptorType = static_cast<vk::DescriptorType>(spvBinding.descriptor_type);
    apiBinding.stageFlags = stage;
    apiBinding.pImmutableSamplers = nullptr;
    apiBinding.binding = spvBinding.binding;

    if (apiBinding.descriptorCount == SPV_REFLECT_ARRAY_DIM_RUNTIME)
    {
      apiBinding.descriptorCount = get_num_descriptors_in_pool_for_type(apiBinding.descriptorType);
      apiFlags |= vk::DescriptorBindingFlagBits::ePartiallyBound;
    }

    addResource(apiBinding, apiFlags);
  }
}

static vk::DescriptorType decriptor_type_from_desc(const SlangShaderResourceDesc& desc)
{
  switch (desc.type)
  {
  case SlangShaderResourceType::COMBINED_TEXTURE_SAMPLER:
    return vk::DescriptorType::eCombinedImageSampler;
  case SlangShaderResourceType::TEXTURE:
    return desc.access == SlangShaderResourceAccess::RO ? vk::DescriptorType::eSampledImage
                                                        : vk::DescriptorType::eStorageImage;
  case SlangShaderResourceType::SAMPLER:
    return vk::DescriptorType::eSampler;
  case SlangShaderResourceType::BUFFER:
    return vk::DescriptorType::eStorageBuffer;
  case SlangShaderResourceType::CONSTBUFFER:
    return vk::DescriptorType::eUniformBuffer;
  }
}

void DescriptorSetInfo::parseReflection(
  vk::ShaderStageFlags stage_mask, std::span<const SlangShaderResourceReflection> resources)
{
  auto processDynArrayBinding = [](auto& api_bind, auto& api_flags) {
    if (api_bind.descriptorCount == UINT32_MAX)
    {
      api_bind.descriptorCount = get_num_descriptors_in_pool_for_type(api_bind.descriptorType);
      api_flags |= vk::DescriptorBindingFlagBits::ePartiallyBound;
    }
  };

  for (const auto& res : resources)
  {
    vk::DescriptorSetLayoutBinding apiBinding{};
    vk::DescriptorBindingFlags apiFlags{};
    apiBinding.descriptorCount = res.binding.arraySize;
    apiBinding.descriptorType = decriptor_type_from_desc(res.desc);
    apiBinding.stageFlags = stage_mask;
    apiBinding.pImmutableSamplers = nullptr;
    apiBinding.binding = res.binding.binding;
    processDynArrayBinding(apiBinding, apiFlags);
    addResource(apiBinding, apiFlags);
  }
}

void DescriptorSetInfo::merge(const DescriptorSetInfo& info)
{
  for (uint32_t binding = 0; binding < info.usedBindingsCap; binding++)
  {
    if (!info.usedBindings.test(binding))
      continue;
    addResource(info.bindings[binding], info.bindingFlags[binding]);
  }
}

bool DescriptorSetInfo::operator==(const DescriptorSetInfo& rhs) const
{
  if (usedBindingsCap != rhs.usedBindingsCap)
    return false;
  if (usedBindings != rhs.usedBindings)
    return false;
  if (hasDynDescriptorArrayAsLastBinding != rhs.hasDynDescriptorArrayAsLastBinding)
    return false;

  for (uint32_t i = 0; i < usedBindingsCap; i++)
  {
    if (bindings[i] != rhs.bindings[i])
      return false;
    if (bindingFlags[i] != rhs.bindingFlags[i])
      return false;
  }

  return true;
}

vk::DescriptorSetLayout DescriptorSetInfo::createVkLayout(vk::Device device) const
{
  std::vector<vk::DescriptorSetLayoutBinding> apiBindings;
  std::vector<vk::DescriptorBindingFlags> apiFlags;
  for (uint32_t i = 0; i < usedBindingsCap; i++)
  {
    if (!usedBindings.test(i))
      continue;
    apiBindings.push_back(bindings[i]);
    apiFlags.push_back(bindingFlags[i]);
  }

  vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{};
  flagsInfo.setBindingFlags(apiFlags);

  vk::DescriptorSetLayoutCreateInfo info{};
  info.setBindings(apiBindings);
  info.setPNext(&flagsInfo);

  return unwrap_vk_result(device.createDescriptorSetLayout(info));
}

template <typename T>
inline void hash_combine(std::size_t& s, const T& v)
{
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

std::size_t DescriptorSetLayoutHash::operator()(const DescriptorSetInfo& res) const
{
  size_t hash = 0;

  hash_combine(hash, res.hasDynDescriptorArrayAsLastBinding);

  for (uint32_t i = 0; i < res.usedBindingsCap; i++)
  {
    if (!res.usedBindings.test(i))
      continue;

    hash_combine(hash, res.bindings[i].binding);
    hash_combine(hash, static_cast<uint32_t>(res.bindings[i].descriptorType));
    hash_combine(hash, res.bindings[i].descriptorCount);
    hash_combine(hash, static_cast<uint32_t>(res.bindings[i].stageFlags));
    hash_combine(hash, static_cast<uint32_t>(res.bindingFlags[i]));
  }

  return hash;
}

DescriptorLayoutId DescriptorSetLayoutCache::registerLayout(
  vk::Device device, const DescriptorSetInfo& info)
{
  return get(device, info).first;
}

std::pair<DescriptorLayoutId, vk::DescriptorSetLayout> DescriptorSetLayoutCache::get(
  vk::Device device, const DescriptorSetInfo& info)
{
  auto it = map.find(info);
  if (it != map.end())
    return {it->second, vkLayouts[it->second]};

  DescriptorLayoutId id = static_cast<DescriptorLayoutId>(descriptors.size());
  map.insert({info, id});
  descriptors.push_back(info);
  vkLayouts.push_back(info.createVkLayout(device));
  return {id, vkLayouts[id]};
}

void DescriptorSetLayoutCache::clear(vk::Device device)
{
  for (auto layout : vkLayouts)
  {
    device.destroyDescriptorSetLayout(layout);
  }

  map.clear();
  descriptors.clear();
  vkLayouts.clear();
}

} // namespace etna
