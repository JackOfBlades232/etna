#include <etna/SlangShaderParameters.hpp>

#include <etna/GlobalContext.hpp>
#include <etna/Assert.hpp>

#include <span>

namespace etna
{

// @TODO: common header
template <class T>
static std::span<const T> rebase_ser_array(const SlangShaderTypedArrayRef<T>& ref, const char* blob)
{
  return std::span<const T>{(const T*)(blob + ref.byteOffset), ref.elemCount};
}

static std::string_view rebase_ser_string(const SlangShaderString& ref, const char* blob)
{
  return std::string_view{(const char*)(blob + ref.byteOffset), ref.charCount};
}

uint32_t DsetBindingBuilder::buildFlatArrayIndex(
  std::initializer_list<uint32_t> indices, const Field& field, std::string_view name) const
{
  if (field.localArrayDims.empty())
  {
    ETNA_ASSERTF(indices.size() == 0, "\"{}\" is not an array field", name);
    return arrayOffset;
  }
  else
  {
    ETNA_ASSERTF(
      field.localArrayDims.size() == indices.size(),
      "\"{}\" is an array field of different dimensionality {}",
      name,
      field.localArrayDims.size());
  }
  uint32_t flat = 0;
  for (size_t i = 0; uint32_t id : indices)
  {
    flat += id;
    if (i < field.localArrayDims.size() - 1)
      flat *= uint32_t(field.localArrayDims[i + 1]);
  }
  return arrayOffset * uint32_t(field.flatArraySize) + flat;
}

DsetBindingBuilder& DsetBindingBuilder::setTex(
  std::string_view name, ImageBinding&& value, std::initializer_list<uint32_t> local_array_indices)
{
  const auto& field = cache->getTypeField(typeId, name);
  ETNA_ASSERTF(
    field.subtypeId == TypeId::INVALID, "\"{}\" is a subtype field, not a texture", name);
  const auto& desc = cache->getResourceDesc(parsetId, offset + field.resourceOffset);
  if (value.descriptor_info.sampler)
  {
    ETNA_ASSERTF(
      desc.desc.type == SlangShaderResourceType::COMBINED_TEXTURE_SAMPLER,
      "\"{}\" is not a combined image sampler",
      name);
  }
  else
  {
    ETNA_ASSERTF(
      desc.desc.type == SlangShaderResourceType::TEXTURE, "\"{}\" is not a texture", name);
  }
  ETNA_ASSERT(arrayOffset < desc.binding.arraySize);
  builtBindings.emplace_back(
    desc.binding.binding, std::move(value), buildFlatArrayIndex(local_array_indices, field, name));
  return *this;
}

DsetBindingBuilder& DsetBindingBuilder::setSmp(
  std::string_view name,
  SamplerBinding&& value,
  std::initializer_list<uint32_t> local_array_indices)
{
  const auto& field = cache->getTypeField(typeId, name);
  ETNA_ASSERTF(
    field.subtypeId == TypeId::INVALID, "\"{}\" is a subtype field, not a sampler", name);
  ETNA_ASSERTF(field.localArrayDims.empty(), "\"{}\" is an array field", name);
  const auto& desc = cache->getResourceDesc(parsetId, offset + field.resourceOffset);
  ETNA_ASSERTF(desc.desc.type == SlangShaderResourceType::SAMPLER, "\"{}\" is not a sampler", name);
  ETNA_ASSERT(desc.desc.access == SlangShaderResourceAccess::RO);
  ETNA_ASSERT(arrayOffset < desc.binding.arraySize);
  builtBindings.emplace_back(
    desc.binding.binding, std::move(value), buildFlatArrayIndex(local_array_indices, field, name));
  return *this;
}

DsetBindingBuilder& DsetBindingBuilder::setBuf(
  std::string_view name, BufferBinding&& value, std::initializer_list<uint32_t> local_array_indices)
{
  const auto& field = cache->getTypeField(typeId, name);
  ETNA_ASSERTF(field.subtypeId == TypeId::INVALID, "\"{}\" is a subtype field, not a buffer", name);
  ETNA_ASSERTF(field.localArrayDims.empty(), "\"{}\" is an array field", name);
  const auto& desc = cache->getResourceDesc(parsetId, offset + field.resourceOffset);
  ETNA_ASSERTF(desc.desc.type == SlangShaderResourceType::BUFFER, "\"{}\" is not a buffer", name);
  ETNA_ASSERT(arrayOffset < desc.binding.arraySize);
  builtBindings.emplace_back(
    desc.binding.binding, std::move(value), buildFlatArrayIndex(local_array_indices, field, name));
  return *this;
}

DsetBindingBuilder& DsetBindingBuilder::setCbuf(
  std::string_view name, BufferBinding&& value, std::initializer_list<uint32_t> local_array_indices)
{
  const auto& field = cache->getTypeField(typeId, name);
  ETNA_ASSERTF(
    field.subtypeId == TypeId::INVALID, "\"{}\" is a subtype field, not a constbuffer", name);
  ETNA_ASSERTF(field.localArrayDims.empty(), "\"{}\" is an array field", name);
  const auto& desc = cache->getResourceDesc(parsetId, offset + field.resourceOffset);
  ETNA_ASSERTF(
    desc.desc.type == SlangShaderResourceType::CONSTBUFFER, "\"{}\" is not a constbuffer", name);
  ETNA_ASSERT(desc.desc.access == SlangShaderResourceAccess::RO);
  ETNA_ASSERT(arrayOffset < desc.binding.arraySize);
  builtBindings.emplace_back(
    desc.binding.binding, std::move(value), buildFlatArrayIndex(local_array_indices, field, name));
  return *this;
}

DsetBindingBuilder DsetBindingBuilder::subtype(
  std::string_view name, std::initializer_list<uint32_t> local_array_indices)
{
  const auto& field = cache->getTypeField(typeId, name);
  ETNA_ASSERTF(field.subtypeId != TypeId::INVALID, "{} is a resource field, not a subtype", name);
  ETNA_ASSERTF(field.localArrayDims.empty(), "{} is an array field", name);
  return {
    *this,
    field.subtypeId,
    offset + field.resourceOffset,
    buildFlatArrayIndex(local_array_indices, field, name)};
}

DsetBindingBuilder& DsetBindingBuilder::end()
{
  ETNA_ASSERT(parent);
  return *parent;
}

DsetBindingDesc DsetBindingBuilder::build()
{
  ETNA_ASSERT(!parent);
  DsetBindingDesc desc{};
  desc.parsetId = parsetId;
  desc.dsetLayoutId = cache->getRootDsetLayoutId(parsetId);
  desc.bindings = std::move(builtBindings); // @TODO: validate completeness?
  *this = {};
  return desc;
}

DsetBindingBuilder::DsetBindingBuilder(const ParameterSetCache& cache, ParameterSetId parset_id)
  : cache{&cache}
  , parsetId{parset_id}
  , typeId{cache.getRootTypeId(parset_id)}
{
}

DsetBindingBuilder::DsetBindingBuilder(
  DsetBindingBuilder& parent, TypeId type_id, uint32_t offset, uint32_t array_offset)
  : cache{parent.cache}
  , parent{&parent}
  , parsetId{parent.parsetId}
  , typeId{type_id}
  , offset{offset}
  , arrayOffset{array_offset}
{
}

BindlessDsetBindingBuilder& BindlessDsetBindingBuilder::setTex(uint32_t index, ImageBinding&& value)
{
  if (value.descriptor_info.sampler)
  {
    const auto& field = cache->getTypeField(typeId, "$CombinedTextureSamplerHeap");
    ETNA_ASSERT(field.subtypeId == TypeId::INVALID);
    const auto& desc = cache->getResourceDesc(parsetId, field.resourceOffset);
    ETNA_ASSERT(desc.desc.type == SlangShaderResourceType::COMBINED_TEXTURE_SAMPLER);
    ETNA_ASSERT(desc.desc.access == SlangShaderResourceAccess::RO);
    builtBindings.emplace_back(desc.binding.binding, std::move(value), uint64_t(index));
  }
  else
  {
    const auto& field = cache->getTypeField(typeId, "$TextureHeap");
    ETNA_ASSERT(field.subtypeId == TypeId::INVALID);
    const auto& desc = cache->getResourceDesc(parsetId, field.resourceOffset);
    ETNA_ASSERT(desc.desc.type == SlangShaderResourceType::TEXTURE);
    ETNA_ASSERT(desc.desc.access == SlangShaderResourceAccess::RO);
    builtBindings.emplace_back(desc.binding.binding, std::move(value), uint64_t(index));
  }
  return *this;
}

BindlessDsetBindingBuilder& BindlessDsetBindingBuilder::setSmp(
  uint32_t index, SamplerBinding&& value)
{
  const auto& field = cache->getTypeField(typeId, "$SamplerHeap");
  ETNA_ASSERT(field.subtypeId == TypeId::INVALID);
  const auto& desc = cache->getResourceDesc(parsetId, field.resourceOffset);
  ETNA_ASSERT(desc.desc.type == SlangShaderResourceType::SAMPLER);
  ETNA_ASSERT(desc.desc.access == SlangShaderResourceAccess::RO);
  builtBindings.emplace_back(desc.binding.binding, std::move(value), uint64_t(index));
  return *this;
}

BindlessDsetBindingBuilder& BindlessDsetBindingBuilder::setBuf(
  uint32_t index, BufferBinding&& value)
{
  const auto& field = cache->getTypeField(typeId, "$BufferHeap"); // Aliased with $RWBufferHeap
  ETNA_ASSERT(field.subtypeId == TypeId::INVALID);
  const auto& desc = cache->getResourceDesc(parsetId, field.resourceOffset);
  ETNA_ASSERT(desc.desc.type == SlangShaderResourceType::BUFFER);
  builtBindings.emplace_back(desc.binding.binding, std::move(value), uint64_t(index));
  return *this;
}

DsetBindingDesc BindlessDsetBindingBuilder::build()
{
  DsetBindingDesc desc{};
  desc.parsetId = parsetId;
  desc.dsetLayoutId = cache->getRootDsetLayoutId(parsetId);
  desc.bindings = std::move(builtBindings); // @TODO: validate completeness?
  *this = {};
  return desc;
}

BindlessDsetBindingBuilder::BindlessDsetBindingBuilder(
  const ParameterSetCache& cache, ParameterSetId parset_id)
  : cache{&cache}
  , parsetId{parset_id}
  , typeId{cache.getRootTypeId(parset_id)}
{
}

static bool compare_loaded_parset(
  std::span<const SlangShaderResourceReflection> resources,
  TypeId type_id,
  vk::ShaderStageFlags stage_mask,
  const ParameterSet& parset)
{
  if (type_id != parset.typeId)
    return false;
  if (resources.size() != parset.resources.size())
    return false;
  bool resourcesEq = true;
  for (size_t rid = 0; rid < resources.size(); ++rid)
  {
    if (
      resources[rid].binding != parset.resources[rid].binding ||
      resources[rid].desc != parset.resources[rid].desc)
    {
      resourcesEq = false;
      break;
    }
  }
  if (!resourcesEq)
    return false;
  return (parset.stageMask & stage_mask) == stage_mask; // subsuming is ok for reuse
}

ParameterSetCache::RootParameterSetParseResult ParameterSetCache::parseRootParameterSet(
  std::span<const SlangShaderResourceReflection> resources,
  TypeId type_id,
  vk::ShaderStageFlags stage_mask,
  DescriptorSetLayoutCache& layout_cache)
{
  auto handleBindless = [&](ParameterSetId assigned_id) {
    if (type_id == bindlessTypeId)
    {
      if (bindlessParsetId == ParameterSetId::INVALID)
        ETNA_VERIFY(assigned_id == bindlessParsetId);
      else
        bindlessParsetId = assigned_id;
    }
  };

  for (size_t i = 0; const auto& parset : parameterSets)
  {
    if (!compare_loaded_parset(resources, type_id, stage_mask, parset))
    {
      ++i;
      continue;
    }

    handleBindless(ParameterSetId(i));
    return {ParameterSetId(i), layout_cache.getLayoutInfo(parset.dsetLayoutId)};
  }

  ParameterSetId retId = ParameterSetId(parameterSets.size());

  auto& parset = parameterSets.emplace_back();
  parset.typeId = type_id;

  DescriptorSetInfo dsInfo;
  if (type_id == bindlessTypeId)
    dsInfo.parseBindlessReflection(resources);
  else
    dsInfo.parseReflection(stage_mask, resources);

  parset.dsetLayoutId = layout_cache.get(get_context().getDevice(), dsInfo).first;

  parset.resources.resize(resources.size());
  for (size_t i = 0; const auto& res : resources)
    parset.resources[i++] = {res.desc, res.binding};

  parset.stageMask = stage_mask;

  handleBindless(retId);
  return {retId, dsInfo};
}

template <class T>
static bool compare_numeric_spans(std::span<const T> s1, std::span<const T> s2)
{
  if (s1.size() != s2.size())
    return false;
  auto i1 = s1.begin(), e1 = s1.end();
  auto i2 = s2.begin();
  while (i1 != e1)
  {
    if (*i1++ != *i2++)
      return false;
  }
  return true;
}

template <class T>
static std::vector<T> vec_from_span(std::span<const T> spn)
{
  std::vector<T> v(spn.size());
  std::copy(spn.begin(), spn.end(), v.begin());
  return v;
}

static bool compare_rt_field_to_loaded_field(
  const Field& rtf,
  const SlangShaderTypeFieldReflection& ldf,
  const char* blob,
  const auto& typeid_remapping)
{
  if (rtf.resourceOffset != ldf.resourceOffset)
    return false;
  if (rtf.subtypeId != typeid_remapping[ldf.subtypeId])
    return false;
  if (!compare_numeric_spans(std::span{rtf.localArrayDims}, rebase_ser_array(ldf.arrayDims, blob)))
    return false;
  return true;
}

static bool compare_rt_type_to_loaded_type(
  const Type& rtt,
  const SlangShaderTypeReflection& ldt,
  const char* blob,
  const auto& typeid_remapping)
{
  if (rtt.fields.size() != ldt.fields.elemCount)
    return false;

  auto ldfs = rebase_ser_array(ldt.fields, blob);
  for (const auto& ldf : ldfs)
  {
    auto name = rebase_ser_string(ldf.name, blob);
    auto it = rtt.fields.find(std::string{name});
    if (it == rtt.fields.end())
      return false;
    if (!compare_rt_field_to_loaded_field(it->second, ldf, blob, typeid_remapping))
      return false;
  }
  return true;
}

std::vector<TypeId> ParameterSetCache::addTypes(const SlangShaderTypeDatabase& db, const char* blob)
{
  // New types are required to be topsorted in the subtype relationship order, so just add
  auto newTypes = rebase_ser_array(db.types, blob);
  auto newHashes = rebase_ser_array(db.typeHashes, blob);
  assert(newHashes.size() == newTypes.size() - 1);
  std::vector<TypeId> remapping(newTypes.size(), TypeId::INVALID);
  auto remap = [&](const auto& type, auto hash) {
    for (size_t i = 0; i < typeHashes.size(); ++i)
    {
      if (hash != typeHashes[i])
        continue;
      if (!compare_rt_type_to_loaded_type(types[i + 1], type, blob, remapping))
        continue;
      return TypeId(i + 1);
    }
    return TypeId(types.size());
  };
  for (size_t i = 0; const auto& type : newTypes.subspan(1))
  {
    auto hash = newHashes[i];
    auto id = remapping[i + 1] = remap(type, hash);
    if (size_t(id) >= types.size())
    {
      auto& rtType = types.emplace_back();
      typeHashes.emplace_back() = hash;

      auto fields = rebase_ser_array(type.fields, blob);
      for (const auto& field : fields)
      {
        auto name = rebase_ser_string(field.name, blob);
        auto dims = rebase_ser_array(field.arrayDims, blob);
        uint32_t flatSize = 1;
        for (auto dim : dims)
          flatSize *= dim;
        rtType.fields.emplace(
          std::string{name},
          Field{
            ResourceOffset(field.resourceOffset),
            remapping[field.subtypeId],
            vec_from_span(dims),
            flatSize});
      }
    }
    ++i;
  }
  if (db.bindlessTypeId)
  {
    if (bindlessTypeId != TypeId::INVALID)
    {
      ETNA_VERIFY(remapping[db.bindlessTypeId] == bindlessTypeId);
      ETNA_VERIFY(getRootTypeId(bindlessParsetId) == bindlessTypeId);
    }
    else
    {
      bindlessTypeId = remapping[db.bindlessTypeId];
    }
  }
  return remapping;
}

BindlessDsetBindingBuilder ParameterSetCache::bindlessDsetBuilder() const
{
  ETNA_VERIFY(bindlessParsetId != ParameterSetId::INVALID);
  return BindlessDsetBindingBuilder{*this, bindlessParsetId};
}

} // namespace etna
