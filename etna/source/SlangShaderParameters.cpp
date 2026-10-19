#include <etna/SlangShaderParameters.hpp>

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

ParameterSetId ParameterSetCache::parseRootParameterSet(
  const SlangShaderDescriptorSetReflection& dset,
  const char* blob,
  std::span<const TypeId> typeid_remapping)
{
  (void)dset;
  (void)blob;
  (void)typeid_remapping;
  // ETNA_ASSERTF(0, "TODO");
  return ParameterSetId::INVALID;
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
        rtType.fields.emplace(
          std::string{name},
          Field{
            ResourceOffset(field.resourceOffset),
            remapping[field.subtypeId],
            vec_from_span(rebase_ser_array(field.arrayDims, blob))});
      }
    }
    ++i;
  }
  return remapping;
}

} // namespace etna
