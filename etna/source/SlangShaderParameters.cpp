#include <etna/SlangShaderParameters.hpp>

#include <etna/Assert.hpp>

namespace etna
{

ParameterSetId ParameterSetCache::parseRootParameterSet(
  const SlangShaderDescriptorSetReflection& dset, const char* blob, ParameterSetId remapped_typeid)
{
  (void)dset;
  (void)blob;
  (void)remapped_typeid;
  // ETNA_ASSERTF(0, "TODO");
  return ParameterSetId::INVALID;
}

std::vector<ParameterSetId> ParameterSetCache::addTypes(
  const SlangShaderReflectionHeader& refl, const char* blob)
{
  (void)refl;
  (void)blob;
  // ETNA_ASSERTF(0, "TODO");
  return {};
}

} // namespace etna
