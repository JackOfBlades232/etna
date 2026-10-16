#include <etna/SlangShaderParameters.hpp>

#include <etna/Assert.hpp>

namespace etna
{

ParameterSetId ParameterSetCache::parseRootParameterSet(
  const SlangShaderDescriptorSetReflection& refl, const char* blob)
{
  (void)refl;
  (void)blob;
  //ETNA_ASSERTF(0, "TODO");
  return ParameterSetId::INVALID;
}

} // namespace etna
