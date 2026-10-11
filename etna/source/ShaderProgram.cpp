#include <etna/ShaderProgram.hpp>

#include <fstream>
#include <spirv_reflect.h>
#include <fmt/std.h>

#include <etna/GlobalContext.hpp>
#include <etna/SlangShaderFormat.hpp>


namespace etna
{

ShaderModule::ShaderModule(vk::Device device, std::filesystem::path shader_path)
  : path{shader_path}
{
  reload(device);
}

static std::vector<char> read_file(std::filesystem::path filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open())
  {
    ETNA_PANIC("Failed to open file {}", filename);
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
}

struct SpvModDeleter
{
  SpvModDeleter() {}
  void operator()(SpvReflectShaderModule* mod) const
  {
    spvReflectDestroyShaderModule(mod);
    delete mod;
  }
};


#define ETNA_SPV_REFLECT_VERIFY(res, path)                                                         \
  ETNA_VERIFYF((res) == SPV_REFLECT_RESULT_SUCCESS, "SPIR-V parse error in {}", (path))

void ShaderModule::reload(vk::Device device)
{
  vkModule = {};

  auto code = read_file(path);
  vk::ShaderModuleCreateInfo info{};
  info.setPCode(reinterpret_cast<const uint32_t*>(code.data()));
  info.setCodeSize(code.size());

  if (code.size() % 4 != 0)
    ETNA_PANIC("SPIRV {} broken", path);

  vkModule = unwrap_vk_result(device.createShaderModuleUnique(info));

  std::unique_ptr<SpvReflectShaderModule, SpvModDeleter> spvModule;
  spvModule.reset(new SpvReflectShaderModule{});

  ETNA_SPV_REFLECT_VERIFY(
    spvReflectCreateShaderModule(code.size(), code.data(), spvModule.get()), path);

  stage = static_cast<vk::ShaderStageFlagBits>(spvModule->shader_stage);
  entryPoint = spvModule->entry_point_name;

  resources.clear();

  uint32_t count = 0;
  ETNA_SPV_REFLECT_VERIFY(
    spvReflectEnumerateDescriptorSets(spvModule.get(), &count, nullptr), path);

  std::vector<SpvReflectDescriptorSet*> sets(count);
  ETNA_SPV_REFLECT_VERIFY(
    spvReflectEnumerateDescriptorSets(spvModule.get(), &count, sets.data()), path);

  resources.reserve(sets.size());

  for (auto pSet : sets)
  {
    DescriptorSetInfo dsInfo;
    dsInfo.clear();
    dsInfo.parseShader(stage, *pSet);
    resources.push_back({pSet->set, dsInfo});
  }

  if (spvModule->push_constant_block_count == 1)
  {
    auto& blk = spvModule->push_constant_blocks[0];
    if (blk.offset != 0)
    {
      ETNA_PANIC("SPIRV {} parse error: PushConst offset is not zero", path);
    }
    pushConst.stageFlags = stage;
    pushConst.offset = 0u;
    pushConst.size = blk.size;
  }
  else if (spvModule->push_constant_block_count > 1)
  {
    ETNA_PANIC("SPIRV {} parse error: only 1 push_const block per shader supported", path);
  }
  else
  {
    pushConst.size = 0u;
    pushConst.offset = 0u;
    pushConst.stageFlags = vk::ShaderStageFlags{};
  }
}

uint32_t ShaderProgramManager::registerModule(std::filesystem::path path)
{
  auto it = shaderModuleNames.find(path);
  if (it != shaderModuleNames.end())
    return it->second;

  uint32_t modId = static_cast<uint32_t>(shaderModules.size());
  std::unique_ptr<ShaderModule> newMod;
  newMod.reset(new ShaderModule{get_context().getDevice(), path});
  shaderModules.push_back(std::move(newMod));
  return modId;
}

static void validate_program_shaders(
  const std::string& name, const std::vector<vk::ShaderStageFlagBits>& stages)
{
  auto supportedShaders = vk::ShaderStageFlagBits::eVertex |
    vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute;

  bool isComputePipeline = false;
  vk::ShaderStageFlags usageMask = static_cast<vk::ShaderStageFlags>(0);

  for (auto stage : stages)
  {
    if (!(stage & supportedShaders))
    {
      ETNA_PANIC(
        "Shader program {} creating error, unsupported shader stage {}",
        name,
        vk::to_string(stage));
    }

    if (stage & usageMask)
    {
      ETNA_PANIC(
        "Shader program {} creating error, multiple usage of {} shader stage",
        name,
        vk::to_string(stage));
    }

    isComputePipeline |= (stage == vk::ShaderStageFlagBits::eCompute);
    usageMask |= stage;
  }

  if (isComputePipeline && stages.size() != 1)
  {
    ETNA_PANIC("Shader program {} creating error, usage of compute shader with other stages", name);
  }
}

ShaderProgramId ShaderProgramManager::loadProgram(
  const char* name, std::span<std::filesystem::path const> shaders_path)
{
  if (programNames.find(name) != programNames.end())
    ETNA_PANIC("Shader program {} redefenition", name);

  std::vector<uint32_t> moduleIds;
  std::vector<vk::ShaderStageFlagBits> stages;
  for (const auto& path : shaders_path)
  {
    auto id = registerModule(path);
    auto stage = getModule(id).getStage();

    moduleIds.push_back(id);
    stages.push_back(stage);
  }

  validate_program_shaders(name, stages);

  ShaderProgramId progId = static_cast<ShaderProgramId>(programs.size());
  programs.emplace_back(new ShaderProgramInternal{name, std::move(moduleIds)});
  programs[static_cast<std::underlying_type_t<ShaderProgramId>>(progId)]->reload(*this);
  programNames[name] = progId;
  return progId;
}

ShaderProgramId ShaderProgramManager::loadProgram(
  const char* name, std::filesystem::path combined_shader_path)
{
  if (programNames.find(name) != programNames.end())
    ETNA_PANIC("Shader program {} redefenition", name);

  ShaderProgramId progId = static_cast<ShaderProgramId>(programs.size());
  programs.emplace_back(new ShaderProgramInternal{name, std::move(combined_shader_path)});
  programs[static_cast<std::underlying_type_t<ShaderProgramId>>(progId)]->reload(*this);
  programNames[name] = progId;
  return progId;
}

ShaderProgramId ShaderProgramManager::tryGetProgram(const char* name) const
{
  auto it = programNames.find(name);
  if (it == programNames.end())
    return ShaderProgramId::Invalid;
  return it->second;
}

ShaderProgramId ShaderProgramManager::getProgram(const char* name) const
{
  auto it = programNames.find(name);
  if (it == programNames.end())
    ETNA_PANIC("Shader program {} not found", name);
  return it->second;
}

void ShaderProgramManager::ShaderProgramInternal::reload(ShaderProgramManager& manager)
{
  if (auto* split = std::get_if<SplitShaderProgramData>(&data))
    reloadSplit(manager, *split);
  else
    reloadCombined(manager, std::get<CombinedShaderProgramData>(data));
}

void ShaderProgramManager::ShaderProgramInternal::reloadSplit(
  ShaderProgramManager& manager, SplitShaderProgramData& split)
{
  progLayout = {};
  usedDescriptors = {};
  pushConst = vk::PushConstantRange{};

  std::array<DescriptorSetInfo, MAX_PROGRAM_DESCRIPTORS> dstDescriptors;
  auto& descriptorLayoutCache = get_context().getDescriptorSetLayouts();

  uint32_t usedDescriptorSetRange = 0;

  for (auto id : split.moduleIds)
  {
    auto& shaderMod = manager.getModule(id);

    if (shaderMod.getPushConst().size > 0) // merge push constants
    {
      auto modPushConst = shaderMod.getPushConst();
      if (pushConst.size == 0u)
      {
        pushConst = modPushConst;
      }
      else
      {
        ETNA_ASSERTF(
          pushConst.size == modPushConst.size,
          "ShaderProgram {}: push constant blocks differ between modules (stages), "
          "expected {} bytes but got {} bytes in module {}",
          name,
          pushConst.size,
          modPushConst.size,
          shaderMod.getName());
        pushConst.stageFlags |= modPushConst.stageFlags;
      }
    }

    const auto& resources = shaderMod.getResources(); // merge descriptors
    for (auto& desc : resources)
    {
      if (desc.first >= MAX_PROGRAM_DESCRIPTORS)
        ETNA_PANIC(
          "ShaderProgram {} : set {} out of max sets ({})",
          name,
          desc.first,
          MAX_PROGRAM_DESCRIPTORS);

      usedDescriptors.set(desc.first);
      dstDescriptors[desc.first].merge(desc.second);

      usedDescriptorSetRange = std::max(desc.first + 1, usedDescriptorSetRange);
    }
  }

  static constexpr DescriptorSetInfo NULL_DSET_INFO{};

  std::vector<vk::DescriptorSetLayout> vkLayouts;

  for (uint32_t i = 0; i < usedDescriptorSetRange; i++)
  {
    const DescriptorSetInfo& dsetInfo =
      usedDescriptors.test(i) ? dstDescriptors[i] : NULL_DSET_INFO;
    auto res = descriptorLayoutCache.get(get_context().getDevice(), dsetInfo);
    descriptorIds[i] = res.first;
    vkLayouts.push_back(res.second);
  }

  vk::PipelineLayoutCreateInfo info{};
  info.setSetLayouts(vkLayouts);

  if (pushConst.size > 0)
  {
    info.setPPushConstantRanges(&pushConst);
    info.setPushConstantRangeCount(1u);
  }

  progLayout = unwrap_vk_result(get_context().getDevice().createPipelineLayoutUnique(info));
}

template <class T>
static std::span<const T> rebase_ser_array(
  const SlangShaderTypedArrayRef<T>& ref, const char* blob)
{
  return std::span<const T>{(const T*)(blob + ref.byteOffset), ref.elemCount};
}

static std::string_view rebase_ser_string(const SlangShaderString& ref, const char* blob)
{
  return std::string_view{(const char*)(blob + ref.byteOffset), ref.charCount};
}

void ShaderProgramManager::ShaderProgramInternal::reloadCombined(
  ShaderProgramManager&, CombinedShaderProgramData& combined)
{
  auto data = read_file(combined.path);
  if (data.size() < sizeof(SlangShaderHeader))
    ETNA_PANIC("Combined shader program {} is malformed: missing header", name);

  const auto* header = reinterpret_cast<const SlangShaderHeader*>(data.data());
  if (header->magic != SLANG_SHADER_MAGIC)
  {
    ETNA_PANIC("Combined shader program {} is malformed: invalid magic {}", name, header->magic);
  }
  if (header->version != SLANG_SHADER_VERSION)
  {
    ETNA_PANIC(
      "Combined shader program {} is malformed: invalid version {}", name, header->version);
  }

  std::vector<vk::ShaderStageFlagBits> stages{};
  vk::ShaderStageFlags stageMask{};
  vk::ShaderStageFlags pcStageMask{};

  vk::Device device = get_context().getDevice();
  auto& descriptorLayoutCache = get_context().getDescriptorSetLayouts();

  auto loadModule = [&](
                      const SlangShaderModuleRef& mr,
                      vk::ShaderStageFlagBits stage,
                      vk::UniqueShaderModule& vkModule,
                      std::string& entry_point) {
    const SlangShaderBytecodeRef& bc = mr.bytecode;

    if (!slang_shader_ref_has_value(bc))
      return;

    std::span<const char> code{data.data() + bc.byteOffset, bc.byteCount};

    vk::ShaderModuleCreateInfo info{};
    info.setPCode(reinterpret_cast<const uint32_t*>(code.data()));
    info.setCodeSize(code.size());

    if (code.size() % 4 != 0)
      ETNA_PANIC("SPIRV {} broken", name);

    vkModule = unwrap_vk_result(device.createShaderModuleUnique(info));
    stages.push_back(stage);
    stageMask |= stage;
    if (mr.declaresPushConsts)
      pcStageMask |= stage;

    entry_point = rebase_ser_string(mr.entryPoint, data.data());
  };

  loadModule(
    header->vert, vk::ShaderStageFlagBits::eVertex, combined.vert, combined.vertEntryPoint);
  loadModule(
    header->frag, vk::ShaderStageFlagBits::eFragment, combined.frag, combined.fragEntryPoint);
  loadModule(
    header->tesc,
    vk::ShaderStageFlagBits::eTessellationControl,
    combined.tesc,
    combined.tescEntryPoint);
  loadModule(
    header->tese,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    combined.tese,
    combined.teseEntryPoint);
  loadModule(
    header->geom, vk::ShaderStageFlagBits::eGeometry, combined.geom, combined.geomEntryPoint);
  loadModule(
    header->comp, vk::ShaderStageFlagBits::eCompute, combined.comp, combined.compEntryPoint);

  validate_program_shaders(name, stages);

  usedDescriptors = {};
  descriptorIds = {};

  std::array<DescriptorSetInfo, MAX_PROGRAM_DESCRIPTORS> dstDescriptors;
  uint32_t usedDescriptorSetRange = 0;

  auto dsets = rebase_ser_array(header->refl.descriptorSets, data.data());
  for (const auto& dset : dsets)
  {
    auto cbufs = rebase_ser_array(dset.constBuffers, data.data());
    auto resources = rebase_ser_array(dset.resources, data.data());

    DescriptorSetInfo dsInfo;
    dsInfo.parseReflection(stageMask, cbufs, resources);

    usedDescriptors.set(dset.index);
    dstDescriptors[dset.index] = dsInfo;
    usedDescriptorSetRange = std::max(uint32_t(dset.index + 1), usedDescriptorSetRange);
  }

  pushConst = {};
  pushConst.size = header->refl.pushConstants.totalByteCount;
  pushConst.stageFlags = pcStageMask;

  // @TODO: reflection data for builder

  constexpr DescriptorSetInfo NULL_DSET_INFO{};

  std::vector<vk::DescriptorSetLayout> vkLayouts;
  for (uint32_t i = 0; i < usedDescriptorSetRange; i++)
  {
    const DescriptorSetInfo& dsetInfo =
      usedDescriptors.test(i) ? dstDescriptors[i] : NULL_DSET_INFO;
    auto res = descriptorLayoutCache.get(get_context().getDevice(), dsetInfo);
    descriptorIds[i] = res.first;
    vkLayouts.push_back(res.second);
  }

  vk::PipelineLayoutCreateInfo info{};
  info.setSetLayouts(vkLayouts);

  if (pushConst.size > 0)
  {
    info.setPPushConstantRanges(&pushConst);
    info.setPushConstantRangeCount(1u);
  }

  progLayout = unwrap_vk_result(get_context().getDevice().createPipelineLayoutUnique(info));
}

void ShaderProgramManager::reloadPrograms()
{
  for (auto& mod : shaderModules)
  {
    mod->reload(get_context().getDevice());
  }

  for (auto& progPtr : programs)
  {
    progPtr->reload(*this);
  }
}

void ShaderProgramManager::clear()
{
  programNames.clear();
  programs.clear();
  shaderModuleNames.clear();
  shaderModules.clear();
}

std::vector<vk::PipelineShaderStageCreateInfo> ShaderProgramManager::getShaderStages(
  ShaderProgramId id) const
{
  auto& prog = getProgInternal(id);
  std::vector<vk::PipelineShaderStageCreateInfo> stages;
  if (const auto* split = std::get_if<SplitShaderProgramData>(&prog.data))
  {
    stages.reserve(split->moduleIds.size());
    for (auto modId : split->moduleIds)
    {
      const auto& shaderMod = getModule(modId);
      vk::PipelineShaderStageCreateInfo info{};
      info.setModule(shaderMod.getVkModule());
      info.setStage(shaderMod.getStage());
      info.setPName(shaderMod.getName().c_str());
      stages.push_back(info);
    }
  }
  else
  {
    const auto& cm = std::get<CombinedShaderProgramData>(prog.data);
    struct Csi
    {
      const vk::UniqueShaderModule& mod;
      vk::ShaderStageFlagBits stage;
      const std::string& entry;
    };
    for (const auto& stage :
         {Csi{cm.vert, vk::ShaderStageFlagBits::eVertex, cm.vertEntryPoint},
          Csi{cm.frag, vk::ShaderStageFlagBits::eFragment, cm.fragEntryPoint},
          Csi{cm.tesc, vk::ShaderStageFlagBits::eTessellationControl, cm.tescEntryPoint},
          Csi{cm.tese, vk::ShaderStageFlagBits::eTessellationEvaluation, cm.teseEntryPoint},
          Csi{cm.geom, vk::ShaderStageFlagBits::eGeometry, cm.geomEntryPoint},
          Csi{cm.comp, vk::ShaderStageFlagBits::eCompute, cm.compEntryPoint}})
    {
      if (!stage.mod)
        continue;
      vk::PipelineShaderStageCreateInfo info{};
      info.setModule(stage.mod.get());
      info.setStage(stage.stage);
      info.setPName(stage.entry.c_str());
      stages.push_back(info);
    }
  }
  return stages;
}

vk::DescriptorSetLayout ShaderProgramManager::getDescriptorLayout(
  ShaderProgramId id, uint32_t set) const
{
  auto layoutId = getDescriptorLayoutId(id, set);
  return get_context().getDescriptorSetLayouts().getVkLayout(layoutId);
}

vk::PushConstantRange ShaderProgramInfo::getPushConst() const
{
  auto& prog = mgr.getProgInternal(id);
  return prog.pushConst;
}

vk::PipelineLayout ShaderProgramInfo::getPipelineLayout() const
{
  auto& prog = mgr.getProgInternal(id);
  return prog.progLayout.get();
}

bool ShaderProgramInfo::isDescriptorSetUsed(uint32_t set) const
{
  auto& prog = mgr.getProgInternal(id);
  return set < MAX_PROGRAM_DESCRIPTORS && prog.usedDescriptors.test(set);
}

DescriptorLayoutId ShaderProgramInfo::getDescriptorLayoutId(uint32_t set) const
{
  ETNA_VERIFY(isDescriptorSetUsed(set));
  return mgr.getProgInternal(id).descriptorIds.at(set);
}

vk::DescriptorSetLayout ShaderProgramInfo::getDescriptorSetLayout(uint32_t set) const
{
  return get_context().getDescriptorSetLayouts().getVkLayout(getDescriptorLayoutId(set));
}

const DescriptorSetInfo& ShaderProgramInfo::getDescriptorSetInfo(uint32_t set) const
{
  return get_context().getDescriptorSetLayouts().getLayoutInfo(getDescriptorLayoutId(set));
}

} // namespace etna
