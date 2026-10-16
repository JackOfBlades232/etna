#define _CRT_SECURE_NO_WARNINGS
#include "Slang.hpp"
#include "SlangShaderSerialization.hpp"
#include "Common.hpp"

#include <etna/SlangShaderFormat.hpp>

#include <spdlog/spdlog.h>

#include <spirv-tools/libspirv.hpp>

#include <cstdlib>
#include <cassert>
#include <optional>
#include <filesystem>
#include <vector>
#include <array>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace etna;

// @TODO: proper error handling/reporting

static std::optional<std::string> get_env_var(const char* name)
{
  if (const char* value = std::getenv(name); value && *value)
    return std::string{value};
  return std::nullopt;
}

static void* try_load_dynlib(const std::filesystem::path& full_path)
{
  std::error_code ec;
  if (!std::filesystem::exists(full_path, ec))
    return nullptr;

#if defined(_WIN32)
  return static_cast<void*>(::LoadLibraryW(full_path.wstring().c_str()));
#else
  return ::dlopen(full_path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

static void unload_dynlib(void* handle)
{
  if (handle == nullptr)
    return;
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  ::dlclose(handle);
#endif
}

static void* try_load_slang_dynlib()
{
#if defined(_WIN32)
  constexpr const char* LIB_FILE_NAME = "slang.dll";
  constexpr const char* LIB_BIN_SUBDIR = "Bin";
#elif defined(__APPLE__)
  constexpr const char* LIB_FILE_NAME = "libslang.dylib";
  constexpr const char* LIB_BIN_SUBDIR = "lib";
#else
  constexpr const char* LIB_FILE_NAME = "libslang.so";
  constexpr const char* LIB_BIN_SUBDIR = "lib";
#endif

  std::vector<std::filesystem::path> searchDirs;

  if (auto slangSdk = get_env_var("SLANG_SDK_DIR"))
    searchDirs.emplace_back(std::filesystem::path(*slangSdk) / LIB_BIN_SUBDIR);

  if (auto vulkanSdk = get_env_var("VULKAN_SDK"))
    searchDirs.emplace_back(std::filesystem::path(*vulkanSdk) / LIB_BIN_SUBDIR);

  for (const auto& dir : searchDirs)
  {
    if (void* handle = try_load_dynlib(dir / LIB_FILE_NAME))
      return handle;
  }

  return nullptr;
}

SlangCompiler::SlangCompiler(const CreateInfo& ci)
{
  dynlibHandle = try_load_slang_dynlib();
  assert(dynlibHandle);

  SlangGlobalSessionDesc desc = {};
  [[maybe_unused]] /* @TODO any mode verification */ auto gsRes =
    slang::createGlobalSession(&desc, globalSession.writeRef());
  assert(SLANG_SUCCEEDED(gsRes));

  slang::SessionDesc sessionDesc = {};

  slang::TargetDesc targetDesc = {
    .format = SLANG_SPIRV, .profile = globalSession->findProfile("spirv_1_6")};
  sessionDesc.targets = &targetDesc;
  sessionDesc.targetCount = 1;

  std::vector<std::string> incDirStrings{};
  std::vector<slang::CompilerOptionEntry> options = {
    {slang::CompilerOptionName::EmitSpirvDirectly, {slang::CompilerOptionValueKind::Int, 1}},
    {slang::CompilerOptionName::SkipSPIRVValidation, {slang::CompilerOptionValueKind::Int, 0}},
    {slang::CompilerOptionName::VulkanUseEntryPointName, {slang::CompilerOptionValueKind::Int, 1}},
    {slang::CompilerOptionName::MatrixLayoutColumn, {slang::CompilerOptionValueKind::Int, 1}}};
  if (ci.emitSpirvDebugInfo)
  {
    options.push_back(slang::CompilerOptionEntry{
      slang::CompilerOptionName::DebugInformation,
      slang::CompilerOptionValue{
        slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL}});
  }
  if (ci.warningsAsErrors)
  {
    options.push_back(slang::CompilerOptionEntry{
      slang::CompilerOptionName::WarningsAsErrors,
      slang::CompilerOptionValue{
        .kind = slang::CompilerOptionValueKind::String, .stringValue0 = "all"}});
  }
  for (const auto& incDir : ci.commonIncludeDirs)
  {
    const auto& str = incDirStrings.emplace_back(to_char_str(incDir.string()));
    options.push_back(slang::CompilerOptionEntry{
      slang::CompilerOptionName::Include,
      slang::CompilerOptionValue{
        .kind = slang::CompilerOptionValueKind::String, .stringValue0 = str.c_str()}});
  }

  sessionDesc.compilerOptionEntries = options.data();
  sessionDesc.compilerOptionEntryCount = uint32_t(options.size());

  [[maybe_unused]] /* @TODO any mode verification */ auto sRes =
    globalSession->createSession(sessionDesc, session.writeRef());
  assert(SLANG_SUCCEEDED(sRes));
}

SlangCompiler::~SlangCompiler()
{
  if (session)
    session.setNull();
  if (globalSession)
    globalSession.setNull();
  if (dynlibHandle)
    unload_dynlib(dynlibHandle);
}

static void diagnose_if_needed(const Slang::ComPtr<slang::IBlob>& blob, bool op_failed)
{
  if (!blob)
    return;
  if (op_failed)
    spdlog::error("{}", (const char*)blob->getBufferPointer());
  else
    spdlog::warn("{}", (const char*)blob->getBufferPointer());
}

static const char* stage_name(SlangStage stage)
{
  static constexpr const char* STAGE_NAMES[SLANG_STAGE_COUNT] = {
    "NONE",
    "VERTEX",
    "HULL",
    "DOMAIN",
    "GEOMETRY",
    "FRAGMENT",
    "COMPUTE",
    "RAY_GENERATION",
    "INTERSECTION",
    "ANY_HIT",
    "CLOSEST_HIT",
    "MISS",
    "CALLABLE",
    "MESH",
    "AMPLIFICATION",
    "DISPATCH",
    "NODE",
  };
  if (stage >= 0 && stage < SLANG_STAGE_COUNT)
    return STAGE_NAMES[stage];
  else
    return nullptr;
}

static SlangShaderModuleRef* module_slot(SlangShaderHeader* header, SlangStage stage)
{
  assert(header);
  switch (stage)
  {
  case SLANG_STAGE_VERTEX:
    return &header->vert;
  case SLANG_STAGE_HULL:
    return &header->tesc;
  case SLANG_STAGE_DOMAIN:
    return &header->tese;
  case SLANG_STAGE_GEOMETRY:
    return &header->geom;
  case SLANG_STAGE_FRAGMENT:
    return &header->frag;
  case SLANG_STAGE_COMPUTE:
    return &header->comp;
  default:
    spdlog::error(
      "Unsupported entry point stage {}, etna-slangc only supports VERTEX, HULL, DOMAIN, GEOMETRY, "
      "FRAGMENT and COMPUTE at this point",
      stage_name(stage));
    return nullptr;
  }
}

// Do it this way to be able to import entry points. Need to beware of conflicts though.
static Slang::ComPtr<slang::IEntryPoint> find_entry_point_in_session(
  slang::ISession* session, const char* name)
{
  const SlangInt count = session->getLoadedModuleCount();

  for (SlangInt i = 0; i < count; ++i)
  {
    slang::IModule* slangModule = session->getLoadedModule(i);

    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    if (SLANG_SUCCEEDED(slangModule->findEntryPointByName(name, entryPoint.writeRef())))
    {
      return entryPoint;
    }
  }

  return {};
}

// @TODO: come up with a way to remove duplication or nah?
static constexpr size_t PREALLOC_COUNT = 512; // Imposed max for pointer stability

struct ResourceReflection
{
  std::string name;
  SlangShaderResourceDesc desc;
  SlangShaderResourceBinding binding;
};

struct DescriptorSetReflection
{
  std::string name;
  std::vector<ResourceReflection> resources;
  uint64_t index;
  uint64_t typeId = UINT64_MAX;
};

struct PushConstantsReflection : ResourceReflection
{
};

struct FieldReflection
{
  std::string name;
  uint64_t resourceOffset = UINT64_MAX;
  uint64_t subtypeId = UINT64_MAX;
  std::vector<uint64_t> arrayDims;

  bool operator==(const FieldReflection&) const = default;
};

struct TypeReflection
{
  std::vector<FieldReflection> fields;

  bool operator==(const TypeReflection&) const = default;
};

struct TypeDb
{
  std::vector<TypeReflection> types;
  std::vector<size_t> typeHashes;
};

struct ShaderReflection
{
  std::vector<DescriptorSetReflection> dsets;
  PushConstantsReflection pushConsts;
  TypeDb typeDb;
};

struct AccessPathNode
{
  slang::VariableLayoutReflection* varLayout;
  AccessPathNode* outer;
};

struct AccessPath
{
  AccessPathNode* leaf;
  AccessPathNode* deepestCbuf;
  AccessPathNode* deepestParblock;
};

struct CombinedOffset
{
  uint32_t dset;
  uint32_t binding;
  uint32_t byteOffset;
};

template <class T>
static void hash_combine(size_t& seed, const T& v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

static uint64_t add_type(TypeReflection&& type, TypeDb& db)
{
  size_t hash = 0;
  for (const auto& f : type.fields)
  {
    hash_combine(hash, f.name);
    hash_combine(hash, f.resourceOffset);
    hash_combine(hash, f.subtypeId);
    hash_combine(hash, f.arrayDims.size());
    for (uint64_t dim : f.arrayDims)
      hash_combine(hash, dim);
  }

  for (size_t i = 0; i < db.types.size(); ++i)
  {
    if (db.typeHashes[i] == hash && db.types[i] == type)
    {
      TypeReflection localDestruction{std::move(type)};
      return uint64_t(i);
    }
  }

  auto id = uint64_t(db.types.size());
  db.types.emplace_back() = std::move(type);
  db.typeHashes.emplace_back() = hash;
  return id;
}

static CombinedOffset calculate_combined_offset(AccessPath path)
{
  CombinedOffset offset = {};
  bool reachedCbuf = false;
  bool reachedParblock = false;
  for (auto* node = path.leaf; node; node = node->outer)
  {
    if (!reachedParblock)
    {
      reachedParblock |= node == path.deepestParblock;
      if (!reachedParblock)
      {
        offset.binding += node->varLayout->getOffset(slang::ParameterCategory::DescriptorTableSlot);
        offset.dset +=
          node->varLayout->getBindingSpace(slang::ParameterCategory::DescriptorTableSlot);
      }
    }

    if (!reachedCbuf)
    {
      reachedCbuf |= node == path.deepestCbuf;
      if (!reachedCbuf)
        offset.byteOffset += node->varLayout->getOffset();
    }

    if (reachedCbuf && reachedParblock)
      break;
  }
  assert(offset.dset == 0);
  for (auto* node = path.deepestParblock; node; node = node->outer)
    offset.dset += node->varLayout->getOffset(slang::ParameterCategory::SubElementRegisterSpace);
  return offset;
}

static bool dset_is_empty(const DescriptorSetReflection& dset)
{
  return dset.resources.empty();
}

static uint64_t array_size_mul(size_t base, size_t mul)
{
  if (base == 0 || mul == 0)
    return 0;
  if (base == UINT64_MAX || mul == UINT64_MAX)
    return UINT64_MAX;
  return base * mul;
}

// @NOTE: working around https://github.com/shader-slang/slang/issues/13529
static uint64_t get_total_slang_array_element_count(
  slang::TypeLayoutReflection* type_layout,
  slang::ProgramLayout* program_layout,
  std::string& typedesc,
  FieldReflection* field)
{
  size_t result = 1;
  auto* type = type_layout->getType();
  for (;;)
  {
    if (!type->isArray())
      return result;

    const auto c = type->getElementCount((SlangReflection*)program_layout);
    if (c == SLANG_UNKNOWN_SIZE)
    {
      typedesc += "[?]";
      if (field)
        field->arrayDims.push_back(UINT64_MAX - 1);
      result = SLANG_UNKNOWN_SIZE;
    }
    if (
      c == SLANG_UNBOUNDED_SIZE ||
      c == INT32_MAX /* Very weird, but this is what it seems to return */)
    {
      typedesc += "[]";
      if (field)
        field->arrayDims.push_back(UINT64_MAX);
      result = SLANG_UNBOUNDED_SIZE;
    }
    else
    {
      typedesc += fmt::format("[{}]", c);
      if (field)
        field->arrayDims.push_back(c);
      if (result != SLANG_UNKNOWN_SIZE && result != SLANG_UNBOUNDED_SIZE)
        result *= c;
    }
    type = type->getElementType();
  }
}

static void compile_dset_element(
  ShaderReflection& refl,
  DescriptorSetReflection& dset,
  ResourceReflection* cbuf,
  FieldReflection* field,
  uint64_t* current_field_offset,
  slang::ProgramLayout* program,
  slang::VariableLayoutReflection* var,
  AccessPath path,
  size_t arraySize,
  const char* base_name);

static void compile_dset_element(
  ShaderReflection& refl,
  DescriptorSetReflection& dset,
  ResourceReflection* cbuf,
  FieldReflection* field,
  uint64_t* current_field_offset,
  slang::ProgramLayout* program,
  slang::TypeLayoutReflection* type,
  AccessPath path,
  size_t arraySize,
  const char* name)
{
  auto typeKind = type->getKind();
  auto parCat = type->getParameterCategory();
  if (
    parCat != slang::ParameterCategory::Mixed &&               //
    parCat != slang::ParameterCategory::Uniform &&             //
    parCat != slang::ParameterCategory::DescriptorTableSlot && //
    parCat != slang::ParameterCategory::PushConstantBuffer &&  //
    parCat != slang::ParameterCategory::SubElementRegisterSpace)
  {
    return;
  }
  bool hasResources = type->getSize(slang::ParameterCategory::DescriptorTableSlot) > 0;
  switch (typeKind)
  {
  case slang::TypeReflection::Kind::Struct: {
    int fieldCount = type->getFieldCount();
    std::optional<TypeReflection> typeRefl{};
    if (hasResources)
      typeRefl.emplace();
    uint64_t innerOffset = 0;
    for (int f = 0; f < fieldCount; ++f)
    {
      slang::VariableLayoutReflection* fv = type->getFieldByIndex(f);
      slang::TypeLayoutReflection* ft = fv->getTypeLayout();
      auto* nfield = hasResources && ft->getSize(slang::ParameterCategory::DescriptorTableSlot) > 0
        ? &typeRefl->fields.emplace_back()
        : nullptr;
      auto* noff = nfield ? &innerOffset : nullptr;
      compile_dset_element(refl, dset, cbuf, nfield, noff, program, fv, path, arraySize, name);
    }
    if (typeRefl)
    {
      uint64_t& idDest = field ? field->subtypeId : dset.typeId;
      assert(idDest == UINT64_MAX);
      idDest = add_type(std::move(*typeRefl), refl.typeDb);
      if (current_field_offset)
      {
        field->resourceOffset = *current_field_offset;
        *current_field_offset += innerOffset;
      }
    }
    break;
  }
  case slang::TypeReflection::Kind::Array: {
    // We don't really care about dimensionality here, since etna binding API is flat
    std::string arrayName = name;

    FieldReflection* nfield = nullptr;
    uint64_t* noff = nullptr;
    if (type->getElementTypeLayout()->getSize(slang::ParameterCategory::DescriptorTableSlot) > 0)
    {
      nfield = field;
      noff = current_field_offset;
    }

    size_t slangElementCount =
      get_total_slang_array_element_count(type, program, arrayName, nfield);
    assert(slangElementCount != SLANG_UNKNOWN_SIZE);
    uint64_t elementCount =
      slangElementCount == SLANG_UNBOUNDED_SIZE ? UINT64_MAX : slangElementCount;
    compile_dset_element(
      refl,
      dset,
      cbuf,
      nfield,
      noff,
      program,
      type->getElementTypeLayout(),
      path,
      array_size_mul(arraySize, elementCount),
      arrayName.c_str());
    break;
  }
  case slang::TypeReflection::Kind::Matrix:
  case slang::TypeReflection::Kind::Vector:
  case slang::TypeReflection::Kind::Scalar:
    assert(cbuf);
    break;
  case slang::TypeReflection::Kind::ConstantBuffer: {
    slang::VariableLayoutReflection* implicitVar = type->getContainerVarLayout();
    slang::TypeLayoutReflection* innerType = type->getElementTypeLayout();

    ResourceReflection* nestedCbuf = nullptr;

    slang::TypeLayoutReflection* implicitVarType = implicitVar->getTypeLayout();
    if (auto binds = implicitVarType->getSize(slang::ParameterCategory::DescriptorTableSlot);
        binds > 0)
    {
      auto offset = calculate_combined_offset({path.leaf, path.leaf, path.deepestParblock});
      assert(dset.resources.size() < PREALLOC_COUNT);
      nestedCbuf = &dset.resources.emplace_back();
      nestedCbuf->name = name;
      nestedCbuf->binding = {offset.binding, array_size_mul(binds, arraySize)};
      nestedCbuf->desc.type = SlangShaderResourceType::CONSTBUFFER;
      nestedCbuf->desc.access = SlangShaderResourceAccess::RO;
      nestedCbuf->desc.cbufTotalByteCount = innerType->getStride(slang::ParameterCategory::Uniform);
      assert(binds == 1);
      assert(nestedCbuf->desc.cbufTotalByteCount > 0);
    }
    else if (auto binds = implicitVarType->getSize(slang::ParameterCategory::PushConstantBuffer);
             binds > 0)
    {
      assert(refl.pushConsts.desc.cbufTotalByteCount == 0);
      refl.pushConsts.name = name;
      refl.pushConsts.desc.cbufTotalByteCount =
        innerType->getStride(slang::ParameterCategory::Uniform);
      nestedCbuf = &refl.pushConsts;
      assert(binds == 1);
      assert(arraySize == 1);
      assert(nestedCbuf->desc.cbufTotalByteCount > 0);
    }
    else
    {
      assert(innerType->getSize(slang::ParameterCategory::Uniform) == 0);
    }

    compile_dset_element(
      refl,
      dset,
      nestedCbuf,
      field,
      current_field_offset,
      program,
      innerType,
      {path.leaf, path.leaf, path.deepestParblock},
      arraySize,
      name);

    break;
  }
  case slang::TypeReflection::Kind::ParameterBlock: {
    slang::VariableLayoutReflection* implicitVar = type->getContainerVarLayout();
    slang::TypeLayoutReflection* innerType = type->getElementTypeLayout();

    assert(refl.dsets.size() < PREALLOC_COUNT);
    DescriptorSetReflection& nestedDset = refl.dsets.emplace_back();
    ResourceReflection* nestedCbuf = nullptr;

    // Dsets are root names even when inside implicit ones
    const char* dsetName = strrchr(name, ':');
    if (dsetName)
      ++dsetName;
    else
      dsetName = name;

    nestedDset.name = dsetName;

    slang::TypeLayoutReflection* implicitVarType = implicitVar->getTypeLayout();
    auto slots = implicitVarType->getSize(slang::ParameterCategory::SubElementRegisterSpace);
    assert(slots == 1);
    assert(arraySize == 1);

    auto offset = calculate_combined_offset({path.leaf, path.leaf, path.leaf});
    nestedDset.index = offset.dset;

    if (auto binds = implicitVarType->getSize(slang::ParameterCategory::DescriptorTableSlot);
        binds > 0)
    {
      nestedCbuf = &nestedDset.resources.emplace_back();
      nestedCbuf->name = dsetName;
      nestedCbuf->binding = {offset.binding, binds};
      nestedCbuf->desc.type = SlangShaderResourceType::CONSTBUFFER;
      nestedCbuf->desc.access = SlangShaderResourceAccess::RO;
      nestedCbuf->desc.cbufTotalByteCount = innerType->getStride(slang::ParameterCategory::Uniform);
      assert(binds == 1);
      assert(nestedCbuf->desc.cbufTotalByteCount > 0);
    }

    assert(implicitVarType->getSize(slang::ParameterCategory::PushConstantBuffer) == 0);

    compile_dset_element(
      refl,
      nestedDset,
      nestedCbuf,
      field,
      current_field_offset,
      program,
      innerType,
      {path.leaf, path.leaf, path.leaf},
      arraySize,
      dsetName);

    break;
  }
  case slang::TypeReflection::Kind::Resource: {
    assert(field);
    field->resourceOffset = *current_field_offset;
    field->subtypeId = UINT64_MAX;
    ++(*current_field_offset);

    auto& res = dset.resources.emplace_back();
    res.name = name;
    res.desc = {};
    res.binding = {calculate_combined_offset(path).binding, arraySize};

    switch (auto shape = type->getResourceShape(); shape & SLANG_RESOURCE_BASE_SHAPE_MASK)
    {
    case SLANG_TEXTURE_1D:
    case SLANG_TEXTURE_2D:
    case SLANG_TEXTURE_3D:
    case SLANG_TEXTURE_CUBE:
      res.desc.type = (shape & SLANG_TEXTURE_COMBINED_FLAG)
        ? SlangShaderResourceType::COMBINED_TEXTURE_SAMPLER
        : SlangShaderResourceType::TEXTURE;
      break;

    case SLANG_STRUCTURED_BUFFER:
    case SLANG_BYTE_ADDRESS_BUFFER:
      res.desc.type = SlangShaderResourceType::BUFFER;
      break;

    case SLANG_ACCELERATION_STRUCTURE:
      assert(0);
      break;

    default:
      assert(0);
      break;
    }
    switch (type->getResourceAccess())
    {
    case SLANG_RESOURCE_ACCESS_READ:
      res.desc.access = SlangShaderResourceAccess::RO;
      break;
    case SLANG_RESOURCE_ACCESS_READ_WRITE:
    case SLANG_RESOURCE_ACCESS_WRITE:
      res.desc.access = SlangShaderResourceAccess::RW;
      break;

    default:
      assert(0);
      break;
    }
    break;
  }

  case slang::TypeReflection::Kind::SamplerState: {
    assert(field);
    field->resourceOffset = *current_field_offset;
    field->subtypeId = UINT64_MAX;
    ++(*current_field_offset);

    auto& res = dset.resources.emplace_back();
    res.name = name;
    res.desc = {SlangShaderResourceType::SAMPLER, SlangShaderResourceAccess::RO, 0};
    res.binding = {calculate_combined_offset(path).binding, arraySize};
    break;
  }

  case slang::TypeReflection::Kind::TextureBuffer:
  case slang::TypeReflection::Kind::ShaderStorageBuffer: {
    assert(0);
    break;
  }

  default:
    assert(0);
    break;
  }
}

static void compile_dset_element(
  ShaderReflection& refl,
  DescriptorSetReflection& dset,
  ResourceReflection* cbuf,
  FieldReflection* field,
  uint64_t* current_field_offset,
  slang::ProgramLayout* program,
  slang::VariableLayoutReflection* var,
  AccessPath path,
  size_t arraySize,
  const char* base_name)
{
  const char* leaf_name = var->getName();
  assert(base_name || leaf_name);
  std::string name{base_name};
  if (leaf_name)
  {
    if (base_name)
      name += "::";
    name += leaf_name;
  }

  AccessPathNode node{var, path.leaf};
  compile_dset_element(
    refl,
    dset,
    cbuf,
    field,
    current_field_offset,
    program,
    var->getTypeLayout(),
    {&node, path.deepestCbuf, path.deepestParblock},
    arraySize,
    name.c_str());
}

static void merge_res_reflection(ResourceReflection& combined, const ResourceReflection& src)
{
  if (src.name.empty())
    return;
  if (combined.name.empty())
  {
    combined = src;
    return;
  }
  assert(combined.name == src.name);
  assert(combined.desc == src.desc);
  assert(combined.binding == src.binding); // @TODO: laxer constraints for array sizes?
}

static void merge_dset_reflection(
  DescriptorSetReflection& combined, const DescriptorSetReflection& src)
{
  assert(!dset_is_empty(combined) && !dset_is_empty(src));
  assert(combined.name == src.name);
  for (const auto& res : src.resources)
  {
    bool merged = false;
    for (auto& combRes : combined.resources)
    {
      if (combRes.binding.binding == res.binding.binding)
      {
        merge_res_reflection(combRes, res);
        merged = true;
      }
    }
    if (!merged)
      combined.resources.push_back(res);
  }
}

static void merge_shader_reflection(ShaderReflection& combined, const ShaderReflection& src)
{
  for (const auto& dset : src.dsets)
  {
    bool merged = false;
    for (auto& combDset : combined.dsets)
    {
      if (combDset.index == dset.index)
      {
        merge_dset_reflection(combDset, dset);
        merged = true;
      }
    }
    if (!merged)
      combined.dsets.push_back(dset);
  }
  merge_res_reflection(combined.pushConsts, src.pushConsts);
}

static void serialize_shader_reflection(
  const ShaderReflection& src,
  SlangShaderReflectionHeader& header,
  SlangShaderSerializationContext& sctx)
{
  std::vector<SlangShaderDescriptorSetReflection> serializedDsets;

  for (const auto& dset : src.dsets)
  {
    auto& dst = serializedDsets.emplace_back();
    dst.name = slang_shader_ser_write_string(dset.name, sctx);

    std::vector<SlangShaderResourceReflection> serializedResources;
    serializedResources.emplace_back(); // Id 0
    for (const auto& res : dset.resources)
    {
      auto& dstRes = serializedResources.emplace_back();
      dstRes.name = slang_shader_ser_write_string(res.name, sctx);
      dstRes.desc = res.desc;
      dstRes.binding = res.binding;
    }
    dst.resources = slang_shader_ser_write_typed_array(
      std::span<const SlangShaderResourceReflection>{serializedResources}, sctx);

    dst.index = dset.index;
  }

  header.descriptorSets = slang_shader_ser_write_typed_array(
    std::span<const SlangShaderDescriptorSetReflection>{serializedDsets}, sctx);

  header.pushConstants.name = slang_shader_ser_write_string(src.pushConsts.name, sctx);
  header.pushConstants.totalByteCount = src.pushConsts.desc.cbufTotalByteCount;
}

int SlangCompiler::compile(
  const std::filesystem::path& source,
  std::span<const std::string> entry_points,
  const std::filesystem::path& dest,
  const std::filesystem::path& dest_depfile)
{
  Slang::ComPtr<slang::IModule> slangModule;

  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    auto modulePath = source.string();
    slangModule = session->loadModule(modulePath.c_str(), diagnosticsBlob.writeRef());
    diagnose_if_needed(diagnosticsBlob, slangModule);
    if (!slangModule)
      return -1;
  }

  SlangShaderSerializationContext sctx = {};
  auto* header =
    sctx.ref<SlangShaderHeader>(slang_shader_ser_write_type<SlangShaderHeader>({}, sctx));
  header->magic = SLANG_SHADER_MAGIC;
  header->version = SLANG_SHADER_VERSION;

  ShaderReflection combinedShaderReflection = {};

  spvtools::SpirvTools spvTools(SPV_ENV_UNIVERSAL_1_6); // for validation
  auto printMsgToErrlog =
    [](spv_message_level_t, const char*, const spv_position_t&, const char* m) {
      spdlog::error("{}", m);
    };
  spvTools.SetMessageConsumer(printMsgToErrlog);

  for (const auto& ep : entry_points)
  {
    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    {
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      slangModule->findEntryPointByName(ep.data(), entryPoint.writeRef());
      if (!entryPoint)
        entryPoint = find_entry_point_in_session(session.get(), ep.data());
      if (!entryPoint)
      {
        spdlog::error("Can't get entrypoint {} from {}", ep, source.string());
        return -2;
      }
    }

    std::array<slang::IComponentType*, 2> componentTypes = {slangModule, entryPoint};
    Slang::ComPtr<slang::IComponentType> composedProgram;
    {
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      SlangResult result = session->createCompositeComponentType(
        componentTypes.data(),
        componentTypes.size(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef());
      diagnose_if_needed(diagnosticsBlob, !SLANG_SUCCEEDED(result));
      if (!SLANG_SUCCEEDED(result))
      {
        spdlog::error("Can't compose program with entrypoint {} from {}", ep, source.string());
        return -3;
      }
    }

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    {
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      SlangResult result =
        composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
      diagnose_if_needed(diagnosticsBlob, !SLANG_SUCCEEDED(result));
      if (!SLANG_SUCCEEDED(result))
      {
        spdlog::error("Can't link program with entrypoint {} from {}", ep, source.string());
        return -3;
      }
    }

    slang::ProgramLayout* progLayout = linkedProgram->getLayout();
    slang::EntryPointReflection* entryPointRefl = progLayout->findEntryPointByName(ep.c_str());
    assert(progLayout->getEntryPointCount() == 1);

    SlangStage stage = entryPointRefl->getStage();
    auto* moduleDest = module_slot(header, stage);
    if (!moduleDest)
    {
      return -3;
    }
    if (slang_shader_ref_has_value(moduleDest->bytecode))
    {
      spdlog::error("Duplicate entry point {} for stage {}", ep, stage_name(stage));
      return -3;
    }

    {
      ShaderReflection shaderReflection = {};
      shaderReflection.dsets.reserve(PREALLOC_COUNT);

      slang::VariableLayoutReflection* globalScope = progLayout->getGlobalParamsVarLayout();
      slang::VariableLayoutReflection* entryPointScope = entryPointRefl->getVarLayout();

      DescriptorSetReflection& globalDset = shaderReflection.dsets.emplace_back();
      globalDset.name = "";

      compile_dset_element(
        shaderReflection,
        globalDset,
        nullptr,
        nullptr,
        nullptr,
        progLayout,
        globalScope,
        {},
        1,
        "");
      compile_dset_element(
        shaderReflection,
        globalDset,
        nullptr,
        nullptr,
        nullptr,
        progLayout,
        entryPointScope,
        {},
        1,
        "$EntryPoint");

      std::erase_if(shaderReflection.dsets, dset_is_empty);

      if (shaderReflection.pushConsts.desc.cbufTotalByteCount > 0)
        moduleDest->declaresPushConsts = 1;
      else
        moduleDest->declaresPushConsts = 0;

      merge_shader_reflection(combinedShaderReflection, shaderReflection);
    }

    if (stage == SLANG_STAGE_COMPUTE)
    {
      SlangUInt sizes[3];
      entryPointRefl->getComputeThreadGroupSize(3, sizes);
      header->refl.numThreadsX = sizes[0];
      header->refl.numThreadsY = sizes[1];
      header->refl.numThreadsZ = sizes[2];
    }

    Slang::ComPtr<slang::IBlob> spirvCode;
    {
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      SlangResult result =
        linkedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnosticsBlob.writeRef());
      diagnose_if_needed(diagnosticsBlob, !SLANG_SUCCEEDED(result));
      if (!SLANG_SUCCEEDED(result))
      {
        spdlog::error(
          "Can't generate spirv for program with entrypoint {} from {}", ep, source.string());
        return -3;
      }
    }

    if (!spvTools.Validate(
          (const uint32_t*)spirvCode->getBufferPointer(),
          spirvCode->getBufferSize() / sizeof(uint32_t)))
    {
      spdlog::error(
        "Generated invalid spirv for program with entrypoint {} from {}", ep, source.string());
      return -3;
    }

    moduleDest->bytecode = slang_shader_ser_write_bytecode(
      {(const uint8_t*)spirvCode->getBufferPointer(), spirvCode->getBufferSize()}, sctx);
    moduleDest->entryPoint = slang_shader_ser_write_string(ep, sctx);
  }

  if (
    slang_shader_ref_has_value(header->comp.bytecode) &&
    (slang_shader_ref_has_value(header->vert.bytecode) ||
     slang_shader_ref_has_value(header->frag.bytecode) ||
     slang_shader_ref_has_value(header->tesc.bytecode) ||
     slang_shader_ref_has_value(header->tese.bytecode) ||
     slang_shader_ref_has_value(header->geom.bytecode)))
  {
    spdlog::error("Can't require both graphics and compute entrypoints");
    return -3;
  }
  if (
    !slang_shader_ref_has_value(header->vert.bytecode) &&
    (slang_shader_ref_has_value(header->frag.bytecode) ||
     slang_shader_ref_has_value(header->tesc.bytecode) ||
     slang_shader_ref_has_value(header->tese.bytecode) ||
     slang_shader_ref_has_value(header->geom.bytecode)))
  {
    spdlog::error("Graphics pipeline missing a vertex shader");
    return -3;
  }
  if (
    (slang_shader_ref_has_value(header->tesc.bytecode) ||
     slang_shader_ref_has_value(header->tese.bytecode)) &&
    (!slang_shader_ref_has_value(header->tesc.bytecode) ||
     !slang_shader_ref_has_value(header->tese.bytecode)))
  {
    spdlog::error("Tessellation stage requires both hull and domain shaders");
    return -3;
  }

  serialize_shader_reflection(combinedShaderReflection, header->refl, sctx);

  std::string error{};
  if (!slang_shader_ser_write_to_file(sctx, dest.c_str(), error))
  {
    spdlog::error("{}", error);
    return -3;
  }

  if (!dest_depfile.empty())
  {
    if (FILE* df = fopen(to_char_str(dest_depfile.string()).c_str(), "w+"))
    {
      fprintf(df, "%s:", to_char_str(dest.string()).c_str());
      for (SlangInt32 i = 0; i < slangModule->getDependencyFileCount(); ++i)
      {
        const char* path = slangModule->getDependencyFilePath(i);
        fprintf(df, " %s", path);
      }
      fclose(df);
    }
    else
    {
      spdlog::error("Can't open depfile {}", dest_depfile.string());
    }
  }

  spdlog::info(
    "Compiled module from {} to {} with depfile {}!",
    source.string(),
    dest.string(),
    dest_depfile.string());
  return 0;
}
