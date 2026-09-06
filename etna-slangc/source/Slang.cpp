#define _CRT_SECURE_NO_WARNINGS
#include "Slang.hpp"
#include "SlangShaderSerialization.hpp"
#include "Common.hpp"

#include <etna/SlangShaderFormat.hpp>

#include <spdlog/spdlog.h>

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

// @TODO: proper error handling

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
  assert(dynlibHandle); // @TODO: proper handling

  SlangGlobalSessionDesc desc = {};
  auto gsRes = slang::createGlobalSession(&desc, globalSession.writeRef());
  assert(SLANG_SUCCEEDED(gsRes)); // @TODO: proper handling

  slang::SessionDesc sessionDesc = {};

  slang::TargetDesc targetDesc = {
    .format = SLANG_SPIRV, .profile = globalSession->findProfile("spirv_1_6")};
  sessionDesc.targets = &targetDesc;
  sessionDesc.targetCount = 1;

  std::vector<slang::CompilerOptionEntry> options = {
    {slang::CompilerOptionName::EmitSpirvDirectly, {slang::CompilerOptionValueKind::Int, 1}}};
  for (const auto& incDir : ci.commonIncludeDirs)
  {
    options.push_back(slang::CompilerOptionEntry{
      slang::CompilerOptionName::Include,
      slang::CompilerOptionValue{
        .kind = slang::CompilerOptionValueKind::String,
        .stringValue0 = to_char_str(incDir.string()).c_str()}});
  }

  sessionDesc.compilerOptionEntries = options.data();
  sessionDesc.compilerOptionEntryCount = uint32_t(options.size());

  auto sRes = globalSession->createSession(sessionDesc, session.writeRef());
  assert(SLANG_SUCCEEDED(sRes)); // @TODO: proper handling
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

static SlangShaderBytecodeRef* bytecode_slot(SlangShaderHeader* header, SlangStage stage)
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

  for (const auto& ep : entry_points)
  {
    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    {
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      slangModule->findEntryPointByName(ep.data(), entryPoint.writeRef());
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

    SlangStage stage = entryPointRefl->getStage();
    auto* bytecodeDest = bytecode_slot(header, stage);
    if (!bytecodeDest)
    {
      return -3;
    }
    if (slang_shader_ref_has_value(*bytecodeDest))
    {
      spdlog::error("Duplicate entry point {} for stage {}", ep, stage_name(stage));
      return -3;
    }

    if (stage == SLANG_STAGE_COMPUTE)
    {
      SlangUInt sizes[3];
      entryPointRefl->getComputeThreadGroupSize(3, sizes);
      header->refl.numThreadsX = sizes[0];
      header->refl.numThreadsY = sizes[1];
      header->refl.numThreadsZ = sizes[2];
    }

    // @TODO: fill resource reflection

    Slang::ComPtr<slang::IBlob> spirvCode;
    {
      Slang::ComPtr<slang::IBlob> diagnosticsBlob;
      SlangResult result = linkedProgram->getEntryPointCode(
        0, // entryPointIndex
        0, // targetIndex
        spirvCode.writeRef(),
        diagnosticsBlob.writeRef());
      diagnose_if_needed(diagnosticsBlob, !SLANG_SUCCEEDED(result));
      if (!SLANG_SUCCEEDED(result))
      {
        spdlog::error(
          "Can't generate spirv for program with entrypoint {} from {}", ep, source.string());
        return -3;
      }
    }

    *bytecodeDest = slang_shader_ser_write_bytecode(
      {(const uint8_t*)spirvCode->getBufferPointer(), spirvCode->getBufferSize()}, sctx);
  }

  if (
    slang_shader_ref_has_value(header->comp) &&
    (slang_shader_ref_has_value(header->vert) || slang_shader_ref_has_value(header->frag) ||
     slang_shader_ref_has_value(header->tesc) || slang_shader_ref_has_value(header->tese) ||
     slang_shader_ref_has_value(header->geom)))
  {
    spdlog::error("Can't require both graphics and compute entrypoints");
    return -3;
  }
  if (
    !slang_shader_ref_has_value(header->vert) &&
    (slang_shader_ref_has_value(header->frag) || slang_shader_ref_has_value(header->tesc) ||
     slang_shader_ref_has_value(header->tese) || slang_shader_ref_has_value(header->geom)))
  {
    spdlog::error("Graphics pipeline missing a vertex shader");
    return -3;
  }
  if (
    (slang_shader_ref_has_value(header->tesc) || slang_shader_ref_has_value(header->tese)) &&
    (!slang_shader_ref_has_value(header->tesc) || !slang_shader_ref_has_value(header->tese)))
  {
    spdlog::error("Tessellation stage requires both hull and domain shaders");
    return -3;
  }

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
