#define _CRT_SECURE_NO_WARNINGS
#include "Slang.hpp"

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

// @TODO: proper error handling

template <class TS>
  requires(std::same_as<TS, std::string> || std::same_as<TS, std::wstring>)
std::string to_char_str(const TS& s)
{
  if constexpr (std::same_as<TS, std::string>)
    return s;
  else
    return std::to_string(s);
}

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
    options.push_back(
      slang::CompilerOptionEntry{
        slang::CompilerOptionName::Include,
        slang::CompilerOptionValue{
          .kind = slang::CompilerOptionValueKind::String,
          .stringValue0 = to_char_str(incDir.string()).c_str()}});
  }

  sessionDesc.compilerOptionEntries = options.data();
  sessionDesc.compilerOptionEntryCount = options.size();

  auto sRes = globalSession->createSession(sessionDesc, session.writeRef());
  assert(SLANG_SUCCEEDED(sRes)); // @TODO: proper handling

  // @TEST
  spdlog::info(
    "SLANG: dynlib={}, globalSession={}, session={}",
    dynlibHandle,
    (void*)globalSession.get(),
    (void*)session.get());
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

int SlangCompiler::compile(
  const std::filesystem::path& source,
  std::string_view entry_point,
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

  Slang::ComPtr<slang::IEntryPoint> entryPoint;
  {
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slangModule->findEntryPointByName(entry_point.data(), entryPoint.writeRef());
    if (!entryPoint)
    {
      spdlog::info("Can't get entrypoint {} from {}", entry_point, source.string());
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
      spdlog::info(
        "Can't compose program with entrypoint {} from {}", entry_point, source.string());
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
      spdlog::info("Can't link program with entrypoint {} from {}", entry_point, source.string());
      return -3;
    }
  }

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
      spdlog::info(
        "Can't generate spirv for program with entrypoint {} from {}",
        entry_point,
        source.string());
      return -3;
    }
  }

  if (FILE* f = fopen(to_char_str(dest.string()).c_str(), "wb+"))
  {
    // @TODO: serialize header and reflection
    auto chunks = fwrite(spirvCode->getBufferPointer(), spirvCode->getBufferSize(), 1, f);
    assert(chunks == 1);
    fclose(f);
  }
  else
  {
    spdlog::info("Can't open output file {}", dest.string());
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
      spdlog::info("Can't open depfile {}", dest_depfile.string());
    }
  }

  spdlog::info(
    "Compiled module from {} with entrypoint {} to {} with depfile {}!",
    source.string(),
    entry_point,
    dest.string(),
    dest_depfile.string());
  return 0;
}
