#define _CRT_SECURE_NO_WARNINGS
#include <etna/Slang.hpp>

#include <etna/Assert.hpp>

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <optional>
#include <filesystem>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace etna
{

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

SlangRuntime::SlangRuntime(CreateInfo&& ci)
{
  (void)ci;

  dynlibHandle = try_load_slang_dynlib();
  ETNA_ASSERT(dynlibHandle); // @TODO: proper handling

  SlangGlobalSessionDesc desc = {};
  auto gsRes = slang::createGlobalSession(&desc, globalSession.writeRef());
  ETNA_ASSERT(SLANG_SUCCEEDED(gsRes)); // @TODO: proper handling

  slang::SessionDesc sessionDesc = {};
  auto sRes = globalSession->createSession(sessionDesc, session.writeRef());
  ETNA_ASSERT(SLANG_SUCCEEDED(sRes)); // @TODO: proper handling

  // @TEST
  spdlog::info(
    "SLANG: dynlib={}, globalSession={}, session={}",
    dynlibHandle,
    (void*)globalSession.get(),
    (void*)session.get());
}

SlangRuntime::~SlangRuntime()
{
  if (session)
    session.setNull();
  if (globalSession)
    globalSession.setNull();
  if (dynlibHandle)
    unload_dynlib(dynlibHandle);
}

} // namespace etna
