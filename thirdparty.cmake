cmake_minimum_required(VERSION 3.25)


find_package(Vulkan 1.4.357 REQUIRED)
find_package(Slang 2026.13.1 REQUIRED)

# GPU-side allocator for Vulkan by AMD
CPMAddPackage(
  NAME VulkanMemoryAllocator
  GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  # A little bit after v3.1.0 to get nice cmake
  GIT_TAG b8e57472fffa3bd6e0a0b675f4615bf0a823ec4d
)
# VMA headers emit a bunch of warnings >:(
set_property(TARGET VulkanMemoryAllocator PROPERTY SYSTEM TRUE)

# Official library for parsing SPIRV bytecode
# We grab the exact SDK version from github so that
# you don't have to remember to mark a checkbox while installing the SDK =)
CPMAddPackage(
  NAME SpirvReflect
  GITHUB_REPOSITORY KhronosGroup/SPIRV-Reflect
  GIT_TAG "vulkan-sdk-${Vulkan_VERSION}"
  OPTIONS
    "SPIRV_REFLECT_EXECUTABLE OFF"
    "SPIRV_REFLECT_STRIPPER OFF"
    "SPIRV_REFLECT_EXAMPLES OFF"
    "SPIRV_REFLECT_BUILD_TESTS OFF"
    "SPIRV_REFLECT_STATIC_LIB ON"
)
# SpirvReflect headers also emit warnings now
set_property(TARGET spirv-reflect-static PROPERTY SYSTEM TRUE)

CPMAddPackage(
  NAME SpirvHeaders
  GITHUB_REPOSITORY KhronosGroup/SPIRV-Headers
  GIT_TAG "vulkan-sdk-${Vulkan_VERSION}"
  OPTIONS
    "SPIRV_HEADERS_EXECUTABLE OFF"
    "SPIRV_HEADERS_STRIPPER OFF"
    "SPIRV_HEADERS_EXAMPLES OFF"
    "SPIRV_HEADERS_BUILD_TESTS OFF"
    "SPIRV_HEADERS_STATIC_LIB ON"
)
CPMAddPackage(
  NAME SpirvTools
  GITHUB_REPOSITORY KhronosGroup/SPIRV-Tools
  GIT_TAG "vulkan-sdk-${Vulkan_VERSION}"
  OPTIONS
    "SKIP_SPIRV_TOOLS_INSTALL ON"
    "SPIRV_BUILD_FUZZER OFF"
    "SPIRV_BUILD_COMPRESSION OFF"
    "SPIRV_BUILD_LIBFUZZER_TARGETS OFF"
    "SPIRV_SKIP_EXECUTABLES ON"
    "SPIRV_SKIP_TESTS ON"
    "SPIRV_TOOLS_BUILD_STATIC ON"
)
set_property(TARGET SPIRV-Tools-static PROPERTY SYSTEM TRUE)

# Fmt is a dependency of spdlog, but to be
# safe we explicitly specify our version
CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.2.0
  OPTIONS
    "FMT_SYSTEM_HEADERS"
    "FMT_DOC OFF"
    "FMT_INSTALL OFF"
    "FMT_TEST OFF"
)

# Simple logging without headaches
CPMAddPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  VERSION 1.15.3
  OPTIONS
    SPDLOG_FMT_EXTERNAL
)

# A profiler for both CPU and GPU
CPMAddPackage(
  GITHUB_REPOSITORY wolfpld/tracy
  GIT_TAG v0.11.1
  OPTIONS
    "TRACY_ON_DEMAND ON"
)
