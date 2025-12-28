#pragma once
#ifndef ETNA_PER_FRAME_TRANSFER_HELPER_HPP_INCLUDED
#define ETNA_PER_FRAME_TRANSFER_HELPER_HPP_INCLUDED

#include <etna/Vulkan.hpp>
#include <etna/Buffer.hpp>
#include <etna/Image.hpp>
#include <etna/GpuSharedResource.hpp>

#include <type_traits>

namespace etna
{

class PerFrameTransferHelper;

// @TODO: consider moving dst (and maybe src too) out of the async structs
// @TODO: consider not requiring stable transferHelper for uploads

struct AsyncBufferUploadState
{
  const PerFrameTransferHelper* transferHelper;
  uint64_t lastFrame;
  Buffer* dst;
  uint32_t offset;
  std::span<std::byte const> src;

  bool done() const { return transferHelper && src.size() == 0; }
};

struct AsyncBufferReadbackState
{
  // @TODO
};

struct AsyncImageUploadState
{
  const PerFrameTransferHelper* transferHelper;
  uint64_t lastFrame;
  Image* dst;
  uint32_t mipLevel;
  uint32_t layer;
  size_t bytesPerPixel;
  vk::Offset3D offset;
  std::span<std::byte const> src;

  bool done() const { return transferHelper && src.size() == 0; }
};

/**
 * "non-blocking" GPU-CPU transfer helper:
 * Accepts upload/readback requests and serves them inside update call.
 * Sync API is for small-sized one frame uploads (no readback).
 * Functions return true if transfer was issued, false if out of staging mem.
 * Async API is for bigger transfers to be done over multiple frames.
 * start*Async initizalize an Async*State structure, not transferring yet.
 * progress*Async transfer as much data as fits in staging and update state.
 * progress calls return false if out of staging.
 * endFrame must be called at the end of frame to reset the staging buffer.
 */
class PerFrameTransferHelper
{
public:
  struct CreateInfo
  {
    vk::DeviceSize stagingSize;
    const GpuWorkCount* wc;
  };

  explicit PerFrameTransferHelper(CreateInfo info);

  PerFrameTransferHelper(const PerFrameTransferHelper&) = delete;
  PerFrameTransferHelper& operator=(const PerFrameTransferHelper&) = delete;
  PerFrameTransferHelper(PerFrameTransferHelper&&) = delete;
  PerFrameTransferHelper& operator=(PerFrameTransferHelper&&) = delete;

  template <class T>
    requires std::is_trivially_copyable_v<T>
  bool uploadBufferSync(
    vk::CommandBuffer cmd_buf, Buffer& dst, uint32_t offset, std::span<T const> src)
  {
    std::span<std::byte const> raw{
      reinterpret_cast<const std::byte*>(src.data()), src.size_bytes()};
    return uploadBuffer(cmd_buf, dst, offset, raw);
  }

  bool uploadBufferSync(
    vk::CommandBuffer cmd_buf, Buffer& dst, uint32_t offset, std::span<std::byte const> src);

  bool uploadImageSync(
    vk::CommandBuffer cmd_buf,
    Image& dst,
    uint32_t mip_level,
    uint32_t layer,
    std::span<std::byte const> src);

  template <class T>
    requires std::is_trivially_copyable_v<T>
  AsyncBufferUploadState startUploadBufferAsync(
    Buffer& dst, uint32_t offset, std::span<T const> src) const
  {
    std::span<std::byte const> raw{
      reinterpret_cast<const std::byte*>(src.data()), src.size_bytes()};
    return startUploadBufferAsync(dst, offset, raw);
  }

  AsyncBufferUploadState startUploadBufferAsync(
    Buffer& dst, uint32_t offset, std::span<std::byte const> src) const;

  template <class T>
    requires std::is_trivially_copyable_v<T>
  AsyncBufferReadbackState startReadbackBufferAsync(
    std::span<T> dst, const Buffer& src, uint32_t offset) const
  {
    std::span<std::byte> raw{reinterpret_cast<std::byte*>(dst.data()), dst.size_bytes()};
    return startReadbackBufferAsync(raw, src, offset);
  }

  AsyncBufferReadbackState startReadbackBufferAsync(
    std::span<std::byte> dst, const Buffer& src, uint32_t offset) const;

  // @NOTE: for now doesn't support 3D images (unlike sync API)
  AsyncImageUploadState startUploadImageAsync(
    Image& dst, uint32_t mip_level, uint32_t layer, std::span<std::byte const> src) const;

  bool progressBufferUploadAsync(vk::CommandBuffer cmd_buf, AsyncBufferUploadState& state);
  bool progressBufferReadbackAsync(vk::CommandBuffer cmd_buf, AsyncBufferReadbackState& state);
  bool progressImageUploadAsync(vk::CommandBuffer cmd_buf, AsyncImageUploadState& state);

  // @TODO: somehow validate the call order
  void endFrame();

private:
  vk::DeviceSize stagingSize;
  vk::DeviceSize curFrameStagingOffset;
  GpuSharedResource<Buffer> stagingBuffer;
  const GpuWorkCount& wc;

  void transferBufferRegion(
    vk::CommandBuffer cmd_buf,
    Buffer& dst,
    uint32_t offset,
    vk::DeviceSize staging_offset,
    size_t size);

  void transferImageRegion(
    vk::CommandBuffer cmd_buf,
    Image& dst,
    uint32_t mip_level,
    uint32_t layer,
    size_t bytes_per_pixel,
    vk::Offset3D offset,
    vk::DeviceSize staging_offset,
    size_t size);

  void transferImageRect(
    vk::CommandBuffer cmd_buf,
    Image& dst,
    uint32_t mip_level,
    uint32_t layer,
    vk::Offset3D offset,
    vk::Extent3D extent,
    vk::DeviceSize staging_offset);
};

} // namespace etna

#endif
