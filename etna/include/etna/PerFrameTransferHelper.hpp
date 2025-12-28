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

/** @TODO: update
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

  enum class ProcessingState
  {
    IDLE,
    READY,
    READBACK,
    READBACK_DONE,
    UPLOAD,
    UPLOAD_DONE,
  };

  explicit PerFrameTransferHelper(CreateInfo info);

  PerFrameTransferHelper(const PerFrameTransferHelper&) = delete;
  PerFrameTransferHelper& operator=(const PerFrameTransferHelper&) = delete;
  PerFrameTransferHelper(PerFrameTransferHelper&&) = delete;
  PerFrameTransferHelper& operator=(PerFrameTransferHelper&&) = delete;

  class FrameProcessor;

  class UploadProcessor
  {
  public:
    UploadProcessor(const UploadProcessor&) = delete;
    UploadProcessor& operator=(const UploadProcessor&) = delete;
    UploadProcessor(UploadProcessor&&) = delete;
    UploadProcessor& operator=(UploadProcessor&&) = delete;

    ~UploadProcessor();

    template <class T>
      requires std::is_trivially_copyable_v<T>
    bool uploadBufferSync(
      vk::CommandBuffer cmd_buf, Buffer& dst, uint32_t offset, std::span<T const> src)
    {
      std::span<std::byte const> raw{
        reinterpret_cast<const std::byte*>(src.data()), src.size_bytes()};
      return uploadBufferSync(cmd_buf, dst, offset, raw);
    }

    bool uploadBufferSync(
      vk::CommandBuffer cmd_buf, Buffer& dst, uint32_t offset, std::span<std::byte const> src)
    {
      return self->uploadBufferSync(cmd_buf, dst, offset, src);
    }

    bool uploadImageSync(
      vk::CommandBuffer cmd_buf,
      Image& dst,
      uint32_t mip_level,
      uint32_t layer,
      std::span<std::byte const> src)
    {
      return self->uploadImageSync(cmd_buf, dst, mip_level, layer, src);
    }

    bool progressBufferUploadAsync(vk::CommandBuffer cmd_buf, AsyncBufferUploadState& state)
    {
      return self->progressBufferUploadAsync(cmd_buf, state);
    }
    bool progressImageUploadAsync(vk::CommandBuffer cmd_buf, AsyncImageUploadState& state)
    {
      return self->progressImageUploadAsync(cmd_buf, state);
    }

    bool hasSpaceThisFrame() const { return self->curFrameStagingOffset < self->stagingSize; }

    operator bool() const { return self != nullptr; }

  private:
    PerFrameTransferHelper* self;

    explicit UploadProcessor(PerFrameTransferHelper* self)
      : self{self}
    {
    }

    friend class FrameProcessor;
  };

  class ReadbackProcessor
  {
  public:
    ReadbackProcessor(const ReadbackProcessor&) = delete;
    ReadbackProcessor& operator=(const ReadbackProcessor&) = delete;
    ReadbackProcessor(ReadbackProcessor&&) = delete;
    ReadbackProcessor& operator=(ReadbackProcessor&&) = delete;

    ~ReadbackProcessor();

    bool progressBufferReadbackAsync(vk::CommandBuffer cmd_buf, AsyncBufferReadbackState& state)
    {
      return self->progressBufferReadbackAsync(cmd_buf, state);
    }

    bool hasSpaceThisFrame() const { return self->curFrameStagingOffset < self->stagingSize; }

    operator bool() const { return self != nullptr; }

  private:
    PerFrameTransferHelper* self;

    explicit ReadbackProcessor(PerFrameTransferHelper* self)
      : self{self}
    {
    }

    friend class FrameProcessor;
  };

  class FrameProcessor
  {
  public:
    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;
    FrameProcessor(FrameProcessor&&) = delete;
    FrameProcessor& operator=(FrameProcessor&&) = delete;

    ~FrameProcessor();

    ReadbackProcessor beginReadback();
    UploadProcessor beginUpload();

    operator bool() const { return self != nullptr; }

  private:
    PerFrameTransferHelper* self;

    explicit FrameProcessor(PerFrameTransferHelper* self)
      : self{self}
    {
    }

    friend class PerFrameTransferHelper;
  };

  friend class UploadProcessor;
  friend class ReadbackProcessor;
  friend class FrameProcessor;

  FrameProcessor beginFrame();

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

private:
  ProcessingState state;
  uint64_t lastFrame;
  vk::DeviceSize stagingSize;
  vk::DeviceSize curFrameStagingOffset;
  GpuSharedResource<Buffer> stagingBuffer;
  const GpuWorkCount& wc;

  bool uploadBufferSync(
    vk::CommandBuffer cmd_buf, Buffer& dst, uint32_t offset, std::span<std::byte const> src);

  bool uploadImageSync(
    vk::CommandBuffer cmd_buf,
    Image& dst,
    uint32_t mip_level,
    uint32_t layer,
    std::span<std::byte const> src);

  bool progressBufferUploadAsync(vk::CommandBuffer cmd_buf, AsyncBufferUploadState& state);
  bool progressBufferReadbackAsync(vk::CommandBuffer cmd_buf, AsyncBufferReadbackState& state);
  bool progressImageUploadAsync(vk::CommandBuffer cmd_buf, AsyncImageUploadState& state);

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
