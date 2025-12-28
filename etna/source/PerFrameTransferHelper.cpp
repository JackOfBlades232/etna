#include <etna/PerFrameTransferHelper.hpp>
#include <etna/Etna.hpp>

#include <vulkan/vulkan_format_traits.hpp>
#include <type_traits>

namespace etna
{

template <class T>
  requires std::is_integral_v<T>
static T align_up(T val, T alignment)
{
  return ((val - 1) / alignment + 1) * alignment;
}

template <class T>
  requires std::is_integral_v<T>
static T align_down(T val, T alignment)
{
  return (val / alignment) * alignment;
}

static uint32_t offset3d_to_linear(vk::Offset3D o, vk::Extent3D e)
{
  return uint32_t(o.z) * e.height * e.width + uint32_t(o.y) * e.width + uint32_t(o.x);
}

static vk::Offset3D linear_to_offset3d(uint32_t l, vk::Extent3D e)
{
  return vk::Offset3D{
    int32_t(l % e.width), int32_t((l / e.width) % e.height), int32_t(l / (e.width * e.height))};
}

PerFrameTransferHelper::PerFrameTransferHelper(CreateInfo info)
  : state{ProcessingState::IDLE}
  , lastFrame{uint64_t(-1)}
  , stagingSize{info.stagingSize}
  , curFrameStagingOffset{0}
  , stagingBuffer{*info.wc, [&](size_t) {
                    return etna::get_context().createBuffer(
                      Buffer::CreateInfo{
                        .size = info.stagingSize,
                        .bufferUsage = vk::BufferUsageFlagBits::eTransferDst |
                          vk::BufferUsageFlagBits::eTransferSrc,
                        .memoryUsage = VMA_MEMORY_USAGE_AUTO,
                        .allocationCreate = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT,
                        .name = "PerFrameTransferHelper::stagingBuffer",
                      });
                  }}
  , wc{*info.wc}
{
  stagingBuffer.iterate([](auto& buf) { buf.map(); });
}

PerFrameTransferHelper::UploadProcessor::~UploadProcessor()
{
  if (!self)
    return;

  if (self->state == ProcessingState::UPLOAD)
    self->state = ProcessingState::UPLOAD_DONE;
  else
    ETNA_PANIC("PerFrameTransferHelper: upload scope must end before performing other actions.");
}

PerFrameTransferHelper::ReadbackProcessor::~ReadbackProcessor()
{
  if (!self)
    return;

  if (self->state == ProcessingState::READBACK)
    self->state = ProcessingState::READBACK_DONE;
  else
    ETNA_PANIC("PerFrameTransferHelper: readback scope must end before performing other actions.");
}

PerFrameTransferHelper::FrameProcessor::~FrameProcessor()
{
  if (!self)
    return;

  if (
    self->state == ProcessingState::READY || self->state == ProcessingState::READBACK_DONE ||
    self->state == ProcessingState::UPLOAD_DONE)
  {
    self->state = ProcessingState::IDLE;
    self->lastFrame = self->wc.batchIndex();
  }
  else
  {
    if (self->state == ProcessingState::READBACK)
      ETNA_PANIC("PerFrameTransferHelper: readback not finished at scope end.");
    else if (self->state == ProcessingState::UPLOAD)
      ETNA_PANIC("PerFrameTransferHelper: upload not finished at scope end.");
    else
      ETNA_PANIC("PerFrameTransferHelper: multiple scope ends.");
  }
}

PerFrameTransferHelper::ReadbackProcessor PerFrameTransferHelper::FrameProcessor::beginReadback()
{
  ETNA_ASSERT(self);

  if (self->state != ProcessingState::READY)
  {
    ETNA_PANIC("PerFrameTransferHelper: readbacks must be processed first.");
    return PerFrameTransferHelper::ReadbackProcessor{nullptr};
  }

  self->state = ProcessingState::READBACK;
  return PerFrameTransferHelper::ReadbackProcessor{self};
}

PerFrameTransferHelper::UploadProcessor PerFrameTransferHelper::FrameProcessor::beginUpload()
{
  ETNA_ASSERT(self);

  if (self->state != ProcessingState::READY && self->state != ProcessingState::READBACK_DONE)
  {
    ETNA_PANIC("PerFrameTransferHelper: uploads must be done after readbacks.");
    return PerFrameTransferHelper::UploadProcessor{nullptr};
  }

  self->state = ProcessingState::UPLOAD;
  self->curFrameStagingOffset = 0;
  return PerFrameTransferHelper::UploadProcessor{self};
}

PerFrameTransferHelper::FrameProcessor PerFrameTransferHelper::beginFrame()
{
  if (state != ProcessingState::IDLE)
  {
    ETNA_PANIC("PerFrameTransferHelper: already processing frame.");
    return PerFrameTransferHelper::FrameProcessor{nullptr};
  }
  else if (lastFrame == wc.batchIndex())
  {
    ETNA_PANIC("PerFrameTransferHelper: already processed frame {}.", lastFrame);
    return PerFrameTransferHelper::FrameProcessor{nullptr};
  }

  state = ProcessingState::READY;
  return PerFrameTransferHelper::FrameProcessor{this};
}

AsyncBufferUploadState PerFrameTransferHelper::startUploadBufferAsync(
  Buffer& dst, uint32_t offset, std::span<std::byte const> src) const
{
  return AsyncBufferUploadState{this, uint64_t(-1), &dst, offset, src};
}

AsyncBufferReadbackState PerFrameTransferHelper::startReadbackBufferAsync(
  std::span<std::byte> dst, const Buffer& src, uint32_t offset) const
{
  (void)dst, (void)src, (void)offset;
  ETNA_ASSERTF(0, "TODO: readback");
  return {};
}

AsyncImageUploadState PerFrameTransferHelper::startUploadImageAsync(
  Image& dst, uint32_t mip_level, uint32_t layer, std::span<std::byte const> src) const
{
  auto [w, h, d] = dst.getExtent();

  ETNA_ASSERTF(d == 1, "3D image async uploads are not implemented yet!");

  const size_t bytesPerPixel = vk::blockSize(dst.getFormat());
  [[maybe_unused]] const size_t imagePixelCount = w * h * d;

  ETNA_ASSERTF(
    imagePixelCount * bytesPerPixel == src.size(),
    "Image size mismatch between CPU and GPU! Expected {} bytes, but got {}!",
    imagePixelCount * bytesPerPixel,
    src.size());

  return AsyncImageUploadState{
    this, uint64_t(-1), &dst, mip_level, layer, bytesPerPixel, vk::Offset3D{0, 0, 0}, src};
}

bool PerFrameTransferHelper::uploadBufferSync(
  vk::CommandBuffer cmd_buf, Buffer& dst, uint32_t offset, std::span<std::byte const> src)
{
  ETNA_VERIFYF(offset % 4 == 0 && src.size() % 4 == 0, "All GPU access must be 16-byte aligned!");

  vk::DeviceSize stagingOffset = align_up(curFrameStagingOffset, vk::DeviceSize{16});
  if (stagingSize - stagingOffset < src.size())
    return false;

  memcpy(stagingBuffer.get().data() + stagingOffset, src.data(), src.size());
  transferBufferRegion(cmd_buf, dst, offset, stagingOffset, src.size());

  curFrameStagingOffset = stagingOffset + src.size();
  return true;
}

bool PerFrameTransferHelper::uploadImageSync(
  vk::CommandBuffer cmd_buf,
  Image& dst,
  uint32_t mip_level,
  uint32_t layer,
  std::span<std::byte const> src)
{
  auto [w, h, d] = dst.getExtent();

  const size_t bytesPerPixel = vk::blockSize(dst.getFormat());
  [[maybe_unused]] const size_t imagePixelCount = w * h * d;

  ETNA_ASSERTF(
    imagePixelCount * bytesPerPixel == src.size(),
    "Image size mismatch between CPU and GPU! Expected {} bytes, but got {}!",
    imagePixelCount * bytesPerPixel,
    src.size());

  vk::DeviceSize stagingOffset = align_up(curFrameStagingOffset, vk::DeviceSize{bytesPerPixel});
  if (stagingSize - stagingOffset < src.size())
    return false;

  memcpy(stagingBuffer.get().data() + stagingOffset, src.data(), src.size());

  etna::set_state(
    cmd_buf,
    dst.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite,
    vk::ImageLayout::eTransferDstOptimal,
    dst.getAspectMaskByFormat());
  etna::flush_barriers(cmd_buf);

  transferImageRect(
    cmd_buf, dst, mip_level, layer, vk::Offset3D{0, 0, 0}, vk::Extent3D{w, h, d}, stagingOffset);

  etna::set_state(
    cmd_buf,
    dst.get(),
    {},
    {},
    vk::ImageLayout::eShaderReadOnlyOptimal,
    dst.getAspectMaskByFormat());
  etna::flush_barriers(cmd_buf);

  return true;
}

bool PerFrameTransferHelper::progressBufferUploadAsync(
  vk::CommandBuffer cmd_buf, AsyncBufferUploadState& state)
{
  ETNA_ASSERTF(
    state.lastFrame != wc.batchIndex(),
    "Attempting to upload the same buffer twice on frame {}",
    state.lastFrame);
  ETNA_ASSERT(!state.done());
  ETNA_ASSERT(state.transferHelper == this);

  vk::DeviceSize stagingOffset = align_up(curFrameStagingOffset, vk::DeviceSize{16});
  size_t transferSize =
    std::min(state.src.size(), align_down(stagingSize - stagingOffset, vk::DeviceSize{16}));
  if (transferSize == 0)
    return false;

  memcpy(stagingBuffer.get().data() + stagingOffset, state.src.data(), state.src.size());
  transferBufferRegion(cmd_buf, *state.dst, state.offset, stagingOffset, transferSize);

  curFrameStagingOffset = stagingOffset + transferSize;
  state.offset += uint32_t(transferSize);
  state.src = state.src.subspan(transferSize);

  state.lastFrame = wc.batchIndex();

  return state.done();
}

bool PerFrameTransferHelper::progressBufferReadbackAsync(
  vk::CommandBuffer cmd_buf, AsyncBufferReadbackState& state)
{
  (void)cmd_buf, (void)state;
  ETNA_ASSERTF(0, "TODO: readback");
  return {};
}

bool PerFrameTransferHelper::progressImageUploadAsync(
  vk::CommandBuffer cmd_buf, AsyncImageUploadState& state)
{
  ETNA_ASSERTF(
    state.lastFrame != wc.batchIndex(),
    "Attempting to upload the same image twice on frame {}",
    state.lastFrame);
  ETNA_ASSERT(!state.done());
  ETNA_ASSERT(state.transferHelper == this);

  vk::DeviceSize stagingOffset =
    align_up(curFrameStagingOffset, vk::DeviceSize{state.bytesPerPixel});
  size_t transferSize = std::min(
    state.src.size(), align_down(stagingSize - stagingOffset, vk::DeviceSize{state.bytesPerPixel}));
  if (transferSize == 0)
    return false;

  memcpy(stagingBuffer.get().data() + stagingOffset, state.src.data(), state.src.size());

  if (state.offset == vk::Offset3D{0, 0, 0})
  {
    etna::set_state(
      cmd_buf,
      state.dst->get(),
      vk::PipelineStageFlagBits2::eTransfer,
      vk::AccessFlagBits2::eTransferWrite,
      vk::ImageLayout::eTransferDstOptimal,
      state.dst->getAspectMaskByFormat());
    etna::flush_barriers(cmd_buf);
  }

  transferImageRegion(
    cmd_buf,
    *state.dst,
    state.mipLevel,
    state.layer,
    state.bytesPerPixel,
    state.offset,
    stagingOffset,
    transferSize);

  vk::Extent3D imageExtent = state.dst->getExtent();
  curFrameStagingOffset = stagingOffset + transferSize;
  state.offset = linear_to_offset3d(
    offset3d_to_linear(state.offset, imageExtent) + transferSize / state.bytesPerPixel,
    imageExtent);
  state.src = state.src.subspan(transferSize);

  if (state.done())
  {
    etna::set_state(
      cmd_buf,
      state.dst->get(),
      {},
      {},
      vk::ImageLayout::eShaderReadOnlyOptimal,
      state.dst->getAspectMaskByFormat());
    etna::flush_barriers(cmd_buf);
  }

  state.lastFrame = wc.batchIndex();

  return state.done();
}

void PerFrameTransferHelper::transferBufferRegion(
  vk::CommandBuffer cmd_buf,
  Buffer& dst,
  uint32_t offset,
  vk::DeviceSize staging_offset,
  size_t size)
{
  vk::BufferCopy2 copy{
    .srcOffset = staging_offset,
    .dstOffset = offset,
    .size = size,
  };
  vk::CopyBufferInfo2 info{
    .srcBuffer = stagingBuffer.get().get(),
    .dstBuffer = dst.get(),
    .regionCount = 1,
    .pRegions = &copy,
  };
  cmd_buf.copyBuffer2(info);
}

void PerFrameTransferHelper::transferImageRegion(
  vk::CommandBuffer cmd_buf,
  Image& dst,
  uint32_t mip_level,
  uint32_t layer,
  size_t bytes_per_pixel,
  vk::Offset3D offset,
  vk::DeviceSize staging_offset,
  size_t size)
{
  vk::Extent3D imageExtent = dst.getExtent();
  vk::Offset3D finalOffset = linear_to_offset3d(
    offset3d_to_linear(offset, imageExtent) + size / bytes_per_pixel, imageExtent);

  ETNA_ASSERT(imageExtent.depth == 1);
  ETNA_ASSERT(offset.z == 0);
  ETNA_ASSERT(
    finalOffset.z == 0 || (finalOffset.z == 1 && finalOffset.y == 0 && finalOffset.x == 0));

  if (offset.y == finalOffset.y || (finalOffset.y == offset.y + 1 && finalOffset.x == 0))
  {
    transferImageRect(
      cmd_buf,
      dst,
      mip_level,
      layer,
      offset,
      vk::Extent3D{uint32_t(finalOffset.x - offset.x), 1, 1},
      staging_offset);
    return;
  }

  if (offset.x > 0)
  {
    const uint32_t firstLinePixels = imageExtent.width - offset.x;
    transferImageRect(
      cmd_buf, dst, mip_level, layer, offset, vk::Extent3D{firstLinePixels, 1, 1}, staging_offset);
    staging_offset += firstLinePixels * bytes_per_pixel;
    ++offset.y;
    offset.x = 0;
  }

  const uint32_t fullLines = finalOffset.y - offset.y;
  transferImageRect(
    cmd_buf,
    dst,
    mip_level,
    layer,
    offset,
    vk::Extent3D{imageExtent.width, fullLines, 1},
    staging_offset);
  staging_offset += fullLines * imageExtent.width * bytes_per_pixel;
  offset.y += fullLines;

  if (finalOffset.x > 0)
  {
    transferImageRect(
      cmd_buf,
      dst,
      mip_level,
      layer,
      offset,
      vk::Extent3D{uint32_t(finalOffset.x), 1, 1},
      staging_offset);
  }
}

void PerFrameTransferHelper::transferImageRect(
  vk::CommandBuffer cmd_buf,
  Image& dst,
  uint32_t mip_level,
  uint32_t layer,
  vk::Offset3D offset,
  vk::Extent3D extent,
  vk::DeviceSize staging_offset)
{
  vk::BufferImageCopy2 copy{
    .bufferOffset = staging_offset,
    .bufferRowLength = 0,
    .bufferImageHeight = 0,
    .imageSubresource =
      vk::ImageSubresourceLayers{
        .aspectMask = dst.getAspectMaskByFormat(),
        .mipLevel = mip_level,
        .baseArrayLayer = layer,
        .layerCount = 1,
      },
    .imageOffset = offset,
    .imageExtent = extent,
  };
  vk::CopyBufferToImageInfo2 info{
    .srcBuffer = stagingBuffer.get().get(),
    .dstImage = dst.get(),
    .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
    .regionCount = 1,
    .pRegions = &copy,
  };
  cmd_buf.copyBufferToImage2(info);
}

} // namespace etna
