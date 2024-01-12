#include <etna/BindingItems.hpp>
#include <etna/Buffer.hpp>
#include <etna/GlobalContext.hpp>
#include <vulkan/vulkan_enums.hpp>
#include "DebugUtils.hpp"

#include <string>


namespace etna
{

Buffer::Buffer(VmaAllocator alloc, CreateInfo info)
  : allocator{ alloc }, size{ info.size }
{
  vk::BufferCreateInfo buf_info{
    .size = info.size,
    .usage = info.bufferUsage,
    .sharingMode = vk::SharingMode::eExclusive
  };

  VmaAllocationCreateInfo alloc_info{
    .flags = 0,
    .usage = info.memoryUsage,
    .requiredFlags = 0,
    .preferredFlags = 0,
    .memoryTypeBits = 0,
    .pool = nullptr,
    .pUserData = nullptr,
    .priority = 0.f
  };

  VkBuffer buf;
  auto retcode = vmaCreateBuffer(allocator, &static_cast<const VkBufferCreateInfo&>(buf_info), &alloc_info,
    &buf, &allocation, nullptr);
  // Note that usually vulkan.hpp handles doing the assertion
  // and a pretty message, but VMA cannot do that.
  ETNA_ASSERTF(retcode == VK_SUCCESS,
    "Error {} occurred while trying to allocate an etna::Buffer!",
    vk::to_string(static_cast<vk::Result>(retcode)));
  buffer = vk::Buffer(buf);
  etna::set_debug_name(buffer, info.name.data());
}

void Buffer::swap(Buffer& other)
{
  std::swap(allocator, other.allocator);
  std::swap(allocation, other.allocation);
  std::swap(buffer, other.buffer);
  std::swap(mapped, other.mapped);
  std::swap(size, other.size);
  std::swap(stagingBuffer, other.stagingBuffer);
  std::swap(stagingBufferOffset, other.stagingBufferOffset);
}

Buffer::Buffer(Buffer&& other) noexcept
{
  swap(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
  if (this == &other)
    return *this;

  reset();
  swap(other);

  return *this;
}

Buffer::~Buffer()
{
  reset();
}

void Buffer::reset()
{
  if (!buffer)
    return;

  if (mapped != nullptr)
    unmap();

  vmaDestroyBuffer(allocator, VkBuffer(buffer), allocation);
  allocator = {};
  allocation = {};
  buffer = vk::Buffer{};
}

std::byte* Buffer::map()
{
  void* result;

  // I can't think of a use case where failing to do a mapping
  // is acceptable and recoverable from.
  auto retcode = vmaMapMemory(allocator, allocation, &result);
  ETNA_ASSERTF(retcode == VK_SUCCESS,
    "Error %s occurred while trying to map an etna::Buffer!",
    vk::to_string(static_cast<vk::Result>(retcode)));

  return mapped = static_cast<std::byte*>(result);
}

void Buffer::unmap()
{
  ETNA_ASSERT(mapped != nullptr);
  vmaUnmapMemory(allocator, allocation);
  mapped = nullptr;
}

void Buffer::createStagingBuffer(StagingBufferType type)
{
  auto &copyHelper = get_context().getCopyHelper();
  stagingBuffer = std::make_shared<Buffer>(copyHelper.createStagingBuffer(size, type, "dedicated_staging_buffer"));
  stagingBuffer->map();
  stagingBufferOffset = 0;
  stagingBufferType = type;
}

void Buffer::update(const std::byte *src, std::size_t srcSize)
{
  ETNA_ASSERT(size == srcSize);
  ETNA_ASSERT(stagingBufferIsCpuToGpu());
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.updateBuffer(*this, 0, src, size, stagingBuffer ? stagingBuffer.get() : nullptr, stagingBufferOffset);
}

void Buffer::updateOnce(const std::byte *src, std::size_t srcSize)
{
  ETNA_ASSERT(size == srcSize);
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.updateBuffer(*this, 0, src, size);
}

void Buffer::read(std::byte *dst, std::size_t dstSize)
{
  ETNA_ASSERT(size == dstSize);
  ETNA_ASSERT(stagingBufferIsGpuToCpu());
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.readBuffer(*this, 0, dst, size, stagingBuffer.get(), stagingBufferOffset);
}

void Buffer::readOnce(std::byte *dst, std::size_t dstSize)
{
  ETNA_ASSERT(size == dstSize);
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.readBuffer(*this, 0, dst, size);
}

void Buffer::setStagingBuffer(const std::shared_ptr<Buffer> &buff, std::size_t offset)
{
  ETNA_ASSERT(offset + size < buff->size);
  stagingBuffer = buff;
  stagingBufferOffset  = offset;
}

void Buffer::updateFromStagingBuffer()
{
  ETNA_ASSERT(stagingBuffer);
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.copyBufferToBuffer(*this, *stagingBuffer, {{stagingBufferOffset, 0, size}});
}

BufferBinding Buffer::genBinding(vk::DeviceSize offset, vk::DeviceSize range) const
{
  return BufferBinding{*this, vk::DescriptorBufferInfo {get(), offset, range}};
}

bool Buffer::needsStagingBuffer()
{
  return CopyHelper::bufferNeedsStagingToUpdate(*this);
}

bool Buffer::stagingBufferIsCpuToGpu()
{
  return stagingBuffer && (stagingBufferType & StagingBufferType::eCpuToGpu);
}

bool Buffer::stagingBufferIsGpuToCpu()
{
  return stagingBuffer && (stagingBufferType & StagingBufferType::eGpuToCpu);
}

}
