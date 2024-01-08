#include <etna/BindingItems.hpp>
#include <etna/Buffer.hpp>
#include <etna/CopyHelper.hpp>
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
  std::swap(updateStagingBuffer, other.updateStagingBuffer);
  std::swap(updateBufferOffset, other.updateBufferOffset);
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

void Buffer::createUpdateBuffer()
{
  auto &copyHelper = get_context().getCopyHelper();
  updateStagingBuffer = std::make_shared<Buffer>(copyHelper.createStagingBuffer(size, "dedicated_staging_buffer"));
  updateStagingBuffer->map();
}

// @TODO: messages in assertions
void Buffer::setUpdateBuffer(const std::shared_ptr<Buffer> &buff, std::size_t offset)
{
  ETNA_ASSERT(offset + size < buff->size);
  updateStagingBuffer = buff;
  updateBufferOffset  = offset;
}

void Buffer::fill(const std::byte *src, std::size_t srcSize)
{
  ETNA_ASSERT(size == srcSize);
  if (!updateStagingBuffer)
    createUpdateBuffer();

  update(src, size);
}

void Buffer::fillOnce(const std::byte *src, std::size_t srcSize)
{
  ETNA_ASSERT(size == srcSize);
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.updateBuffer(*this, 0, src, size);
}

void Buffer::update(const std::byte *src, std::size_t srcSize)
{
  ETNA_ASSERT(updateStagingBuffer && size == srcSize);
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.updateBuffer(*this, 0, src, size, updateStagingBuffer.get());
}

BufferBinding Buffer::genBinding(vk::DeviceSize offset, vk::DeviceSize range) const
{
  return BufferBinding{*this, vk::DescriptorBufferInfo {get(), offset, range}};
}

}
