#include <etna/BindingItems.hpp>
#include <etna/Buffer.hpp>
#include <etna/CopyHelper.hpp>
#include <etna/GlobalContext.hpp>
#include <vulkan/vulkan_enums.hpp>
#include "DebugUtils.hpp"

#include <iostream>


namespace etna
{

Buffer::Buffer(VmaAllocator alloc, CreateInfo info)
  : allocator{ alloc }
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

void Buffer::createUpdateBuffer(std::size_t size)
{
  auto &copyHelper = get_context().getCopyHelper();
  // @TODO: make spec names based on this buffer's name
  updateStagingBuffer = std::make_shared<Buffer>(copyHelper.createStagingBuffer(size, "dedicated_staging_buffer"));
  updateStagingBuffer->map();
}

void Buffer::setUpdateBuffer(const std::shared_ptr<Buffer> &buff)
{
  updateStagingBuffer = buff;
}

void Buffer::fill(std::byte *src, std::size_t size)
{
  if (!updateStagingBuffer)
    createUpdateBuffer(size);

  memcpy(updateStagingBuffer->data(), src, size);
  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.copyBufferToBuffer(*this, *updateStagingBuffer, {{0, 0, size}});
}

void Buffer::fillOnce(std::byte *src, std::size_t size)
{
  auto &copyHelper = get_context().getCopyHelper();
  Buffer tmpStagingBuff = copyHelper.createStagingBuffer(size);
  memcpy(tmpStagingBuff.map(), src, size);
  copyHelper.copyBufferToBuffer(*this, tmpStagingBuff, {{0, 0, size}});
}

void Buffer::update(std::byte *src, std::size_t size)
{
  ETNA_ASSERT(updateStagingBuffer);
  memcpy(updateStagingBuffer->data(), src, size);

  auto &copyHelper = get_context().getCopyHelper();
  copyHelper.copyBufferToBuffer(*this, *updateStagingBuffer, {{0, 0, size}});
}

BufferBinding Buffer::genBinding(vk::DeviceSize offset, vk::DeviceSize range) const
{
  return BufferBinding{*this, vk::DescriptorBufferInfo {get(), offset, range}};
}

}
