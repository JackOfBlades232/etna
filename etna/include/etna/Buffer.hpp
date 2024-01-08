#pragma once
#ifndef ETNA_BUFFER_HPP_INCLUDED
#define ETNA_BUFFER_HPP_INCLUDED

#include <optional>
#include <string_view>
#include <etna/Vulkan.hpp>
#include <vk_mem_alloc.h>


namespace etna
{

struct BufferBinding;
class CopyHelper;

class Buffer
{
public:
  Buffer() = default;

  struct CreateInfo
  {
    std::size_t size;
    vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer;
    VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
    std::string_view name;
  };

  Buffer(VmaAllocator alloc, CreateInfo info);

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void swap(Buffer& other);
  Buffer(Buffer&&) noexcept;
  Buffer& operator=(Buffer&&) noexcept;

  [[nodiscard]] vk::Buffer get() const { return buffer; }
  [[nodiscard]] std::byte* data() { return mapped; }

  BufferBinding genBinding(vk::DeviceSize offset = 0, vk::DeviceSize range = VK_WHOLE_SIZE) const;

  // @TODO: make a restriction? (buffer is cpu visible)
  std::byte* map();
  void unmap();

  // @TODO: improve this comment and sort out this functionality
  // @NOTE: in this implementation the staging buffer matches the main buffer 1 to 1
  // except for the case when we set an existing buffer as staging, then we can only
  // use a contiguous piece of it.
  void createUpdateBuffer();
  void setUpdateBuffer(const std::shared_ptr<Buffer> &buff, std::size_t offset);
  void fill(const std::byte *src, std::size_t srcSize);
  void fillOnce(const std::byte *src, std::size_t srcSize);
  void update(const std::byte *src, std::size_t srcSize);

  ~Buffer();
  void reset();

private:
  friend class CopyHelper;

  VmaAllocator allocator{};

  VmaAllocation allocation{};
  vk::Buffer buffer{};
  std::byte* mapped{};
  std::size_t size{};

  std::shared_ptr<Buffer> updateStagingBuffer{};
  std::size_t updateBufferOffset{};
};

}

#endif // ETNA_BUFFER_HPP_INCLUDED
