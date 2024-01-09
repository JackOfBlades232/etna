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

  std::byte* map();
  void unmap();

  // Updates are performed with a staging buffer and update the whole buffer

  // Creates a dedicated staging buffer of the same size as the main buffer
  void createUpdateBuffer();
  // Creates an update buffer if needed and update with data from the CPU 
  void fill(const std::byte *src, std::size_t srcSize);
  // Create a temporary staging buffer and update with data from the CPU
  void fillOnce(const std::byte *src, std::size_t srcSize);
  // Update with data from the CPU using the existing dedicated staging buffer
  void update(const std::byte *src, std::size_t srcSize);

  // Sets an existing buffer as the staging buffer (useful if you use one staging buffer for multiple gpu local ones)
  // and it's contiguous region at an offset is used for udpates
  void setUpdateBuffer(const std::shared_ptr<Buffer> &buff, std::size_t offset);
  // Update with data in the existing dedicated staging buffer
  void updateFromStagingBuffer();

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
