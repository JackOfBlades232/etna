#pragma once
#ifndef ETNA_COPY_HELPER_HPP_INCLUDED
#define ETNA_COPY_HELPER_HPP_INCLUDED

#include <vector>
#include <functional>
#include <etna/Buffer.hpp>
#include <etna/Vulkan.hpp>


namespace etna
{
class GlobalContext;

class CopyHelper
{
public:
  CopyHelper(GlobalContext *ctx, vk::Queue queue);
  ~CopyHelper();

  // @TODO: maybe not assert but return results if failed?
  Buffer createStagingBuffer(std::size_t size, StagingBufferType type, const char *name = nullptr) const;
  void copyBufferToBuffer(Buffer &dst, Buffer &src, const std::vector<vk::BufferCopy> &regions);
  void updateBuffer(Buffer &dst, vk::DeviceSize dstOffset, 
                    const std::byte *src, std::size_t size, 
                    Buffer *stagingBuff = nullptr, vk::DeviceSize stagingOffset = 0);
  void readBuffer(Buffer &src, vk::DeviceSize srcOffset, 
                  std::byte *dst, std::size_t size, 
                  Buffer *stagingBuff = nullptr, vk::DeviceSize stagingOffset = 0);

  static inline bool bufferNeedsStagingToUpdate(const Buffer &buff) { return buff.size > UPDATE_BUFFER_CMD_SIZE_LIMIT; }

  /* @TODO:
  * Implement load & update for images
  */

private:
  GlobalContext *context;

  // @NOTE: this is not thread safe
  vk::Device device;
  vk::Queue transferQueue;
  vk::CommandPool cmdPool;
  vk::CommandBuffer cmdBuff;

  static constexpr std::size_t UPDATE_BUFFER_CMD_SIZE_LIMIT = 65535;

  void executeCommands(const std::function<void(vk::CommandBuffer &)> &cmds);
};

}

#endif // ETNA_COPY_HELPER_HPP_INCLUDED
