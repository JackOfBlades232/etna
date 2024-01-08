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

  Buffer createStagingBuffer(std::size_t size, const char *name = nullptr);
  void copyBufferToBuffer(Buffer &dst, Buffer &src, const std::vector<vk::BufferCopy> &regions);
  void updateBuffer(Buffer &dst, vk::DeviceSize dstOffset,
                    const std::byte *src, std::size_t size, 
                    Buffer *stagingBuff = nullptr);

  /* @TODO:
  * Impl read buffer
  * Implement load for images
  * @HUH: do we need update/read/staging for images?
  */

private:
  GlobalContext *context;

  // @NOTE: this is not thread safe
  vk::Device device;
  vk::Queue transferQueue;
  vk::CommandPool cmdPool;
  vk::CommandBuffer cmdBuff;

  static constexpr std::size_t SMALL_BUFF_SIZE = 65536;

  // @TODO: improve, this is unclear
  // @NOTE: cmds must be [...](cmdBuff){cmdBuff.cmd***(); xN}
  void executeCommands(const std::function<void(vk::CommandBuffer &)> &cmds);
};

}

#endif // ETNA_COPY_HELPER_HPP_INCLUDED
