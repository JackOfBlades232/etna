#pragma once
#ifndef ETNA_COPY_HELPER_HPP_INCLUDED
#define ETNA_COPY_HELPER_HPP_INCLUDED

#include <vector>
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
  void copyBufferToBuffer(Buffer &dst, Buffer &src, std::vector<vk::BufferCopy> regions);

  /* @TODO:
  * Sort out regions API
  * Small buffer update
  * Move fill/fillOnce/update here (so as to choose between small and regular update)
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
};

}

#endif // ETNA_COPY_HELPER_HPP_INCLUDED
