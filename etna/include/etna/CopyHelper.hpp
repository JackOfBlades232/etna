#pragma once
#ifndef ETNA_COPY_HELPER_HPP_INCLUDED
#define ETNA_COPY_HELPER_HPP_INCLUDED

#include <vector>
#include <functional>
#include <etna/Buffer.hpp>
#include <etna/Image.hpp>
#include <etna/Vulkan.hpp>


namespace etna
{
class GlobalContext;

class CopyHelper
{
public:
  CopyHelper(GlobalContext *ctx, vk::Queue queue);
  ~CopyHelper();

  Buffer createStagingBuffer(std::size_t size, StagingBufferType type, const char *name = nullptr) const;
  void copyBufferToBuffer(Buffer &dst, Buffer &src, const std::vector<vk::BufferCopy> &regions);
  void copyBufferToImage(Image &dst, Buffer &src, const std::vector<vk::BufferImageCopy> &regions);
  void copyImageToBuffer(Buffer &dst, Image &src, const std::vector<vk::BufferImageCopy> &regions);
  void copyImageToImage(Image &dst, Image &src, const std::vector<vk::ImageCopy> &regions);

  // @TODO: maybe not assert but return results if failed?
  void updateBuffer(Buffer &dst, vk::DeviceSize dstOffset, 
                    const std::byte *src, std::size_t size, 
                    Buffer *stagingBuff = nullptr, vk::DeviceSize stagingOffset = 0);
  void readBuffer(Buffer &src, vk::DeviceSize srcOffset, 
                  std::byte *dst, std::size_t size, 
                  Buffer *stagingBuff = nullptr, vk::DeviceSize stagingOffset = 0);
  void updateImage(Image &dst, const std::byte *src, std::size_t size, 
                   Buffer *stagingBuff = nullptr, vk::DeviceSize stagingOffset = 0);
  void readImage(Image &src, std::byte *dst, std::size_t size, 
                 Buffer *stagingBuff = nullptr, vk::DeviceSize stagingOffset = 0);

  static inline bool bufferNeedsStagingToUpdate(const Buffer &buff) { return buff.size > UPDATE_BUFFER_CMD_SIZE_LIMIT; }

  /* @TODO:
  * Add update/read functionality to Image class
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
  vk::BufferImageCopy defaultBufferImageCopy(const Image &img, vk::DeviceSize stagingOffset = 0);
};

}

#endif // ETNA_COPY_HELPER_HPP_INCLUDED
