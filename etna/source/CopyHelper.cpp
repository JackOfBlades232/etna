#include "etna/CopyHelper.hpp"
#include "etna/GlobalContext.hpp"

namespace etna
{

CopyHelper::CopyHelper(GlobalContext *ctx, vk::Queue queue)
  : context(ctx), transferQueue(queue)
{
  device  = context->getDevice();
  cmdPool = context->getCommandPool();
  // @TODO: assert?
  cmdBuff = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
      .commandPool        = cmdPool,
      .commandBufferCount = 1 
    }).value[0];
}

CopyHelper::~CopyHelper()
{
  device.freeCommandBuffers(cmdPool, {cmdBuff});
}

Buffer CopyHelper::createStagingBuffer(std::size_t size, const char *name)
{
  return context->createBuffer(
    {
      .size        = size,
      .bufferUsage = vk::BufferUsageFlagBits::eTransferSrc,
      .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
      .name        = name ? name : "tmp_staging_buffer"
    });
}

void CopyHelper::copyBufferToBuffer(Buffer &dst, Buffer &src, std::vector<vk::BufferCopy> regions)
{
  cmdBuff.reset();

  // @TODO: asserts?
  cmdBuff.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
  cmdBuff.copyBuffer(src.get(), dst.get(), regions);
  cmdBuff.end();

  // @TODO: assert?
  transferQueue.submit({ 
      vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmdBuff 
      }
    });
  // @TODO: assert?
  transferQueue.waitIdle();
}

}