#include "etna/CopyHelper.hpp"
#include "etna/GlobalContext.hpp"

namespace etna
{

CopyHelper::CopyHelper(GlobalContext *ctx, vk::Queue queue)
  : context(ctx), transferQueue(queue)
{
  device  = context->getDevice();
  cmdPool = context->getCommandPool();
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

void CopyHelper::copyBufferToBuffer(Buffer &dst, Buffer &src, const std::vector<vk::BufferCopy> &regions)
{
  executeCommands([&](vk::CommandBuffer &buff){ cmdBuff.copyBuffer(dst.get(), src.get(), regions); });
}

// @TODO: assert messages
void CopyHelper::updateBuffer(Buffer &dst, vk::DeviceSize dstOffset,
                              const std::byte *src, std::size_t size, 
                              Buffer *stagingBuff)
{
  ETNA_ASSERT(dstOffset % 4 == 0);
  ETNA_ASSERT(size % 4 == 0);
  ETNA_ASSERT(size <= dst.size + dstOffset);

  if (size <= SMALL_BUFF_SIZE)
    executeCommands([&](vk::CommandBuffer &buff){ cmdBuff.updateBuffer(dst.get(), dstOffset, size, src); });
  else if (stagingBuff)
  {
    ETNA_ASSERT(size <= stagingBuff->size);
    if (!stagingBuff->data())
      stagingBuff->map();

    memcpy(stagingBuff->data(), src, size);
    copyBufferToBuffer(dst, *stagingBuff, {{0, dstOffset, size}});
  }
  else
  {
    Buffer tmpStagingBuff = createStagingBuffer(size, "tmp_staging_buffer");
    memcpy(tmpStagingBuff.map(), src, size);
    copyBufferToBuffer(dst, tmpStagingBuff, {{0, dstOffset, size}});
  }
}

void CopyHelper::executeCommands(const std::function<void(vk::CommandBuffer &)> &cmds)
{
  cmdBuff.reset();

  // @TODO: assert messages
  ETNA_VK_ASSERT(cmdBuff.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit }));
  cmds(cmdBuff);
  ETNA_VK_ASSERT(cmdBuff.end());

  ETNA_VK_ASSERT(transferQueue.submit({ 
      vk::SubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmdBuff 
      }
    }));
  ETNA_VK_ASSERT(transferQueue.waitIdle());
}

}