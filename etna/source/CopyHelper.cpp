#include "etna/CopyHelper.hpp"
#include "etna/GlobalContext.hpp"

namespace etna
{

CopyHelper::CopyHelper(GlobalContext *ctx, vk::Queue queue)
  : context(ctx), transferQueue(queue)
{
  cmdPool = context->getCommandPool();
  cmdBuff = context->createCommandBuffer();
}

CopyHelper::~CopyHelper()
{
  context->freeCommandBuffer(cmdBuff);
}

Buffer CopyHelper::createStagingBuffer(std::size_t size, StagingBufferType type, const char *name) const
{
  vk::BufferUsageFlags usageFlags;
  if (type & eCpuToGpu)
    usageFlags |= vk::BufferUsageFlagBits::eTransferSrc;
  if (type & eGpuToCpu)
    usageFlags |= vk::BufferUsageFlagBits::eTransferDst;
  return context->createBuffer(
    {
      .size        = size,
      .bufferUsage = usageFlags,
      .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
      .name        = name ? name : "tmp_staging_buffer"
    });
}

void CopyHelper::copyBufferToBuffer(Buffer &dst, Buffer &src, const std::vector<vk::BufferCopy> &regions)
{
  executeCommands([&](vk::CommandBuffer &cbuff){ cbuff.copyBuffer(src.get(), dst.get(), regions); });
}

void CopyHelper::updateBuffer(Buffer &dst, vk::DeviceSize dstOffset, 
                              const std::byte *src, std::size_t size, 
                              Buffer *stagingBuff, vk::DeviceSize stagingOffset)
{
  // @TODO: is this needed?
  ETNA_ASSERT(dstOffset % 4 == 0);
  ETNA_ASSERT(stagingOffset % 4 == 0);
  ETNA_ASSERT(size % 4 == 0);
  ETNA_ASSERT(size <= dst.size - dstOffset);

  if (stagingBuff)
  {
    ETNA_ASSERT(size <= stagingBuff->size - stagingOffset);
    if (!stagingBuff->data())
      stagingBuff->map();

    memcpy(stagingBuff->data() + (std::size_t)stagingOffset, src, size);
    copyBufferToBuffer(dst, *stagingBuff, {{stagingOffset, dstOffset, size}});
  }
  else if (size <= UPDATE_BUFFER_CMD_SIZE_LIMIT)
    executeCommands([&](vk::CommandBuffer &cbuff) { cbuff.updateBuffer(dst.get(), dstOffset, size, src); });
  else
  {
    Buffer tmpStagingBuff = createStagingBuffer(size, eCpuToGpu, "tmp_staging_buffer");
    memcpy(tmpStagingBuff.map(), src, size);
    copyBufferToBuffer(dst, tmpStagingBuff, {{0, dstOffset, size}});
  }
}

void CopyHelper::readBuffer(Buffer &src, vk::DeviceSize srcOffset,
                            std::byte *dst, std::size_t size,
                            Buffer *stagingBuff, vk::DeviceSize stagingOffset)
{
  // @TODO: is this needed?
  ETNA_ASSERT(srcOffset % 4 == 0);
  ETNA_ASSERT(stagingOffset % 4 == 0);
  ETNA_ASSERT(size % 4 == 0);
  ETNA_ASSERT(size <= src.size - srcOffset);

  if (stagingBuff)
  {
    ETNA_ASSERT(size <= stagingBuff->size - stagingOffset);
    if (!stagingBuff->data())
      stagingBuff->map();

    copyBufferToBuffer(*stagingBuff, src, {{srcOffset, stagingOffset, size}});
    memcpy(dst, stagingBuff->data() + (std::size_t)stagingOffset, size);
  }
  else
  {
    Buffer tmpStagingBuff = createStagingBuffer(size, eGpuToCpu, "tmp_staging_buffer");
    copyBufferToBuffer(tmpStagingBuff, src, {{srcOffset, 0, size}});
    memcpy(dst, tmpStagingBuff.map(), size);
  }
}

void CopyHelper::executeCommands(const std::function<void(vk::CommandBuffer &)> &cmds)
{
  cmdBuff.reset();

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