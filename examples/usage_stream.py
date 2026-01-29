import asyncio
import os
from dotenv import load_dotenv
from dais_sdk import (
    LLM, LlmProviders, LlmRequestParams,
    UserMessage, AssistantMessage,
    TextChunk, UsageChunk
)

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

async def main():
    print("=" * 60)
    print("流式请求示例 - Token 使用统计")
    print("=" * 60)

    # 流式请求
    message_chunk, full_message_queue = await llm.stream_text(
        LlmRequestParams(
            model="deepseek-v3.1",
            messages=[UserMessage(content="请用一句话介绍 Python 编程语言")]))

    print("\n实时响应内容：")
    print("-" * 60)

    # 处理流式响应块
    async for chunk in message_chunk:
        match chunk:
            case TextChunk(content=content):
                # 实时打印文本内容
                print(content, end="", flush=True)
            case UsageChunk(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t):
                # 流式响应结束时会收到 UsageChunk
                print("\n" + "-" * 60)
                print("\n📊 Token 使用统计 (来自流式响应):")
                print(f"  输入 tokens:   {input_t}")
                print(f"  输出 tokens:   {output_t}")
                print(f"  总计 tokens:   {total_t}")

    print("\n" + "=" * 60)

    # 获取完整消息（包含 usage 信息）
    full_message = await full_message_queue.get()
    if isinstance(full_message, AssistantMessage) and full_message.usage:
        print("\n📋 完整消息的 Token 统计：")
        print(f"  输入 tokens (prompt):      {full_message.usage.prompt_tokens}")
        print(f"  输出 tokens (completion):  {full_message.usage.completion_tokens}")
        print(f"  总计 tokens:               {full_message.usage.total_tokens}")
        print("=" * 60)

asyncio.run(main())
