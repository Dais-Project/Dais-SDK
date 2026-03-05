import asyncio
import os
from dotenv import load_dotenv
from dais_sdk import LLM, LlmRequestParams, UserMessage
from dais_sdk.providers import OpenAIProvider
from dais_sdk.types.event import TextChunkEvent, UsageChunkEvent, AssistantMessageEvent

load_dotenv()

provider = OpenAIProvider(
    base_url=os.getenv("BASE_URL", ""),
    api_key=os.getenv("API_KEY", ""))
llm = LLM(provider=provider)

async def main():
    async for chunk in llm.stream_text(
        LlmRequestParams(
            model="deepseek-v3.1",
            messages=[UserMessage(content="请用一句话介绍 Python 编程语言")])):
        match chunk:
            case TextChunkEvent(content=content):
                print(content, end="", flush=True)
            case AssistantMessageEvent(message=message):
                print(f"\nFull message: {message}")
            case UsageChunkEvent(input_tokens=input_t, output_tokens=output_t, total_tokens=total_t):
                print("\n" + "-" * 60)
                print("\n📊 Token 使用统计 (来自流式响应):")
                print(f"  输入 tokens:   {input_t}")
                print(f"  输出 tokens:   {output_t}")
                print(f"  总计 tokens:   {total_t}")

asyncio.run(main())
