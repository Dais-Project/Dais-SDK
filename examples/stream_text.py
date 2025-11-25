import asyncio
import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams, UserMessage

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

async def main():
    message_chunk, full_message_queue = await llm.stream_text(
        LlmRequestParams(
            model="deepseek-v3.1",
            messages=[UserMessage("Hello world")]))
    async for chunk in message_chunk:
        print(chunk.content, end="", flush=True)
    print()

    print("Full message: ", await full_message_queue.get())

asyncio.run(main())
