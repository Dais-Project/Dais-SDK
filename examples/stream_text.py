import asyncio
import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams, UserMessage

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

async def main():
    stream = llm.stream_text(
        LlmRequestParams(
            model="deepseek-v3.1",
            messages=[UserMessage("Hello world")]))
    async for chunk in stream:
        print(chunk)
        # if chunk.finish_reason is None:
        #     print(chunk.delta.content, end="", flush=True)

asyncio.run(main())
