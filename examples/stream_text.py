import asyncio
import os
from dotenv import load_dotenv
from dais_sdk import LLM, LlmProviders, LlmRequestParams, UserMessage
from dais_sdk.types.event import AssistantMessageEvent, TextChunkEvent

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

async def main():
    async for chunk in llm.stream_text(
        LlmRequestParams(
            model="deepseek-v3.1",
            messages=[UserMessage(content="Hello world")])):
        match chunk:
            case TextChunkEvent(content=content):
                print("[Message chunk]: ", content)
            case AssistantMessageEvent(message=message):
                print("[Full message]: ", message)

asyncio.run(main())
