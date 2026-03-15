import asyncio
import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.types.event import (
    AssistantMessageEvent,
    TextChunkEvent,
    ToolCallChunkEvent,
    UsageChunkEvent,
)
from dais_sdk.types.message import UserMessage
from dais_sdk.types.request_params import LlmRequestParams

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)
llm = LLM("deepseek-v3.1", provider)


async def main() -> None:
    params = LlmRequestParams(
        messages=[UserMessage(content="请用两句话解释什么是流式输出。")],
    )

    print("[assistant streaming] ", end="", flush=True)

    async for event in llm.stream_text(params):
        match event:
            case TextChunkEvent(content=content):
                print(content, end="", flush=True)
            case ToolCallChunkEvent(id=tool_id, name=name, arguments=arguments, index=index):
                print(
                    f"\n[tool-call-chunk] id={tool_id} name={name} index={index} arguments={arguments}",
                    flush=True,
                )
            case UsageChunkEvent(input_tokens=inp, output_tokens=out, total_tokens=total):
                print(f"\n[usage] input={inp} output={out} total={total}", flush=True)
            case AssistantMessageEvent(message=message):
                print("\n[assistant final]", message)


if __name__ == "__main__":
    asyncio.run(main())
