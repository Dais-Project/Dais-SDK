import asyncio
import os
from dotenv import load_dotenv
from dais_sdk import LLM, OpenAIProvider, LlmRequestParams, UserMessage
from dais_sdk.types.event import TextChunkEvent, AssistantMessageEvent, ToolCallChunkEvent

load_dotenv()

def get_weather(city: str):
    """
    Get the weather in a city.
    """
    return f"The weather in {city} is sunny."

provider = OpenAIProvider(
    base_url=os.getenv("BASE_URL", ""),
    api_key=os.getenv("API_KEY", ""))
llm = LLM(provider=provider)

params = LlmRequestParams(
        model="gpt-5.3-codex",
        tools=[get_weather],
        messages=[UserMessage(content="Please tell me the weather in Beijing.")])

async def main():
    async for chunk in llm.stream_text(params):
        match chunk:
            case TextChunkEvent(content=content):
                print(content, flush=True, end="")
            case ToolCallChunkEvent(id=id, name=name, arguments=arguments, index=index):
                print(f"\nTool call chunk: {id} {name} {arguments} {index}")
            case AssistantMessageEvent(message=message):
                print(f"\nFull message: {message}")

if __name__ == "__main__":
    asyncio.run(main())
