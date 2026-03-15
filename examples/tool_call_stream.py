import asyncio
import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.tool import ToolCallExecutor
from dais_sdk.types import (
    LlmRequestParams, ToolLike,
    AssistantMessageEvent, TextChunkEvent, ToolCallChunkEvent, UsageChunkEvent,
    BaseMessage, AssistantMessage, ToolMessage, UserMessage
)


load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")
MAX_TURNS = 8

if not API_KEY:
    raise RuntimeError("API_KEY is required.")


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city} 当前天气是晴天，25°C。"


def get_time(city: str) -> str:
    """Get local time for a city."""
    return f"{city} 当前本地时间是 10:00。"


provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)
llm = LLM("deepseek-v3.1", provider)


async def stream_one_turn(params: LlmRequestParams) -> AssistantMessage:
    final_message: AssistantMessage | None = None

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
                final_message = message

    if final_message is None:
        raise RuntimeError("No AssistantMessageEvent received from stream.")

    print("\n[assistant final]", final_message)
    return final_message


async def main() -> None:
    messages: list[BaseMessage] = [
        UserMessage(content="请先调用工具查询北京天气和时间，再给我一段简短行程建议。"),
    ]
    tools: list[ToolLike] = [get_weather, get_time]
    tool_call_executor = ToolCallExecutor()

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n=== turn {turn} ===")
        params = LlmRequestParams(
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        assistant = await stream_one_turn(params)
        messages.append(assistant)

        if not assistant.tool_calls:
            print("[done] no tool calls, conversation finished.")
            break

        for tool_call in assistant.tool_calls:
            tool = params.find_tool(tool_call.name)
            if tool is None:
                result, error = None, f"Tool not found: {tool_call.name}"
            else:
                result, error = await tool_call_executor.execute(tool, tool_call.arguments)

            tool_message = ToolMessage(
                call_id=tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=result,
                error=error,
            )
            messages.append(tool_message)

            status = "error" if error else "result"
            payload = error if error else result
            print(f"[tool:{tool_call.name}] {status}={payload}")
    else:
        print("[stopped] reached MAX_TURNS without completion.")


if __name__ == "__main__":
    asyncio.run(main())
