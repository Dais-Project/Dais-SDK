import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import OpenAIProvider
from dais_sdk.tool import ToolCallExecutor
from dais_sdk.types import (
    LlmRequestParams,
    ToolLike,
    ChatMessage, ToolMessage, UserMessage
)

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")
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


provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)
llm = LLM(provider)

messages: list[ChatMessage] = [
    UserMessage(content="请先调用工具查询北京天气和时间，再给我一段简短行程建议。"),
]
tools: list[ToolLike] = [get_weather, get_time]
tool_call_executor = ToolCallExecutor()

for turn in range(1, MAX_TURNS + 1):
    print(f"\n=== turn {turn} ===")

    params = LlmRequestParams(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant = llm.generate_text_sync(params)
    messages.append(assistant)

    print("[assistant]", assistant.content)

    if not assistant.tool_calls:
        print("[done] no tool calls, conversation finished.")
        break

    for tool_call in assistant.tool_calls:
        tool = params.find_tool(tool_call.name)

        if tool is None:
            result, error = None, f"Tool not found: {tool_call.name}"
        else:
            result, error = tool_call_executor.execute_sync(tool, tool_call.arguments)

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
