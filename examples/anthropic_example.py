import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.types import (
    LlmRequestParams, UserMessage,
    AssistantMessageEvent, TextChunkEvent, ToolCallChunkEvent, UsageChunkEvent
)

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.anthropic.com")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

provider = LLM.create_provider(LlmProviders.ANTHROPIC, BASE_URL, API_KEY)

def generate_text_example():
    llm = LLM("claude-4.5-haiku", provider)

    response = llm.generate_text_sync(
        LlmRequestParams(
            messages=[UserMessage(content="你好，请用一句话介绍你自己。")],
        )
    )

    print("[assistant]", response.content)
    if response.reasoning_content:
        print("\n[reasoning]", response.reasoning_content)
    if response.usage:
        print(
            "\n[usage]",
            f"input={response.usage.input_tokens}",
            f"output={response.usage.output_tokens}",
            f"total={response.usage.total_tokens}",
        )

def stream_text_example():
    llm = LLM("claude-4.5-haiku", provider)

    params = LlmRequestParams(
        messages=[UserMessage(content="请用两句话解释什么是流式输出。")],
    )

    print("[assistant streaming] ", end="", flush=True)

    for event in llm.stream_text_sync(params):
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

def stream_tool_call_example():
    def get_weather(city: str) -> str:
        """获取城市天气"""
        return f"{city} 当前天气是晴天，25°C。"

    llm = LLM("claude-4.5-haiku", provider)

    params = LlmRequestParams(
        messages=[UserMessage(content="请调用 get_weather 工具查询北京和上海的天气。")],
        tool_choice="auto",
        tools=[get_weather],
    )

    print("[assistant streaming] ", end="", flush=True)
    for event in llm.stream_text_sync(params):
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
    generate_text_example()
    stream_text_example()
    stream_tool_call_example()