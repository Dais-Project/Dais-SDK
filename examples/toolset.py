import os
from pathlib import Path

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.tool import ToolCallExecutor, PythonToolset, python_tool
from dais_sdk.types import LlmRequestParams, ChatMessage, ToolMessage, UserMessage


load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")
MAX_TURNS = 8

if not API_KEY:
    raise RuntimeError("API_KEY is required.")


class FileSystem(PythonToolset):
    def __init__(self, cwd: str | None = None):
        self._root = Path(cwd or os.getcwd()).resolve()

    def _resolve(self, path: str) -> Path:
        target = (self._root / path).resolve()
        target.relative_to(self._root)
        return target

    @python_tool
    def read_file(self, path: str) -> str:
        """Read the content of a UTF-8 text file under current working directory."""
        target = self._resolve(path)
        return target.read_text(encoding="utf-8")

    @python_tool
    def write_file(self, path: str, content: str) -> str:
        """Write UTF-8 content to a file under current working directory."""
        ...

    @python_tool
    def list_files(self, path: str = ".") -> list[str]:
        """List files and directories under a directory."""
        ...


provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)
llm = LLM("deepseek-v3.1", provider)
tool_call_executor = ToolCallExecutor()

messages: list[ChatMessage] = [
    UserMessage(
        content=(
            "请先调用 FileSystem 工具读取 README.md 的前几行，"
            "然后再用一句话总结这个项目是做什么的。"
        )
    )
]

for turn in range(1, MAX_TURNS + 1):
    print(f"\n=== turn {turn} ===")

    params = LlmRequestParams(
        messages=messages,
        toolsets=[FileSystem()],
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
