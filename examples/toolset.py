import os
from pathlib import Path

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import OpenAIProvider
from dais_sdk.tool.toolset import PythonToolset, python_tool
from dais_sdk.types.message import ChatMessage, ToolMessage, UserMessage
from dais_sdk.types.request_params import LlmRequestParams

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")
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
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"written: {target}"

    @python_tool
    def list_files(self, path: str = ".") -> list[str]:
        """List files and directories under a directory."""
        target = self._resolve(path)
        return sorted(item.name for item in target.iterdir())


provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)
llm = LLM(provider)

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
        model=MODEL,
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
            result, error = llm.execute_tool_call_sync(tool, tool_call.arguments)

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
