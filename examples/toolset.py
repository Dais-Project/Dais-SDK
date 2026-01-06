import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams, UserMessage, Toolset, tool

load_dotenv()

class FileSystem(Toolset):
    def __init__(self, cwd: str | None = None):
        self._cwd = cwd or os.getcwd()

    @tool
    def read_file(self, path: str) -> str: ...

    @tool
    def write_file(self, path: str, content: str) -> None: ...

    @tool
    def list_files(self, path: str) -> list[str]: ...


llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

response = llm.generate_text_sync(
    LlmRequestParams(
        model="deepseek-v3.1",
        messages=[UserMessage(content="Hello")],
        toolsets=[FileSystem()]))
