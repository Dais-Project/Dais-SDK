import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel

from dais_sdk import LLM, OneTurn
from dais_sdk.providers import LlmProviders


load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")


provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)
llm = LLM("deepseek-v3.1", provider)


class AuditResult(BaseModel):
    is_safe: bool
    reason: str

class CommandSafetyAudit(OneTurn[str, AuditResult]):
    def __init__(self, llm: LLM):
        super().__init__(
            llm,
            """\
你是一个终端命令安全审核员，用户会给你传入一条命令，
你需要判断这条命令是否安全。请严格按照 JSON Schema 要求返回 JSON。
除 JSON 外不要返回任何额外内容，不要用 Markdown 的代码块包裹 JSON 文本。
""",
            output=AuditResult,
            validate=True,
        )

safety_audit = CommandSafetyAudit(llm)
result = asyncio.run(safety_audit("rm -rf /"))
print(result)
