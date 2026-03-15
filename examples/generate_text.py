import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.types import LlmRequestParams, UserMessage

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)
llm = LLM("deepseek-v3.1", provider)

response = llm.generate_text_sync(
    LlmRequestParams(
        messages=[UserMessage(content="你好，请用一句话介绍 Dais-SDK。")],
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
