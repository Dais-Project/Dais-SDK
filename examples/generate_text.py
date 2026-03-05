import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import OpenAIProvider
from dais_sdk.types.message import UserMessage
from dais_sdk.types.request_params import LlmRequestParams

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)
llm = LLM(provider)

response = llm.generate_text_sync(
    LlmRequestParams(
        model=MODEL,
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
