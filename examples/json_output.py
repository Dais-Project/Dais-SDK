import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dais_sdk import LLM
from dais_sdk.providers import OpenAIProvider
from dais_sdk.types import (
    LlmRequestParams, UserMessage,
)

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")


class Product(BaseModel):
    """商品信息"""
    name: str = Field(description="商品名称")
    price: float = Field(description="商品价格")
    tags: list[str] = Field(description="商品标签")


def main():
    provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)
    llm = LLM(provider)

    response = llm.generate_text_sync(
        LlmRequestParams(
            model="deepseek-v3.1",
            messages=[UserMessage(content="请生成一个商品信息，返回 JSON。")],
            output=Product,
        )
    )
    print("assistant raw output:\n", response.content)

if __name__ == "__main__":
    main()
