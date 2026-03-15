import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dais_sdk import LLM
from dais_sdk.providers import LlmProviders
from dais_sdk.types import LlmRequestParams, UserMessage


load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)

class Product(BaseModel):
    """商品信息"""
    name: str = Field(description="商品名称")
    price: float = Field(description="商品价格")
    tags: list[str] = Field(description="商品标签")

async def main():
    llm = LLM("deepseek-v3.1", provider)

    response = await llm.generate_text(
        LlmRequestParams(
            messages=[UserMessage(content="请生成一个商品信息，返回 JSON。")],
            output=Product,
        )
    )
    print("assistant raw output:\n", response.content)


if __name__ == "__main__":
    asyncio.run(main())
