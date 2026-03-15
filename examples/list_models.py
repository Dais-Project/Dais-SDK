import asyncio
import os

from dotenv import load_dotenv
from dais_sdk import LLM
from dais_sdk.providers import LlmProviders

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

provider = LLM.create_provider(LlmProviders.OPENAI, BASE_URL, API_KEY)
models = asyncio.run(provider.list_models())
print(models)
