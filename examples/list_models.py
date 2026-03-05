import os

from dotenv import load_dotenv

from dais_sdk import LLM
from dais_sdk.providers import OpenAIProvider

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is required.")

provider = OpenAIProvider(base_url=BASE_URL, api_key=API_KEY)
llm = LLM(provider)

for model in llm.list_models_sync():
    print(model)
