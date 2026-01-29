import os
from dotenv import load_dotenv
from dais_sdk import LLM, LlmProviders

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

print(llm.list_models())
