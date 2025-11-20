import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, GenerateTextRequest

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

response = llm.generate_text_sync(
    GenerateTextRequest(
        model="deepseek-v3.1",
        messages=[{"role": "user", "content": "Hello World"}]))
print(response)
