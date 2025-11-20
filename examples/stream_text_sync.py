import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

stream = llm.stream_text_sync(
    LlmRequestParams(
        model="deepseek-v3.1",
        messages=[{"role": "user", "content": "Hello World"}]))
for chunk in stream:
    if chunk.finish_reason is None:
        print(chunk.delta.content, end="", flush=True)
