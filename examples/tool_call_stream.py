import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams, UserMessage

load_dotenv()

def example_tool():
    """
    This is a test tool that is used to test the tool calling functionality.
    """
    print("The example tool is called.")
    return "Hello World"

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

params = LlmRequestParams(
        model="deepseek-v3.1",
        tools=[example_tool],
        execute_tools=True,
        messages=[UserMessage("Please call the tool example_tool twice.")])

print("User: ", "Please call the tool example_tool.")
stream, full_responses = llm.stream_text_sync(params)
print("Model: ", end="")
for chunk in stream:
    if chunk.content is not None:
        print(chunk.content, flush=True, end="")
print("\n")

while (response := full_responses.get()) is not None:
    print("Complete response: ", response)
print("End of stream.")
