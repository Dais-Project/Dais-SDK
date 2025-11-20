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
        messages=[UserMessage("Please call the tool example_tool.")])

print("User: ", "Please call the tool example_tool.")
messages = llm.generate_text_sync(params)
for message in messages:
    match message.role:
        case "assistant":
            print("Assistant: ", message.content)
        case "tool":
            print("Tool: ", message.content)
