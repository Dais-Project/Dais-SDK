# LiteAI-SDK

LiteAI-SDK is a wrapper of LiteLLM which provides a more intuitive API and [AI SDK](https://github.com/vercel/ai) like DX.

## Installation

```
pip install liteai-sdk
```

### Develop with coding agent

You can access the complete usage guidance with [llms.txt](https://raw.githubusercontent.com/BHznJNs/liteai/refs/heads/main/llms.txt), just give it to your coding agent to tell it how to use LiteAI-SDK.

## Examples

Below is a simple example of just a API call:

```python
import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams, UserMessage

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

response = llm.generate_text_sync( # sync API of generate_text
    LlmRequestParams(
        model="deepseek-v3.1",
        messages=[UserMessage(content="Hello.")]))
print(response)
```

Below is an example that shows the automatically tool call:

```python
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
        messages=[UserMessage(content="Please call the tool example_tool.")])

print("User: ", "Please call the tool example_tool.")
messages = llm.generate_text_sync(params)
for message in messages:
    match message.role:
        case "assistant":
            print("Assistant: ", message.content)
        case "tool":
            print("Tool: ", message.result)
```
