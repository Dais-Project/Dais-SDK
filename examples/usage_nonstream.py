import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams, UserMessage, AssistantMessage

load_dotenv()

llm = LLM(provider=LlmProviders.OPENAI,
          api_key=os.getenv("API_KEY", ""),
          base_url=os.getenv("BASE_URL", ""))

# 非流式请求示例
response = llm.generate_text_sync(
    LlmRequestParams(
        model="deepseek-v3.1",
        messages=[UserMessage(content="请用一句话介绍 Python 编程语言")]))

# 获取响应内容和 token 使用情况
for message in response:
    if isinstance(message, AssistantMessage):
        print("=" * 60)
        print("响应内容：")
        print(message.content)
        print("=" * 60)
        
        # 检查并显示 token 使用情况
        if message.usage:
            print("\nToken 使用统计：")
            print(f"  输入 tokens (prompt):      {message.usage.prompt_tokens}")
            print(f"  输出 tokens (completion):  {message.usage.completion_tokens}")
            print(f"  总计 tokens:               {message.usage.total_tokens}")
            print("=" * 60)
        else:
            print("\n⚠️  此响应不包含 token 使用信息")