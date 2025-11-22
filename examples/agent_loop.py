import os
from dotenv import load_dotenv
from liteai_sdk import LLM, LlmProviders, LlmRequestParams,\
                       ChatMessage, UserMessage, SystemMessage, AssistantMessage, ToolMessage

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant with access to a set of tools. You should call the attempt_completion tool when you have completed your task."

def read_file(file_path: str) -> str:
    """
    Read the content of a file
    
    Args:
        file_path
        
    Returns:
        File content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file
    
    Args:
        file_path
        content
        
    Returns:
        Success message or error message
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"成功写入文件：{file_path}"
    except Exception as e:
        return f"写入文件时出错：{str(e)}"


def agent_loop():
    def attempt_completion():
        """
        When the you have completed your task, you should call this tool to terminate it.
        """
        nonlocal is_running
        is_running = False
        return "You has completed the task."

    llm = LLM(provider=LlmProviders.OPENAI,
              api_key=os.getenv("API_KEY", ""),
              base_url=os.getenv("BASE_URL", ""))
    
    test_prompt1 = r"Please read the file `D:\MyPrograms\python_programs\temp.py` and tell me what it does."
    messages: list[ChatMessage] = [SystemMessage(SYSTEM_PROMPT), UserMessage(test_prompt1)]

    tools = [read_file, write_file, attempt_completion]

    is_running = True
    
    print(f"User: {test_prompt1}")
    
    while is_running:
        params = LlmRequestParams(
            model="deepseek-v3.1",
            messages=messages,
            tools=tools,
            execute_tools=True
        )

        responses = llm.generate_text_sync(params)
        messages += responses

        for message in messages:
            if isinstance(message, AssistantMessage):
                print(f"Model: {message.content}")
            elif isinstance(message, ToolMessage):
                print(f"Tool: {message.content}")

if __name__ == "__main__":
    agent_loop()
    print("Agent loop terminated.")
