import os
from dotenv import load_dotenv
from dais_sdk import (LLM, LlmProviders, LlmRequestParams,
                        UserMessage, SystemMessage, AssistantMessage, ToolMessage)

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
    
    test_prompt1 = r"Please read the python script `D:\MyPrograms\python_programs\temp.py` and tell me what it does."
    messages = [SystemMessage(content=SYSTEM_PROMPT), UserMessage(content=test_prompt1)]

    tools = [read_file, attempt_completion]

    is_running = True

    print(f"User: {test_prompt1}")
    
    loop_count = 0

    while is_running:
        loop_count += 1
        print(f"Loop count: {loop_count}")

        params = LlmRequestParams(
            model="deepseek-v3.1",
            messages=messages,
            tools=tools,
            execute_tools=True
        )

        responses = llm.generate_text_sync(params)
        messages += responses

        for message in responses:
            if isinstance(message, AssistantMessage):
                print(f"Model: {message.content}")
            elif isinstance(message, ToolMessage):
                print(f"Tool: {message.result}")

if __name__ == "__main__":
    agent_loop()
    print("Agent loop terminated.")
