from liteai_sdk import LLM, LlmProviders

def test_llm_create():
    llm = LLM(
        LlmProviders.OPENAI,
        base_url="https://api.openai.com/v1",
        api_key="",
    )
    assert llm is not None
    assert llm.provider == LlmProviders.OPENAI
    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.api_key == ""
