from dais_sdk.providers.openai import OpenAIProvider
from dais_sdk.types.message import AssistantMessage


def _message(content: str | None, reasoning_content: str | None = None) -> AssistantMessage:
    message = AssistantMessage()
    message.content = content
    message.reasoning_content = reasoning_content
    return message


def test_extract_thinking_content_with_think_tag_at_start() -> None:
    message = _message("<think>thinking part</think>actual content")

    result = OpenAIProvider._extract_thinking_content(message)

    assert result.content == "actual content"
    assert result.reasoning_content == "thinking part"


def test_extract_thinking_content_with_thinking_tag_at_start() -> None:
    message = _message("<thinking>thinking part</thinking>actual content")

    result = OpenAIProvider._extract_thinking_content(message)

    assert result.content == "actual content"
    assert result.reasoning_content == "thinking part"


def test_extract_thinking_content_without_tag() -> None:
    message = _message("just actual content")

    result = OpenAIProvider._extract_thinking_content(message)

    assert result.content == "just actual content"
    assert result.reasoning_content is None


def test_extract_thinking_content_with_unclosed_tag() -> None:
    message = _message("<think>thinking part actual content")

    result = OpenAIProvider._extract_thinking_content(message)

    assert result.content == "<think>thinking part actual content"
    assert result.reasoning_content is None


def test_extract_thinking_content_with_tag_not_at_start() -> None:
    message = _message("some content <think>thinking part</think> more content")

    result = OpenAIProvider._extract_thinking_content(message)

    assert result.content == "some content <think>thinking part</think> more content"
    assert result.reasoning_content is None


def test_extract_thinking_content_keeps_existing_reasoning_content() -> None:
    message = _message(
        "<think>new reasoning</think>actual content",
        reasoning_content="already exists",
    )

    result = OpenAIProvider._extract_thinking_content(message)

    assert result.content == "<think>new reasoning</think>actual content"
    assert result.reasoning_content == "already exists"
