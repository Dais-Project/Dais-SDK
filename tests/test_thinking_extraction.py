import pytest
from dais_sdk.types.message import AssistantMessage

def test_extract_thinking_content_basic():
    text = "<think>thinking part</think>actual content"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "actual content"
    assert thinking == "thinking part"

def test_extract_thinking_content_thinking_tag():
    text = "<thinking>thinking part</thinking>actual content"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "actual content"
    assert thinking == "thinking part"

def test_extract_thinking_content_no_tag():
    text = "just actual content"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "just actual content"
    assert thinking is None

def test_extract_thinking_content_unclosed():
    text = "<think>thinking part actual content"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "<think>thinking part actual content"
    assert thinking is None

def test_extract_thinking_content_multiple():
    text = "<think>first</think>middle<think>second</think>end"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "middle<think>second</think>end"
    assert thinking == "first"

def test_extract_thinking_content_multiline():
    text = "<think>\nline1\nline2\n</think>content"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "content"
    assert thinking == "\nline1\nline2\n"

def test_extract_thinking_content_mismatched():
    text = "<think>content</thinking>"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "<think>content</thinking>"
    assert thinking is None

def test_extract_thinking_content_in_middle():
    text = "some content <think>thinking part</think> more content"
    remaining, thinking = AssistantMessage.extract_thinking_content(text)
    assert remaining == "some content <think>thinking part</think> more content"
    assert thinking is None
