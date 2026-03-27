from typing import Literal
from dataclasses import dataclass
from binaryornot.helpers import is_binary_string

@dataclass
class BaseResource:
    relative: str

@dataclass
class TextResource(BaseResource):
    content: str
    type: Literal["text"] = "text"

@dataclass
class BinaryResource(BaseResource):
    content: bytes
    type: Literal["binary"] = "binary"

type SkillResource = TextResource | BinaryResource

def create_from_bytes(relative: str, content: bytes) -> SkillResource:
    is_binary = is_binary_string(content)
    if is_binary:
        return BinaryResource(relative=relative, content=content)
    else:
        return TextResource(relative=relative, content=content.decode("utf-8-sig", errors="replace"))

__all__ = [
    "BaseResource",
    "TextResource",
    "BinaryResource",
    "SkillResource",
]
