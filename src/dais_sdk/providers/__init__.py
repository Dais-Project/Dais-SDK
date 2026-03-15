from enum import Enum
from .base_provider import BaseProvider

class LlmProviders(str, Enum):
    OPENAI = "openai"

__all__ = [
    "LlmProviders",
    "BaseProvider",
]
