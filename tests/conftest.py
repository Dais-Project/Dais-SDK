import dataclasses
from typing import Any, Optional
import pytest

@pytest.fixture
def sample_functions():
    def simple_func(x: int) -> int:
        """Simple function"""
        return x

    def complex_func(a: str, b: list[int], c: Optional[dict[str, Any]] = None) -> dict:
        """Complex function"""
        return {}

    return {"simple": simple_func, "complex": complex_func}


@pytest.fixture
def sample_types():
    from typing import TypedDict

    class UserDict(TypedDict):
        name: str
        age: int

    @dataclasses.dataclass
    class UserData:
        username: str
        email: str = "default@example.com"

    return {"typed_dict": UserDict, "dataclass": UserData}