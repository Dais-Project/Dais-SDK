from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from dais_sdk.providers.utils import StrictInlineJsonSchema


def _contains_key(node: Any, key: str) -> bool:
    if isinstance(node, dict):
        if key in node:
            return True
        return any(_contains_key(value, key) for value in node.values())
    if isinstance(node, list):
        return any(_contains_key(value, key) for value in node)
    return False


def test_model_schema_sets_additional_properties_false() -> None:
    class SimpleModel(BaseModel):
        name: str

    schema = SimpleModel.model_json_schema(schema_generator=StrictInlineJsonSchema)

    assert schema.get("additionalProperties") is False


def test_generate_inlines_refs_and_removes_defs() -> None:
    class Child(BaseModel):
        value: int

    class Parent(BaseModel):
        child: Child

    schema = Parent.model_json_schema(schema_generator=StrictInlineJsonSchema)

    assert _contains_key(schema, "$ref") is False
    assert _contains_key(schema, "$defs") is False
    assert schema["properties"]["child"]["properties"]["value"]["type"] == "integer"
