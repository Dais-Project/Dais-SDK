import json
from typing import Literal, Never, overload
from pydantic import BaseModel
from .llm import LLM
from ..types import LlmRequestParams, UserMessage


class OneTurn[Input=str, Output: BaseModel=Never]:
    def __init__(self,
                 llm: LLM,
                 instruction: str,
                 output: Literal["text"] | type[Output] = "text",
                 validate: bool = False,
                 ):
        self._llm = llm
        self._instruction = instruction
        self._output = output
        self._validate = validate
        if validate and output == "text":
            raise ValueError("validate=True requires output to be a pydantic model")

    def _create_request(self, input: Input) -> LlmRequestParams:
        formated_input = self.format_input(input)
        return LlmRequestParams(
            messages=[UserMessage(content=formated_input)],
            instructions=self._instruction,
            output=self._output,
        )

    def format_input(self, input: Input) -> str:
        return str(input)

    @overload
    async def __call__(self, input: Input) -> str: ...
    @overload
    async def __call__(self, input: Input) -> Output: ...
    async def __call__(self, input: Input) -> str | Output:
        request = self._create_request(input)
        response = await self._llm.generate_text(request)
        if response.content is None:
            raise ValueError("Expected JSON output but got None")

        if self._output == "text":
            return response.content

        print(response.content)

        if self._validate:
            return self._output.model_validate_json(response.content)
        else:
            data = json.loads(response.content)
            return self._output.model_construct(**data)
