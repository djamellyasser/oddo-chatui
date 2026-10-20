from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionCall(BaseModel):
    name: str
    arguments: str | dict[str, Any] = "{}"


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def validate_tool_message(self):
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        return self


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ChatTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(None, ge=0)
    top_p: float | None = Field(None, gt=0, le=1)
    max_tokens: int | None = Field(None, gt=0)
    stop: str | list[str] | None = None
    tools: list[ChatTool] | None = None
    tool_choice: str | dict[str, Any] | None = None


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "ollama"
