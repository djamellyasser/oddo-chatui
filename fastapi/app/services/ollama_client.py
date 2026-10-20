import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from ..config import Settings
from ..schemas import ChatCompletionRequest


class OllamaError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class OllamaClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        timeout = httpx.Timeout(settings.ollama_read_timeout, connect=settings.ollama_connect_timeout)
        self.client = httpx.AsyncClient(base_url=str(settings.ollama_url).rstrip("/"), timeout=timeout, transport=transport)

    async def close(self) -> None:
        await self.client.aclose()

    async def models(self) -> list[dict[str, Any]]:
        payload = await self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        for entry in models:
            name = entry.get("name")
            if not name:
                continue
            try:
                details = await self._request_json("POST", "/api/show", json={"model": name})
            except OllamaError:
                continue
            lengths = [
                value for key, value in (details.get("model_info") or {}).items()
                if key.endswith(".context_length") and isinstance(value, int)
            ]
            if lengths:
                entry["context_window"] = max(lengths)
        return models

    @staticmethod
    def _content(content: Any) -> Any:
        if not isinstance(content, list):
            return content
        texts = [part.get("text", "") for part in content if part.get("type") in {"text", "input_text", "output_text"}]
        return "\n".join(filter(None, texts))

    def build_payload(self, request: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
        messages = []
        for entry in request.messages:
            message: dict[str, Any] = {"role": entry.role, "content": self._content(entry.content) or ""}
            if entry.tool_call_id:
                message["tool_call_id"] = entry.tool_call_id
            if entry.tool_calls:
                message["tool_calls"] = [call.model_dump() for call in entry.tool_calls]
            messages.append(message)
        options = {key: value for key, value in {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens,
            "stop": [request.stop] if isinstance(request.stop, str) else request.stop,
        }.items() if value is not None}
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if options:
            body["options"] = options
        if request.tools:
            body["tools"] = [tool.model_dump(exclude_none=True) for tool in request.tools]
        return body

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json("POST", "/api/chat", json=body)

    @asynccontextmanager
    async def stream_chat(self, body: dict[str, Any]):
        try:
            async with self.client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    self._raise_status(response.status_code, raw.decode(errors="replace"))

                async def events() -> AsyncIterator[dict[str, Any]]:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise OllamaError("Ollama returned an invalid streaming event") from exc
                yield events()
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama request timed out", 504) from exc
        except httpx.ConnectError as exc:
            raise OllamaError("Ollama is unavailable", 503) from exc
        except httpx.RequestError as exc:
            raise OllamaError("Ollama transport failure", 502) from exc

    async def _request_json(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = await self.client.request(method, path, **kwargs)
            if response.status_code >= 400:
                self._raise_status(response.status_code, response.text)
            return response.json()
        except OllamaError:
            raise
        except httpx.TimeoutException as exc:
            raise OllamaError("Ollama request timed out", 504) from exc
        except httpx.ConnectError as exc:
            raise OllamaError("Ollama is unavailable", 503) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise OllamaError("Invalid response from Ollama", 502) from exc

    @staticmethod
    def _raise_status(status_code: int, body: str) -> None:
        try:
            message = json.loads(body).get("error") or body
        except ValueError:
            message = body
        code = 404 if status_code == 404 or "not found" in message.lower() else 502
        raise OllamaError(f"Ollama error: {message[:500]}", code)
