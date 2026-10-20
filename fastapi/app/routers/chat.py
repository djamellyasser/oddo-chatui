
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..config import Settings, get_settings
from ..schemas import ChatCompletionRequest
from ..security import authenticate
from ..services.ollama_client import OllamaError

router = APIRouter()


def _tool_calls(raw, completion_id: str):
    result = []
    for index, call in enumerate(raw or []):
        function = call.get("function") or {}
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, separators=(",", ":"))
        result.append({"index": index, "id": call.get("id") or f"call_{completion_id[-12:]}_{index}", "type": "function", "function": {"name": function.get("name", ""), "arguments": arguments}})
    return result


@router.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
async def completions(payload: ChatCompletionRequest, request: Request, settings: Settings = Depends(get_settings)):
    model = payload.model or settings.ollama_default_model
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    body = request.app.state.ollama.build_payload(payload, model, payload.stream)
    if payload.stream:
        return StreamingResponse(_stream(request.app.state.ollama, body, model, completion_id, created), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    result = await request.app.state.ollama.chat(body)
    message = result.get("message") or {}
    answer = {"role": "assistant", "content": message.get("content") or ""}
    calls = _tool_calls(message.get("tool_calls"), completion_id)
    if calls:
        answer["tool_calls"] = [{k: v for k, v in call.items() if k != "index"} for call in calls]
    prompt_tokens = result.get("prompt_eval_count", 0)
    completion_tokens = result.get("eval_count", 0)
    return {"id": completion_id, "object": "chat.completion", "created": created, "model": result.get("model") or model, "choices": [{"index": 0, "message": answer, "finish_reason": "tool_calls" if calls else (result.get("done_reason") or "stop")}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}}


async def _stream(client, body, model: str, completion_id: str, created: int) -> AsyncIterator[str]:
    base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model}
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    try:
        async with client.stream_chat(body) as events:
            async for event in events:
                message = event.get("message") or {}
                if content := message.get("content"):
                    yield _sse({**base, "model": event.get("model") or model, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
                calls = _tool_calls(message.get("tool_calls"), completion_id)
                if calls:
                    yield _sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}]})
                if event.get("done"):
                    usage = {"prompt_tokens": event.get("prompt_eval_count", 0), "completion_tokens": event.get("eval_count", 0)}
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else (event.get("done_reason") or "stop")}], "usage": usage})
    except OllamaError as exc:
        yield _sse({"error": {"message": str(exc), "type": "ollama_error", "code": exc.status_code}})
    yield "data: [DONE]\n\n"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
