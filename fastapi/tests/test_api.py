import json

import httpx
import pytest


def response(data, status=200):
    return httpx.Response(status, json=data)


@pytest.mark.asyncio
async def test_invalid_token(gateway):
    async with gateway(lambda _request: response({})) as client:
        assert (await client.get("/health")).status_code == 401


@pytest.mark.asyncio
async def test_health(gateway, auth):
    async with gateway(lambda _request: response({})) as client:
        assert (await client.get("/health", headers=auth)).json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_model_discovery(gateway, auth):
    def handler(request):
        if request.url.path == "/api/tags":
            return response({"models": [{"name": "llama3.1:8b"}, {"name": "qwen3:8b"}]})
        assert request.url.path == "/api/show"
        return response({"model_info": {"llama.context_length": 8192}})
    async with gateway(handler) as client:
        result = await client.get("/v1/models", headers=auth)
    assert [model["id"] for model in result.json()["data"]] == ["llama3.1:8b", "qwen3:8b"]
    assert result.json()["data"][0]["context_window"] == 8192


@pytest.mark.asyncio
async def test_chat_and_multi_turn_options(gateway, auth):
    def handler(request):
        body = json.loads(request.content)
        assert [message["role"] for message in body["messages"]] == ["system", "user", "assistant", "user"]
        assert body["options"] == {"temperature": 0.2, "top_p": 0.9, "num_predict": 50, "stop": ["END"]}
        return response({"model": body["model"], "message": {"role": "assistant", "content": "four"}, "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 1})
    payload = {"model": "llama3.1:8b", "messages": [
        {"role": "system", "content": "Be concise"}, {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"}, {"role": "user", "content": "spell it"}],
        "temperature": .2, "top_p": .9, "max_tokens": 50, "stop": "END"}
    async with gateway(handler) as client:
        result = await client.post("/v1/chat/completions", headers=auth, json=payload)
    assert result.status_code == 200
    assert result.json()["choices"][0]["message"]["content"] == "four"
    assert result.json()["usage"]["total_tokens"] == 11


@pytest.mark.asyncio
async def test_streaming(gateway, auth):
    lines = [
        {"model": "llama3.1:8b", "message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"model": "llama3.1:8b", "message": {"role": "assistant", "content": "lo"}, "done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2},
    ]
    def handler(_request):
        return httpx.Response(200, content="".join(json.dumps(line) + "\n" for line in lines), headers={"content-type": "application/x-ndjson"})
    async with gateway(handler) as client:
        result = await client.post("/v1/chat/completions", headers=auth, json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert result.status_code == 200
    assert '"role":"assistant"' in result.text
    assert '"content":"Hel"' in result.text
    assert result.text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_tool_call_compatibility(gateway, auth):
    def handler(request):
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "weather"
        return response({"model": body["model"], "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Lagos"}}}]}, "done": True})
    payload = {"messages": [{"role": "user", "content": "weather?"}], "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]}
    async with gateway(handler) as client:
        result = await client.post("/v1/chat/completions", headers=auth, json=payload)
    call = result.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["id"].startswith("call_")
    assert json.loads(call["function"]["arguments"]) == {"city": "Lagos"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error,expected", [(404, "model not found", 404), (500, "server exploded", 502)])
async def test_ollama_errors(gateway, auth, status, error, expected):
    async with gateway(lambda _request: response({"error": error}, status)) as client:
        result = await client.post("/v1/chat/completions", headers=auth, json={"model": "missing", "messages": [{"role": "user", "content": "hi"}]})
    assert result.status_code == expected
    assert error in result.json()["error"]["message"]


@pytest.mark.asyncio
async def test_ollama_unavailable(gateway, auth):
    def handler(request):
        raise httpx.ConnectError("offline", request=request)
    async with gateway(handler) as client:
        result = await client.get("/v1/models", headers=auth)
    assert result.status_code == 503


@pytest.mark.asyncio
async def test_streaming_ollama_unavailable_is_sse_error(gateway, auth):
    def handler(request):
        raise httpx.ConnectError("offline", request=request)
    async with gateway(handler) as client:
        result = await client.post("/v1/chat/completions", headers=auth, json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert result.status_code == 200
    assert '"type":"ollama_error"' in result.text
    assert result.text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_request_timeout(gateway, auth):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    async with gateway(handler) as client:
        result = await client.get("/v1/models", headers=auth)
    assert result.status_code == 504
