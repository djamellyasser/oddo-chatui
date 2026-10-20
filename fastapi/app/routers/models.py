from fastapi import APIRouter, Depends, Request

from ..schemas import ModelObject
from ..security import authenticate

router = APIRouter()


@router.get("/v1/models", dependencies=[Depends(authenticate)])
async def models(request: Request):
    entries = await request.app.state.ollama.models()
    data = []
    for item in entries:
        if not item.get("name"):
            continue
        model = ModelObject(id=item["name"]).model_dump()
        # Ollama's tag listing does not consistently expose a context window.
        # Keep this optional extension absent instead of guessing a capability.
        if isinstance(item.get("context_window"), int):
            model["context_window"] = item["context_window"]
        data.append(model)
    return {"object": "list", "data": data}
