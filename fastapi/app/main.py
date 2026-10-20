import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .routers import chat, health, models
from .services.ollama_client import OllamaClient, OllamaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ollama = OllamaClient(get_settings())
    yield
    await app.state.ollama.close()


app = FastAPI(title="Ollama OpenAI Gateway", version="1.0.0", lifespan=lifespan)
if origins := get_settings().allowed_origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
app.include_router(health.router)
app.include_router(models.router)
app.include_router(chat.router)


@app.exception_handler(OllamaError)
async def ollama_error(_request: Request, exc: OllamaError):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": str(exc), "type": "ollama_error", "code": exc.status_code}})
