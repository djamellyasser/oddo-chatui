import os

os.environ.setdefault("API_BEARER_TOKEN", "test-secret")

import httpx
import pytest
from httpx import ASGITransport

from app.config import get_settings
from app.main import app
from app.services.ollama_client import OllamaClient


@pytest.fixture
def gateway():
    def factory(handler):
        ollama = OllamaClient(get_settings(), transport=httpx.MockTransport(handler))

        class Context:
            async def __aenter__(self):
                app.state.ollama = ollama
                self.client = httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://gateway"
                )
                return self.client

            async def __aexit__(self, *_args):
                await self.client.aclose()
                await ollama.close()

        return Context()
    return factory


@pytest.fixture
def auth():
    return {"Authorization": "Bearer test-secret"}
