from odoo.addons.muk_ai.providers import REGISTRY

from .ollama_fastapi import OllamaFastAPIProvider

REGISTRY[OllamaFastAPIProvider.name] = OllamaFastAPIProvider
