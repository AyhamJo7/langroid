"""GeminiEmbeddings request shape, checked against a fake client (no API key)."""

import math
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

genai = pytest.importorskip("google.genai")

from langroid.embedding_models.models import (  # noqa: E402
    GeminiEmbeddings,
    GeminiEmbeddingsConfig,
)

# Shut down on the Gemini API (https://ai.google.dev/gemini-api/docs/deprecations):
# embedding-001 on 2025-10-30, text-embedding-004 on 2026-01-14.
SHUT_DOWN = {"text-embedding-004", "embedding-001"}


class _FakeModels:
    def __init__(self, native_dims: int):
        self.native_dims = native_dims
        self.calls: List[Dict[str, Any]] = []

    def embed_content(self, model: str, contents: List[str], config: Any = None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        dims = getattr(config, "output_dimensionality", None) or self.native_dims
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[3.0] * dims) for _ in contents]
        )


def _model(monkeypatch, config: GeminiEmbeddingsConfig) -> GeminiEmbeddings:
    fake = _FakeModels(native_dims=3072)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", lambda api_key: SimpleNamespace(models=fake))
    return GeminiEmbeddings(config)


def test_default_model_is_still_served():
    name = GeminiEmbeddingsConfig().model_name
    assert name.removeprefix("models/") not in SHUT_DOWN


def test_requests_the_configured_dims_and_normalizes(monkeypatch):
    model = _model(monkeypatch, GeminiEmbeddingsConfig(dims=768))
    embeddings = model.generate_embeddings(["hello", "world"])

    call = model.client.models.calls[0]
    assert call["config"].output_dimensionality == 768
    assert len(embeddings) == 2
    for emb in embeddings:
        assert len(emb) == model.embedding_dims == 768
        assert math.isclose(math.sqrt(sum(v * v for v in emb)), 1.0, rel_tol=1e-9)


def test_native_dims_are_returned_unchanged(monkeypatch):
    model = _model(monkeypatch, GeminiEmbeddingsConfig(dims=3072))
    embeddings = model.generate_embeddings(["hello"])

    assert model.client.models.calls[0]["config"].output_dimensionality == 3072
    assert embeddings == [[3.0] * 3072]
