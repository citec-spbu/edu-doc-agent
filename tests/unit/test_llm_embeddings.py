"""Эмбеддинги через Ollama: запрос, пачки, повторы, проверка ответа. Сеть не используется."""

import json
import math

import httpx
import pytest

from edu_doc_agent.config import Settings
from edu_doc_agent.llm import FakeEmbedder, LLMError, OllamaEmbedder, embeddings

DIM = 4


def _embedder(handler, **kwargs) -> OllamaEmbedder:
    kwargs.setdefault("dim", DIM)
    return OllamaEmbedder(
        base_url="http://ollama:11434/", transport=httpx.MockTransport(handler), **kwargs
    )


def _vectors(texts: list[str]) -> list[list[float]]:
    # вектор кодирует длину строки — по нему видно, что порядок сохранился
    return [[float(len(t)), 0.0, 0.0, 1.0] for t in texts]


def _ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    return httpx.Response(200, json={"model": "bge-m3", "embeddings": _vectors(texts)})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(embeddings.time, "sleep", slept.append)
    return slept


def test_request_disables_truncation_and_sets_context() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(request)

    assert _embedder(handler).embed(["пункт 1.1", "пункт 2.3"]) == _vectors(
        ["пункт 1.1", "пункт 2.3"]
    )
    (request,) = seen
    assert str(request.url) == "http://ollama:11434/api/embed"
    body = json.loads(request.content)
    assert body == {
        "model": "bge-m3",
        "input": ["пункт 1.1", "пункт 2.3"],
        "truncate": False,
        "options": {"num_ctx": 8192},
    }


def test_batches_keep_order() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["input"]))
        return _ok(request)

    texts = ["а" * n for n in range(1, 6)]
    assert _embedder(handler, batch_size=2).embed(texts) == _vectors(texts)
    assert sizes == [2, 2, 1]


def test_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("запроса быть не должно")

    assert _embedder(handler).embed([]) == []


def test_retries_transient_errors(_no_sleep: list[float]) -> None:
    replies = iter([httpx.Response(503), httpx.ConnectError("нет связи")])

    def handler(request: httpx.Request) -> httpx.Response:
        reply = next(replies, None)
        if isinstance(reply, Exception):
            raise reply
        return reply or _ok(request)

    assert _embedder(handler).embed(["текст"]) == _vectors(["текст"])
    assert _no_sleep == [1, 2]


def test_gives_up_after_retries() -> None:
    with pytest.raises(LLMError, match="после 3 попыток"):
        _embedder(lambda request: httpx.Response(502), retries=2).embed(["текст"])


def test_client_error_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "input length exceeds the context length"})

    with pytest.raises(LLMError, match="exceeds the context length"):
        _embedder(handler).embed(["очень длинный пункт"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "reply",
    [
        {"embeddings": [[0.0] * DIM]},  # векторов меньше, чем строк
        {"embeddings": [[0.0] * DIM, [0.0] * (DIM + 1)]},  # не та размерность
        {"error": "нет поля embeddings"},
    ],
)
def test_rejects_malformed_reply(reply: dict) -> None:
    with pytest.raises(LLMError):
        _embedder(lambda request: httpx.Response(200, json=reply)).embed(["а", "б"])


def test_from_settings_defaults_and_overrides() -> None:
    default = OllamaEmbedder.from_settings(Settings(_env_file=None))
    assert (str(default._http.base_url), default.model, default.dim) == (
        "http://localhost:11434/",
        "bge-m3",
        1024,
    )
    custom = OllamaEmbedder.from_settings(
        Settings(
            _env_file=None, embedding_base_url="http://ollama:11434", embedding_model="bge-m3:567m"
        )
    )
    assert (str(custom._http.base_url), custom.model) == ("http://ollama:11434/", "bge-m3:567m")


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_embedder_is_deterministic_and_meaningful() -> None:
    fake = FakeEmbedder(dim=64)
    a, b, c, empty = fake.embed(
        [
            "Объем программы бакалавриата составляет 240 зачетных единиц",
            "объем программы бакалавриата 240 з.е.",
            "Требования к материально-техническому обеспечению",
            "",
        ]
    )
    assert fake.embed(["Объем программы бакалавриата составляет 240 зачетных единиц"])[0] == a
    assert all(len(v) == 64 and math.isclose(_cosine(v, v), 1.0) for v in (a, b, c, empty))
    assert _cosine(a, b) > _cosine(a, c)
    assert len(fake.calls) == 2
