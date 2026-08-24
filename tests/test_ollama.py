"""HTTP 層。リトライすべき失敗としてはいけない失敗を分ける。"""

import json
import urllib.error

import pytest

from quirkbench.ollama import Ollama, OllamaError


class _Response:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, behaviour):
    calls = {"n": 0}

    def fake(request, timeout=None):
        calls["n"] += 1
        return behaviour(calls["n"])

    monkeypatch.setattr("quirkbench.ollama.urllib.request.urlopen", fake)
    return calls


def test_retries_transient_failure(monkeypatch):
    def behaviour(n):
        if n < 3:
            raise urllib.error.URLError("refused")
        return _Response({"version": "1.2.3"})

    calls = _patch_urlopen(monkeypatch, behaviour)
    assert Ollama(retries=2, backoff=0).version() == "1.2.3"
    assert calls["n"] == 3


def test_gives_up_after_retries(monkeypatch):
    _patch_urlopen(monkeypatch, lambda n: (_ for _ in ()).throw(urllib.error.URLError("down")))
    with pytest.raises(OllamaError):
        Ollama(retries=1, backoff=0).version()


def test_client_error_is_not_retried(monkeypatch):
    """4xx はこちらの誤りなので、繰り返しても直らない。無駄に待たない。"""

    def behaviour(n):
        raise urllib.error.HTTPError("u", 404, "nope", {}, None)

    calls = _patch_urlopen(monkeypatch, behaviour)
    with pytest.raises(OllamaError):
        Ollama(retries=3, backoff=0).version()
    assert calls["n"] == 1


def test_generate_requires_num_ctx():
    """実効値が記録できないと longctx の結果が解釈できなくなる。"""
    with pytest.raises(ValueError, match="num_ctx"):
        Ollama().generate("m", "p", {"temperature": 0})


def test_embed_short_circuits_on_empty_input(monkeypatch):
    calls = _patch_urlopen(monkeypatch, lambda n: _Response({"embeddings": []}))
    assert Ollama().embed("bge-m3", []) == []
    assert calls["n"] == 0


def test_model_info_combines_tags_and_show(monkeypatch):
    """digest は /api/tags にしかなく、context_length は /api/show にしかない。"""

    def behaviour(n):
        if n == 1:
            return _Response({"models": [{"model": "m", "digest": "abc"}]})
        return _Response(
            {
                "details": {
                    "parameter_size": "1.5B",
                    "quantization_level": "Q4",
                    "family": "qwen2",
                },
                "model_info": {"qwen2.context_length": 32768},
            }
        )

    _patch_urlopen(monkeypatch, behaviour)
    info = Ollama().model_info("m")
    assert (info.digest, info.max_context, info.parameter_size) == ("abc", 32768, "1.5B")


def test_unknown_model_is_rejected(monkeypatch):
    _patch_urlopen(monkeypatch, lambda n: _Response({"models": []}))
    with pytest.raises(OllamaError, match="pull"):
        Ollama().model_info("missing")
