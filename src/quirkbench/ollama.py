"""Ollama クライアント。

**HTTP を行うのはこのモジュールだけ。** 依存を増やさないため stdlib の urllib を使う。

``num_ctx`` の実効値は API から読めない（``/api/show`` が返すのはモデルアーキテクチャの
最大長であって、その呼び出しで実際に使われた値ではない）。したがって
**呼び出し側が必ず ``num_ctx`` を明示する**。曖昧さを残さないための設計判断であり、
これを守らないと longctx の結果が解釈できなくなる。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    """Ollama への呼び出しが最終的に失敗した。"""


@dataclass(frozen=True)
class ModelInfo:
    """1 モデルの同定情報。digest は ``/api/tags`` 由来（``/api/show`` は返さない）。"""

    name: str
    digest: str
    parameter_size: str
    quantization_level: str
    family: str
    max_context: int | None


class Ollama:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        timeout: float = 600.0,
        retries: int = 2,
        backoff: float = 2.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    # ------------------------------------------------------------------ HTTP

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.host + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data else {}

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                # 4xx はこちらの誤りなので、繰り返しても直らない
                if 400 <= exc.code < 500:
                    raise OllamaError(f"{path} -> HTTP {exc.code}: {exc.read()[:200]!r}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(self.backoff * (attempt + 1))
        raise OllamaError(f"{path} が {self.retries + 1} 回失敗した: {last!r}") from last

    # ----------------------------------------------------------------- 情報系

    def version(self) -> str:
        return str(self._request("/api/version")["version"])

    def tags(self) -> list[dict[str, Any]]:
        return list(self._request("/api/tags")["models"])

    def show(self, model: str) -> dict[str, Any]:
        return self._request("/api/show", {"model": model})

    def model_info(self, model: str) -> ModelInfo:
        """digest（/api/tags）と諸元（/api/show）を 1 つにまとめて返す。

        **タグを省いた名前を ``:latest`` として解決する。** ollama 自身の規約で、
        ``/api/generate`` と ``/api/embed`` は ``bge-m3`` を受け付けるが
        ``/api/tags`` は ``bge-m3:latest`` としか名乗らない。解決しないと、
        **生成は通るのに digest だけ引けない**という形で落ちる（S4 で実際に踏んだ）。

        返す ``name`` は**解決後の名前**にする。``bge-m3`` と ``bge-m3:latest`` が
        別の名前のまま指紋に入ると、同じモデルが書き方だけで別キーになる（§13.5）。
        """
        candidates = [model] if ":" in model else [model, f"{model}:latest"]
        resolved = ""
        digest = ""
        for entry in self.tags():
            names = {str(entry.get("model", "")), str(entry.get("name", ""))}
            hit = next((c for c in candidates if c in names), None)
            if hit is not None:
                resolved = hit
                digest = str(entry.get("digest", ""))
                break
        if not digest:
            raise OllamaError(
                f"モデル {model!r} が /api/tags に見つからない。ollama pull は済んでいるか"
            )
        model = resolved

        shown = self.show(model)
        details = shown.get("details") or {}
        info = shown.get("model_info") or {}
        max_context = next((int(v) for k, v in info.items() if k.endswith(".context_length")), None)
        return ModelInfo(
            name=model,
            digest=digest,
            parameter_size=str(details.get("parameter_size", "")),
            quantization_level=str(details.get("quantization_level", "")),
            family=str(details.get("family", "")),
            max_context=max_context,
        )

    # ----------------------------------------------------------------- 生成系

    def generate(self, model: str, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """1 回の生成。``options`` には ``num_ctx`` を必ず含めること。"""
        if "num_ctx" not in options:
            raise ValueError("options に num_ctx が無い。実効値が記録できなくなるため必須")
        return self._request(
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": False, "options": dict(options)},
        )

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        return list(self._request("/api/embed", {"model": model, "input": inputs})["embeddings"])
