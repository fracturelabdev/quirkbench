"""生成の実行（オーケストレーションのみ）。

HTTP は :mod:`ollama`、永続化は :mod:`store`、キーの導出は :mod:`keys` が持つ。
ここに処理を集めない。

**モデル単位でまとめて回す。** ケースごとにモデルを切り替えるとその都度モデルのロードが走り、
実時間が数倍になるうえ ``load_duration`` の測定も汚れる。
"""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from . import keys
from .cases import Case
from .clock import now_iso
from .ollama import ModelInfo, Ollama, OllamaError
from .store import RunStore


class DigestDrift(RuntimeError):
    """run の途中でモデルの中身が入れ替わった。"""


class ContextOverflow(RuntimeError):
    """プロンプトが ``num_ctx`` に収まらず、**黙って切り詰められた**（§15.2）。

    ollama はエラーも警告も出さない。実測では 7,821 文字を ``num_ctx`` 4096 に
    投げると ``prompt_eval_count`` が 2,050 で止まり、モデルは
    「文中に記述はありません」と答えた。そのまま採点すると
    「長文脈で事実を保持できない」という結論が出るが、
    **実際には事実を渡していない。**

    `qb lint-cases` の静的見積り（§15.2）でも落とすが、
    **見積りはトークナイザの近似**なので、実測値でも見る。
    """


@dataclass
class Progress:
    total: int = 0
    skipped: int = 0
    generated: int = 0
    failed: int = 0


def host_info() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        try:
            return subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    memsize = sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "ram_gb": round(int(memsize) / 1024**3) if memsize.isdigit() else None,
        "python": platform.python_version(),
    }


def plan_work(
    cases: Iterable[Case], models: Iterable[str], repeats: int, base_seed: int
) -> list[tuple[str, Case, int]]:
    """(model, case, seed) の並び。モデル単位でまとまる順序で返す。

    ケースごとにモデルを切り替えると毎回ロードが走り、``load_duration`` も汚れる
    （実測: 1209ms → 135ms → 97ms）。反復の連番は seed の導出にしか使わないので返さない。
    """
    return [
        (model, case, base_seed + index)
        for model in models
        for case in cases
        for index in range(repeats)
    ]


def run(
    *,
    store: RunStore,
    client: Ollama,
    cases: list[Case],
    models: list[str],
    repeats: int,
    base_seed: int = 1000,
    allow_digest_drift: bool = False,
    report: Callable[[str], None] = print,
) -> Progress:
    completed, read_report = store.completed_keys()
    if read_report.has_warning:
        report(
            f"警告: generations.jsonl の途中に壊れた行が {read_report.corrupt_middle} 件ある。"
            "ディスク障害か二重起動の可能性がある"
        )
    if read_report.corrupt_tail:
        report("注記: 末尾に書きかけの行があったので捨てた（該当の生成は再実行される）")

    unreliable = store.warn_unreliable_fs()
    if unreliable:
        report(f"警告: {unreliable}")

    meta = store.read_meta()
    known_digests: dict[str, str] = dict(meta.get("model_digests") or {})

    infos: dict[str, ModelInfo] = {}
    for model in models:
        info = client.model_info(model)
        previous = known_digests.get(model)
        if previous and previous != info.digest:
            message = (
                f"{model} の digest が run の途中で変わった "
                f"({previous[:12]} -> {info.digest[:12]})。"
            )
            if not allow_digest_drift:
                raise DigestDrift(message + " --allow-digest-drift で続行できる")
            report("警告: " + message + " 続行する（レポートは両方の digest を併記する）")
        infos[model] = info
        known_digests[model] = info.digest

    ollama_version = client.version()
    store.write_meta(
        {
            **meta,
            "run_id": store.run_id,
            "created_at": meta.get("created_at") or now_iso(),
            "updated_at": now_iso(),
            "ollama_version": ollama_version,
            "host": host_info(),
            "model_digests": known_digests,
            "models": {m: vars(i) for m, i in infos.items()},
            "repeats": repeats,
            "base_seed": base_seed,
        }
    )

    work = plan_work(cases, models, repeats, base_seed)
    progress = Progress(total=len(work))

    for position, (model, case, seed) in enumerate(work, start=1):
        info = infos[model]
        key_parts: keys.KeyParts = {
            "model": model,
            "model_digest": info.digest,
            "case_id": case.id,
            "prompt_hash": case.prompt_hash,
            "seed": seed,
            "options_hash": case.options_hash,
            "attempt": 0,
        }
        if keys.completion_key(key_parts) in completed:
            progress.skipped += 1
            continue

        options = {**case.options, "seed": seed}
        started = time.monotonic()
        try:
            response = client.generate(model, case.prompt, options)
            error = None
        except OllamaError as exc:
            response, error = {}, str(exc)
            progress.failed += 1

        # **切り詰めは実測値でしか分からない。** 記録する前に見る —
        # 記録してから気づいても、その run のデータはもう解釈できない（§15.2）
        used = response.get("prompt_eval_count")
        if error is None and used is not None and int(used) >= int(options["num_ctx"]):
            raise ContextOverflow(
                f"{case.id} / {model}: プロンプトが num_ctx={options['num_ctx']} を使い切っている"
                f"（prompt_eval_count={used}）。**黙って切り詰められている**ので、"
                "測っているのは長文脈保持ではなく切り詰め。ケースの num_ctx を上げること"
            )

        store.append_prompt(case.prompt_hash, case.prompt)
        store.append_generation(
            {
                "gen_id": keys.gen_id(run_id=store.run_id, parts=key_parts),
                "run_id": store.run_id,
                "ts": now_iso(),
                "model": model,
                "model_digest": info.digest,
                "parameter_size": info.parameter_size,
                "quantization_level": info.quantization_level,
                "ollama_version": ollama_version,
                "dimension": case.dim,
                "lang": case.lang,
                "case_id": case.id,
                "prompt_hash": case.prompt_hash,
                "options": options,
                "options_hash": case.options_hash,
                "num_ctx": options["num_ctx"],
                "max_context": info.max_context,
                "seed": seed,
                "attempt": 0,
                "response": response.get("response"),
                "done_reason": response.get("done_reason"),
                "load_duration_ns": response.get("load_duration"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
                "eval_count": response.get("eval_count"),
                "eval_duration_ns": response.get("eval_duration"),
                "total_duration_ns": response.get("total_duration"),
                "wall_seconds": round(time.monotonic() - started, 3),
                "error": error,
            }
        )
        if error is None:
            progress.generated += 1
            completed.add(keys.completion_key(key_parts))

        marker = "ERR" if error else (response.get("done_reason") or "?")
        report(
            f"[{position}/{progress.total}] {model} {case.id} seed={seed} "
            f"{time.monotonic() - started:.1f}s {marker}"
        )

    return progress
