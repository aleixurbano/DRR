#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path


from reflect.core.paths import (
    data_root,
    output_root,
    prompts_dir,
    real_world_data_root,
    real_world_tasks_config,
    sim_data_root,
    sim_tasks_config,
    REPO_ROOT,
)


CORE_SIM_DEPS = ("ai2thor", "torch", "moviepy", "ollama")
OPTIONAL_AUDIO_DEPS = ("wav2clip",)


def _check_path(label: str, path: Path, *, should_exist: bool = True) -> bool:
    exists = path.exists()
    status = "OK" if exists == should_exist else "MISSING"
    print(f"[{status}] {label}: {path}")
    return exists == should_exist


def _check_python_dependencies() -> list[bool]:
    print(f"[OK] Active Python: {sys.executable}")
    results = []

    for module_name in CORE_SIM_DEPS:
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        detail = "importable" if ok else "missing"
        _print_optional_check(f"Dependency `{module_name}`", ok, detail)
        results.append(ok)

    for module_name in OPTIONAL_AUDIO_DEPS:
        spec = importlib.util.find_spec(module_name)
        ok = spec is not None
        detail = "importable" if ok else "missing; use `--with-audio 0` or install it for audio detection"
        _print_optional_check(f"Optional dependency `{module_name}`", ok, detail)

    return results


def _print_optional_check(label: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "WARN"
    print(f"[{status}] {label}: {detail}")


def _check_ollama_support() -> None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import ollama
    except ImportError:
        _print_optional_check(
            "Ollama Python package",
            False,
            "missing; install with `python -m pip install \"ollama>=0.6.1\"` for local uncertainty scoring",
        )
        return

    _print_optional_check("Ollama Python package", True, f"importable via {host}")

    client = ollama.Client(host=host)
    try:
        client.ps()
    except Exception as exc:
        _print_optional_check("Ollama server", False, str(exc))
        return

    _print_optional_check("Ollama server", True, host)

    try:
        models = list(getattr(client.list(), "models", []) or [])
    except Exception as exc:
        _print_optional_check("Ollama model listing", False, str(exc))
        return

    if not models:
        _print_optional_check(
            "Ollama logprobs",
            False,
            "server reachable but no local models are available to verify `chat(..., logprobs=True)`",
        )
        return

    model_name = getattr(models[0], "model", None) or getattr(models[0], "name", None)
    if not model_name:
        _print_optional_check("Ollama logprobs", False, "could not determine a model name from `ollama list`")
        return

    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Answer with exactly A.\nA. Yes\nB. No"}],
            stream=False,
            logprobs=True,
            top_logprobs=2,
            options={"num_predict": 1, "temperature": 0},
        )
        has_logprobs = bool(getattr(response, "logprobs", None))
        _print_optional_check(
            "Ollama logprobs",
            has_logprobs,
            f"model={model_name}" if has_logprobs else f"model={model_name} returned no logprobs",
        )
    except Exception as exc:
        _print_optional_check("Ollama logprobs", False, f"model={model_name} error: {exc}")


def main() -> int:
    print("REFLECT hand-off setup check")
    print(f"Repo root: {REPO_ROOT}")
    print(f"REFLECT_DATA_ROOT={os.environ.get('REFLECT_DATA_ROOT', '<unset>')}")
    print(f"REFLECT_OUTPUT_ROOT={os.environ.get('REFLECT_OUTPUT_ROOT', '<unset>')}")
    print()

    checks = [
        _check_path("Simulation task config", sim_tasks_config()),
        _check_path("Real-world task config", real_world_tasks_config()),
        _check_path("Prompt config directory", prompts_dir()),
        _check_path("Data root", data_root()),
        _check_path("Simulation dataset root", sim_data_root()),
        _check_path("Real-world dataset root", real_world_data_root()),
    ]
    dep_checks = _check_python_dependencies()

    output_root().mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output root is writable: {output_root()}")
    _check_ollama_support()

    if all(checks) and all(dep_checks):
        print("\nSetup looks good.")
        return 0

    print(
        "\nSome required paths or dependencies are missing. Set REFLECT_DATA_ROOT to the directory that contains "
        "`sim_data/` and `reflect_dataset/real_data/`, activate the correct Python environment, then rerun this check."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
