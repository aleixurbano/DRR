#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


from reflect.core.paths import prompts_dir, reasoning_sweep_root, sim_data_root


def _split_csv(raw_value: str | None) -> list[str] | None:
    if not raw_value:
        return None
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _discover(task_filters: list[str] | None, episode_filters: list[str] | None) -> pd.DataFrame:
    root = sim_data_root()
    if not root.exists():
        raise FileNotFoundError(f"Simulation dataset root not found: {root}")

    rows = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if task_filters and task_dir.name not in set(task_filters):
            continue
        for episode_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            if episode_filters and episode_dir.name not in set(episode_filters):
                continue
            rows.append({"task": task_dir.name, "episode": episode_dir.name})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GPT reasoning-effort sweep on selected simulation episodes.")
    parser.add_argument("--tasks", help="Comma-separated task filters.")
    parser.add_argument("--episodes", help="Comma-separated episode filters.")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-efforts", help="Comma-separated effort list, e.g. none,low,medium,high,xhigh")
    parser.add_argument("--with-audio", type=int, default=1, choices=[0, 1])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required for the reasoning-effort sweep.")

    prompt_path = prompts_dir() / "sim_prompts.json"
    with open(prompt_path, "r") as fh:
        prompt_template = json.load(fh)

    episodes_df = _discover(_split_csv(args.tasks), _split_csv(args.episodes))
    if args.max_episodes > 0:
        episodes_df = episodes_df.head(args.max_episodes)
    if episodes_df.empty:
        raise SystemExit("No simulation episodes matched the requested filters.")

    from reflect.pipelines.reasoning_sweep import run_reasoning_effort_sweep

    reasoning_efforts = _split_csv(args.reasoning_efforts)
    results = run_reasoning_effort_sweep(
        episodes_df=episodes_df,
        api_key=api_key,
        model=args.model,
        prompt_template=prompt_template,
        sim_data_dir=str(sim_data_root()),
        results_dir=str(reasoning_sweep_root()),
        reasoning_efforts=reasoning_efforts,
        with_audio=args.with_audio,
        max_workers=args.max_workers,
    )

    print(f"[done] saved sweep payloads under {reasoning_sweep_root()}")
    print(f"[efforts] {', '.join(results.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
