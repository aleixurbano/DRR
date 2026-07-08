"""Print the calibration markdown tables from the committed metric CSVs.

The README "Key findings" tables and the Experiment 16 report tables (sections
8 and 10) are generated here rather than typed by hand, so they cannot
drift from the CSVs the calibration notebooks produce. After re-running the
notebooks, run this and paste the output into README.md and
outputs/analysis/llm_calibration_target_distribution/EXPERIMENT_16_REPORT.md.

Reads:
  outputs/analysis/llm_calibration_target_distribution/metrics_by_{model,type,regime,difficulty}.csv
  outputs/analysis/llm_calibration_target_distribution/metrics_composition.csv
  outputs/analysis/llm_calibration_target_distribution/metrics_by_model_scored.csv
  outputs/analysis/llm_calibration_standard_entropy/metrics_by_model_scored.csv

The scored CSVs may contain extra models that were only run on the standard
benchmark (e.g. qwen3.6:*); every table filters to MODEL_ORDER, the three
models with runs on both distributions.

Usage:
    pip install -e .
    python analysis/scripts/make_calibration_tables.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from reflect.core.paths import analysis_experiment_dir

TARGET_DIR = analysis_experiment_dir("llm_calibration_target_distribution")
STANDARD_DIR = analysis_experiment_dir("llm_calibration_standard_entropy")

# Report and README list the frontier model first, then the local models by size.
MODEL_ORDER = ["gpt-5.4", "qwen3.5:27b", "qwen3.5:9b"]

TYPE_LABELS = {
    "T1": "T1 outcome verification",
    "T2": "T2 failure localization",
    "T3": "T3 failure attribution",
    "T4": "T4 missing step",
}
REGIME_LABELS = {
    "C1": "C1 full trace",
    "C2": "C2 local window",
    "C3": "C3 plan only",
}
DIFFICULTY_LABELS = {
    "hard": "hard (panel 0-1)",
    "medium": "medium (2-3)",
    "easy": "easy (4-5)",
}


def read_rows(path: Path) -> list[dict]:
    with open(path) as handle:
        return list(csv.DictReader(handle))


def by_key(rows: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in rows}


def f3(value: str) -> str:
    return f"{float(value):.3f}"


def signed3(value: str) -> str:
    return f"{float(value):+.3f}"


def standard_vs_target_table() -> str:
    standard = by_key(read_rows(STANDARD_DIR / "metrics_by_model_scored.csv"), "model")
    target = by_key(read_rows(TARGET_DIR / "metrics_by_model_scored.csv"), "model")
    lines = [
        "| Model | Std acc | Std ECE | Std Brier | Std NLL "
        "| Tgt acc | Tgt ECE | Tgt Brier | Tgt NLL |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODEL_ORDER:
        s, t = standard[model], target[model]
        lines.append(
            f"| {model} | {f3(s['accuracy'])} | {f3(s['ece'])} "
            f"| {f3(s['brier'])} | {f3(s['nll'])} "
            f"| {f3(t['accuracy'])} | {f3(t['ece'])} "
            f"| {f3(t['brier'])} | {f3(t['nll'])} |"
        )
    return "\n".join(lines)


def brier_reversal_table() -> str:
    standard = by_key(read_rows(STANDARD_DIR / "metrics_by_model_scored.csv"), "model")
    target = by_key(read_rows(TARGET_DIR / "metrics_by_model_scored.csv"), "model")
    lines = [
        "| Model | Standard Brier | Target Brier | Standard Brier (weighted) "
        "| Target Brier (weighted) |",
        "|---|---|---|---|---|",
    ]
    for model in MODEL_ORDER:
        s, t = standard[model], target[model]
        lines.append(
            f"| {model} | {f3(s['brier'])} | {f3(t['brier'])} "
            f"| {f3(s['brier_w'])} | {f3(t['brier_w'])} |"
        )
    return "\n".join(lines)


def cross_model_table() -> str:
    target = by_key(read_rows(TARGET_DIR / "metrics_by_model.csv"), "model")
    lines = [
        "| model | accuracy | mean confidence | ECE | gap (conf - acc) |",
        "|-------|----------|-----------------|-----|------------------|",
    ]
    for model in MODEL_ORDER:
        r = target[model]
        lines.append(
            f"| {model} | {f3(r['accuracy'])} | {f3(r['mean_conf'])} "
            f"| {f3(r['ece'])} | {signed3(r['gap'])} |"
        )
    return "\n".join(lines)


def by_type_table() -> str:
    rows = by_key(read_rows(TARGET_DIR / "metrics_by_type.csv"), "rtype")
    lines = ["| type | accuracy | ECE |", "|------|----------|-----|"]
    for code in ["T1", "T2", "T3", "T4"]:
        r = rows[code]
        lines.append(f"| {TYPE_LABELS[code]} | {f3(r['accuracy'])} | {f3(r['ece'])} |")
    return "\n".join(lines)


def by_regime_table() -> str:
    rows = by_key(read_rows(TARGET_DIR / "metrics_by_regime.csv"), "regime")
    lines = ["| regime | accuracy | ECE |", "|--------|----------|-----|"]
    for code in ["C1", "C2", "C3"]:
        r = rows[code]
        lines.append(f"| {REGIME_LABELS[code]} | {f3(r['accuracy'])} | {f3(r['ece'])} |")
    return "\n".join(lines)


def by_difficulty_table() -> str:
    rows = by_key(read_rows(TARGET_DIR / "metrics_by_difficulty.csv"), "difficulty")
    lines = ["| bucket | accuracy | ECE | gap |", "|--------|----------|-----|-----|"]
    for code in ["hard", "medium", "easy"]:
        r = rows[code]
        lines.append(
            f"| {DIFFICULTY_LABELS[code]} | {f3(r['accuracy'])} "
            f"| {f3(r['ece'])} | {signed3(r['gap'])} |"
        )
    return "\n".join(lines)


def composition_table() -> str:
    rows = by_key(read_rows(TARGET_DIR / "metrics_composition.csv"), "model")
    lines = [
        "| model | micro ECE (as generated) | difficulty-balanced ECE | diagnostic-only ECE (T2+T3) |",
        "|-------|--------------------------|--------------------------|------------------------------|",
    ]
    for model in ["qwen3.5:27b", "qwen3.5:9b", "gpt-5.4"]:
        r = rows[model]
        lines.append(
            f"| {model} | {f3(r['micro_ece'])} | {f3(r['balanced_ece'])} "
            f"| {f3(r['nontrivial_ece'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    sections = [
        ("README Key findings: standard vs target", standard_vs_target_table),
        ("Report section 10.1: cross-model (target set)", cross_model_table),
        ("Report section 10.2: by reasoning type", by_type_table),
        ("Report section 10.3: by context regime", by_regime_table),
        ("Report section 10.4: by difficulty", by_difficulty_table),
        ("Report section 10.5: Brier rank reversal (standard vs target)", brier_reversal_table),
        ("Report section 8: composition sensitivity", composition_table),
    ]
    for title, builder in sections:
        print(f"\n## {title}\n")
        print(builder())


if __name__ == "__main__":
    main()
