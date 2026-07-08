"""Compute the REFLECT reproduction metrics over all 100 episodes from the
committed final-run outputs, and write outputs/results/sim_data/reproduction_metrics.csv.

These are the paper-comparable definitions (REFLECT Table 1), computed per episode
and split by ground-truth failure type (execution / planning):

- Exp (explanation): human score that the predicted failure reason is correct.
  From exp_evaluation_run4.csv (human_score in {0, 1}).
- Loc (localization): the predicted failure step matches the ground-truth step.
  From reflect_results_run4.json reasoning_dict (pred_failure_step contains gt_failure_step).
- Co-plan (correction plan): the corrected plan executed successfully in simulation.
  From reflect_results_run4.json correction_dict.success.

Exp is human-annotated (unchanged); Loc and Co-plan are read directly from the
recorded pipeline output, so re-running the pipeline is not required to reproduce them.
The initial-vs-final Exp improvement uses exp_evaluation_run1.csv (initial reproduction).

Usage:
    pip install -e .
    python analysis/scripts/reproduction_metrics.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from reflect.core.paths import sim_results_root

SIM = sim_results_root()
SPLITS = ("execution", "planning", "all")


def norm_steps(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(s).strip() for s in value]
    return [str(value).strip()]


def exp_rates() -> dict[str, tuple[int, int]]:
    """Human explanation score per split from the run4 evaluation."""
    rows = list(csv.DictReader(open(SIM / "exp_evaluation_run4.csv")))
    acc = defaultdict(lambda: [0, 0])
    for row in rows:
        score = row.get("human_score", "")
        if score in ("", "None"):
            continue
        split = row.get("gt_error_type") or "unknown"
        for key in (split, "all"):
            acc[key][0] += int(round(float(score)))
            acc[key][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def loc_coplan_rates() -> dict[str, dict[str, tuple[int, int]]]:
    """Loc (step match) and Co-plan (correction success) per split from run4 outputs."""
    results = json.load(open(SIM / "reflect_results_run4.json"))
    loc = defaultdict(lambda: [0, 0])
    cop = defaultdict(lambda: [0, 0])
    for episode in results:
        reasoning = episode.get("reasoning_dict") or {}
        correction = episode.get("correction_dict") or {}
        split = reasoning.get("gt_error_type") or "unknown"
        gt = norm_steps(reasoning.get("gt_failure_step"))
        pred = norm_steps(reasoning.get("pred_failure_step"))
        loc_ok = bool(gt) and any(step in pred for step in gt)
        for key in (split, "all"):
            if gt:
                loc[key][0] += int(loc_ok)
                loc[key][1] += 1
            if "success" in correction:
                cop[key][0] += int(bool(correction["success"]))
                cop[key][1] += 1
    return {"Loc": {k: tuple(v) for k, v in loc.items()},
            "Co-plan": {k: tuple(v) for k, v in cop.items()}}


def initial_exp() -> tuple[float, int]:
    rows = list(csv.DictReader(open(SIM / "exp_evaluation_run1.csv")))
    vals = [float(r["human_score"]) for r in rows if r.get("human_score", "") not in ("", "None")]
    return (100 * sum(vals) / len(vals) if vals else float("nan")), len(vals)


def pct(hit: int, n: int) -> str:
    return f"{100 * hit / n:.1f}" if n else "n/a"


def main() -> None:
    exp = exp_rates()
    lc = loc_coplan_rates()
    metrics = {"Exp": exp, "Loc": lc["Loc"], "Co-plan": lc["Co-plan"]}

    out_path = SIM / "reproduction_metrics.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "split", "hits", "n", "rate_pct"])
        for metric in ("Exp", "Loc", "Co-plan"):
            for split in SPLITS:
                hit, n = metrics[metric].get(split, (0, 0))
                writer.writerow([metric, split, hit, n, pct(hit, n)])
        init_mean, init_n = initial_exp()
        writer.writerow(["Exp", "initial_run1", "", init_n, f"{init_mean:.1f}"])
    print("wrote", out_path)

    print("\nREADME reproduction table (all 100 episodes, run4):\n")
    print("| Metric | DRR exec | DRR plan | DRR all | REFLECT (exec / plan) |")
    print("|---|---|---|---|---|")
    paper = {"Exp": "88.4 / 84.2", "Loc": "96.0 / 80.7", "Co-plan": "79.1 / 80.7"}
    labels = {"Exp": "Explanation (Exp)", "Loc": "Localization (Loc)", "Co-plan": "Correction plan (Co-plan)"}
    for metric in ("Exp", "Loc", "Co-plan"):
        e = pct(*metrics[metric].get("execution", (0, 0)))
        p = pct(*metrics[metric].get("planning", (0, 0)))
        a = pct(*metrics[metric].get("all", (0, 0)))
        print(f"| {labels[metric]} | {e} | {p} | {a} | {paper[metric]} |")
    init_mean, init_n = initial_exp()
    print(f"\nExplanation improvement: {init_mean:.1f}% initial (n={init_n}) -> "
          f"{pct(*exp['all'])}% final (n={exp['all'][1]})")


if __name__ == "__main__":
    main()
