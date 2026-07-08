import json
import os
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from reflect.core.paths import reasoning_sweep_output_dir, reasoning_sweep_result_path

try:
    from reflect.pipelines.fast_validation import process_episode_mem
except ImportError:
    from reflect.pipelines.fast_validation import process_episode_mem

try:
    from reflect.llm.prompter import OpenAILLMPrompter
except ImportError:
    from reflect.llm.prompter import OpenAILLMPrompter


GPT5_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")
GPT5_PRO_REASONING_EFFORTS = ("medium", "high", "xhigh")
EFFORT_ORDER = {"baseline_run4": -1, "none": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4}


def default_reasoning_efforts_for_model(model: str) -> Sequence[str]:
    model = str(model).lower()
    if model in {"gpt-5.4-pro", "gpt-5.2-pro"}:
        return GPT5_PRO_REASONING_EFFORTS
    return GPT5_REASONING_EFFORTS


def api_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    if effort in (None, "", "none", "default"):
        return None
    return effort


def effort_slug(effort: Optional[str]) -> str:
    return "none" if api_reasoning_effort(effort) is None else str(effort)


def _episode_num(episode: str) -> int:
    try:
        return int(str(episode).rsplit("-", 1)[-1])
    except Exception:
        return -1


def _progress_checkpoints(total: int) -> Sequence[int]:
    if total <= 0:
        return ()
    return tuple(
        sorted(
            {
                max(1, round(total * frac))
                for frac in (0.25, 0.5, 0.75, 1.0)
            }
        )
    )


def _status_counts(rows: Sequence[Mapping]) -> tuple[int, int]:
    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    error_count = len(rows) - ok_count
    return ok_count, error_count


def _print_progress(label: str, effort_name: str, completed: int, total: int, ok_count: int, error_count: int, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    print(
        f"[{label}] {effort_name}: {completed}/{total} | ok={ok_count} error={error_count} | {elapsed:.0f}s",
        flush=True,
    )


def build_reasoning_work_items(
    episodes_df: pd.DataFrame,
    sim_data_dir: str,
    llm_prompter,
    prompt_template: Mapping,
    with_audio: int = 1,
    status_dict=None,
):
    work_items = []
    deduped = (
        episodes_df[["task", "episode"]]
        .drop_duplicates()
        .sort_values(["task", "episode"], key=lambda col: col.map(_episode_num) if col.name == "episode" else col)
    )

    for row in deduped.to_dict("records"):
        data_path = os.path.join(sim_data_dir, row["task"], row["episode"])
        item = (data_path, row["task"], row["episode"], llm_prompter, prompt_template, with_audio)
        if status_dict is not None:
            item = item + (status_dict,)
        work_items.append(item)
    return work_items


def _save_sweep_payload(path: str, payload: Mapping) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _load_sweep_payload(path: str) -> Dict:
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, list):
        return {"results": payload}
    return payload


def run_reasoning_effort_sweep(
    episodes_df: pd.DataFrame,
    *,
    api_key: str,
    model: str,
    prompt_template: Mapping,
    sim_data_dir: str,
    results_dir: str,
    reasoning_efforts: Optional[Sequence[str]] = None,
    with_audio: int = 1,
    max_workers: Optional[int] = None,
    load_existing: bool = True,
    save_results: bool = True,
    rerun_errors: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to run the reasoning-effort sweep.")

    reasoning_efforts = tuple(reasoning_efforts or default_reasoning_efforts_for_model(model))
    max_workers = max_workers or min(4, os.cpu_count() or 1)
    output_dir = reasoning_sweep_output_dir(results_dir)

    results_by_effort: Dict[str, Dict] = {}

    for effort in reasoning_efforts:
        effort_name = effort_slug(effort)
        output_path = reasoning_sweep_result_path(results_dir, model, effort)
        existing_results = {}
        episodes_to_run_df = episodes_df.copy()
        loaded_count = 0

        if load_existing and os.path.exists(output_path):
            payload = _load_sweep_payload(output_path)
            payload.setdefault("model", model)
            payload.setdefault("reasoning_effort", effort_name)
            loaded_results = payload.get("results", [])
            loaded_count = len(loaded_results)
            loaded_ok_count, loaded_error_count = _status_counts(loaded_results)
            existing_results = {
                (row.get("task"), row.get("episode")): row
                for row in loaded_results
            }

            if rerun_errors:
                episodes_to_run_df = episodes_df[
                    episodes_df.apply(
                        lambda row: existing_results.get((row["task"], row["episode"]), {}).get("status") != "ok",
                        axis=1,
                    )
                ].copy()
            else:
                episodes_to_run_df = episodes_df.iloc[0:0].copy()

            if verbose:
                print(
                    f"[cache] {effort_name}: loaded={loaded_count} ok={loaded_ok_count} "
                    f"error={loaded_error_count} rerun={len(episodes_to_run_df)}",
                    flush=True,
                )

            if episodes_to_run_df.empty:
                results_by_effort[effort_name] = payload
                continue

        prompter = OpenAILLMPrompter(
            api_key=api_key,
            model=model,
            reasoning_effort=api_reasoning_effort(effort),
        )
        work_items = build_reasoning_work_items(
            episodes_df=episodes_to_run_df,
            sim_data_dir=sim_data_dir,
            llm_prompter=prompter,
            prompt_template=prompt_template,
            with_audio=with_audio,
        )

        effort_results = []
        if verbose:
            print(f"[run] {effort_name}: start={len(work_items)} workers={max_workers}", flush=True)

        started_at = time.perf_counter()
        checkpoints = set(_progress_checkpoints(len(work_items)))
        ok_count = 0
        error_count = 0

        if max_workers == 1:
            for index, item in enumerate(work_items, start=1):
                row = process_episode_mem(item)
                effort_results.append(row)
                if row.get("status") == "ok":
                    ok_count += 1
                else:
                    error_count += 1
                if verbose and index in checkpoints:
                    _print_progress("progress", effort_name, index, len(work_items), ok_count, error_count, started_at)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_to_item = {pool.submit(process_episode_mem, item): item for item in work_items}
                for index, future in enumerate(as_completed(future_to_item), start=1):
                    row = future.result()
                    effort_results.append(row)
                    if row.get("status") == "ok":
                        ok_count += 1
                    else:
                        error_count += 1
                    if verbose and index in checkpoints:
                        _print_progress("progress", effort_name, index, len(work_items), ok_count, error_count, started_at)

        merged_results = dict(existing_results)
        for row in effort_results:
            merged_results[(row.get("task"), row.get("episode"))] = row

        effort_results = list(merged_results.values())
        effort_results.sort(key=lambda row: (row.get("task", ""), _episode_num(row.get("episode", ""))))
        merged_ok_count, merged_error_count = _status_counts(effort_results)
        payload = {
            "model": model,
            "reasoning_effort": effort_name,
            "episode_count": len(effort_results),
            "results": effort_results,
        }
        if save_results:
            _save_sweep_payload(output_path, payload)
        results_by_effort[effort_name] = payload
        if verbose:
            elapsed = time.perf_counter() - started_at
            print(
                f"[done] {effort_name}: total={len(effort_results)} ok={merged_ok_count} "
                f"error={merged_error_count} cached={loaded_count} rerun={len(work_items)} | {elapsed:.0f}s",
                flush=True,
            )

    return results_by_effort


def ts_to_sec(ts: str) -> int:
    minutes, seconds = str(ts).strip().split(":")
    return int(minutes) * 60 + int(seconds)


def flatten_steps(value) -> list:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, str):
        return [value]

    flat = []
    for item in value:
        if isinstance(item, list):
            flat.extend(str(x) for x in item)
        else:
            flat.append(str(item))
    return flat


def loc_hit(pred_steps, gt_steps) -> bool:
    pred_steps = flatten_steps(pred_steps)
    gt_steps = flatten_steps(gt_steps)
    if not pred_steps or not gt_steps:
        return False

    try:
        pred_seconds = ts_to_sec(pred_steps[0])
        gt_seconds = [ts_to_sec(step) for step in gt_steps]
    except Exception:
        return False

    if len(gt_seconds) == 1:
        return pred_seconds == gt_seconds[0]
    return gt_seconds[0] <= pred_seconds <= gt_seconds[-1]


def results_to_metrics_frame(results_by_effort: Mapping[str, Mapping]) -> pd.DataFrame:
    rows = []
    for effort_name, payload in results_by_effort.items():
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        for result in results:
            reasoning_dict = result.get("reasoning_dict") or {}
            correction_dict = result.get("correction_dict") or {}
            status_ok = result.get("status") == "ok"
            replay_available = bool(result.get("replay_available", status_ok and not correction_dict.get("skipped")))
            coplan_success = (
                bool(correction_dict.get("success", False))
                if status_ok and replay_available
                else pd.NA
            )
            rows.append(
                {
                    "effort": effort_name,
                    "task": result.get("task"),
                    "episode": result.get("episode"),
                    "status": result.get("status"),
                    "status_ok": status_ok,
                    "artifact_mode": result.get("artifact_mode", "full"),
                    "replay_available": replay_available,
                    "loc_hit": loc_hit(
                        reasoning_dict.get("pred_failure_step"),
                        reasoning_dict.get("gt_failure_step"),
                    )
                    if status_ok
                    else False,
                    "coplan_success": coplan_success,
                    "pred_failure_reason": reasoning_dict.get("pred_failure_reason", ""),
                    "gt_failure_reason": reasoning_dict.get("gt_failure_reason", ""),
                    "pred_failure_step": reasoning_dict.get("pred_failure_step"),
                    "gt_failure_step": reasoning_dict.get("gt_failure_step"),
                    "pred_error_type": reasoning_dict.get("error_type", "unknown"),
                    "gt_error_type": reasoning_dict.get("gt_error_type", "unknown"),
                    "pipeline_error": result.get("error", "") if not status_ok else "",
                }
            )
    return pd.DataFrame(rows)


def results_to_detector_frame(results_by_effort: Mapping[str, Mapping]) -> pd.DataFrame:
    rows = []
    for effort_name, payload in results_by_effort.items():
        results = payload.get("results", payload) if isinstance(payload, dict) else payload
        for result in results:
            reasoning_dict = result.get("reasoning_dict") or {}
            uncertainty_metric = reasoning_dict.get("uncertainty_metric", "entropy")
            detector_trace = reasoning_dict.get("detector_trace") or []
            for trace in detector_trace:
                score = trace.get("score") or {}
                oracle_success = trace.get("oracle_success")
                predicted_success = trace.get("predicted_success")
                predicted_label = trace.get("predicted_label")
                evaluation_active = bool(trace.get("evaluation_active"))
                score_status = score.get("score_status", "unscored")
                constrained_parse_fail = (
                    score_status not in ("available", "text_fallback") or predicted_label is None
                )
                rows.append(
                    {
                        "effort": effort_name,
                        "task": result.get("task"),
                        "episode": result.get("episode"),
                        "step": trace.get("step"),
                        "subgoal": trace.get("subgoal", ""),
                        "evaluation_active": evaluation_active,
                        "predicted_success": predicted_success,
                        "oracle_success": oracle_success,
                        "pred_correct": (
                            predicted_success == oracle_success
                            if predicted_success is not None and oracle_success is not None
                            else False
                        ),
                        "uncertainty_metric": trace.get("uncertainty_metric", uncertainty_metric),
                        "uncertainty_value": trace.get("uncertainty_value"),
                        "confidence": score.get("confidence"),
                        "entropy": score.get("entropy"),
                        "score_status": score_status,
                        "predicted_label": predicted_label,
                        "confidence_label": score.get("confidence_label"),
                        "constrained_parse_fail": constrained_parse_fail,
                        "failed_response_text": (
                            trace.get("response_text", "")
                            if constrained_parse_fail
                            else ""
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _curve_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return float("nan")
    area = 0.0
    for idx in range(1, len(xs)):
        width = xs[idx] - xs[idx - 1]
        area += width * (ys[idx] + ys[idx - 1]) / 2.0
    return area


def detector_accuracy_curve(detector_df: pd.DataFrame) -> pd.DataFrame:
    scored = detector_df[
        detector_df["evaluation_active"]
        & detector_df["uncertainty_value"].notna()
    ].copy()
    if scored.empty:
        return pd.DataFrame(columns=["threshold", "coverage", "accuracy"])

    thresholds = sorted(set(float(value) for value in scored["uncertainty_value"].tolist()) | {0.0, 1.0})
    rows = []
    total = len(scored)
    for threshold in thresholds:
        kept = scored[scored["uncertainty_value"] <= threshold]
        rows.append(
            {
                "threshold": threshold,
                "coverage": len(kept) / total if total else float("nan"),
                "accuracy": kept["pred_correct"].mean() if not kept.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def detector_selective_curve(detector_df: pd.DataFrame) -> pd.DataFrame:
    scored = detector_df[
        detector_df["evaluation_active"]
        & detector_df["uncertainty_value"].notna()
    ].sort_values("uncertainty_value", ascending=False)
    if scored.empty:
        return pd.DataFrame(columns=["abstention_rate", "accuracy"])

    total = len(scored)
    rows = []
    for abstained in range(total + 1):
        retained = scored.iloc[abstained:]
        rows.append(
            {
                "abstention_rate": abstained / total,
                "accuracy": retained["pred_correct"].mean() if not retained.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_detector_metrics(
    detector_df: pd.DataFrame,
    *,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    if detector_df.empty:
        return pd.DataFrame(
            columns=[
                "effort",
                "detector_rows",
                "detection_accuracy",
                "constrained_parse_fail_rate",
                "calibration_auc",
                "selective_auc",
            ]
        )

    rows = []
    for effort_name, effort_df in detector_df.groupby("effort", as_index=False):
        active_df = effort_df[effort_df["evaluation_active"]].copy()
        if active_df.empty:
            continue

        accuracy_curve = detector_accuracy_curve(active_df)
        selective_curve = detector_selective_curve(active_df)

        calibration_auc = _curve_auc(
            accuracy_curve["threshold"].tolist(),
            [0.0 if math.isnan(value) else value for value in accuracy_curve["accuracy"].tolist()],
        )
        selective_auc = _curve_auc(
            selective_curve["abstention_rate"].tolist(),
            [0.0 if math.isnan(value) else value for value in selective_curve["accuracy"].tolist()],
        )

        parse_fail_rate = (
            active_df["constrained_parse_fail"].mean() * 100.0
            if "constrained_parse_fail" in active_df.columns
            else float("nan")
        )

        rows.append(
            {
                "effort": effort_name,
                "detector_rows": len(active_df),
                "detection_accuracy": active_df["pred_correct"].mean() * 100.0 if not active_df.empty else float("nan"),
                "constrained_parse_fail_rate": parse_fail_rate,
                "calibration_auc": calibration_auc,
                "selective_auc": selective_auc,
            }
        )

    return pd.DataFrame(rows)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )


def _mean_optional_bool(series: pd.Series) -> float:
    values = []
    for value in series:
        if pd.isna(value):
            continue
        if isinstance(value, bool):
            values.append(value)
            continue
        lowered = str(value).strip().lower()
        if lowered == "true":
            values.append(True)
        elif lowered == "false":
            values.append(False)
    if not values:
        return float("nan")
    return (sum(values) / len(values)) * 100.0


def _count_non_null(series: pd.Series) -> int:
    return int(series.notna().sum())


def build_baseline_rsn_summary(
    *,
    error_taxonomy_all_path: str,
    error_taxonomy_full_path: str,
) -> pd.DataFrame:
    taxonomy_all = pd.read_csv(error_taxonomy_all_path)
    taxonomy_full = pd.read_csv(error_taxonomy_full_path)

    taxonomy_all = taxonomy_all[taxonomy_all["taxonomy_code"].astype(str).str.startswith("RSN")].copy()
    taxonomy_full = taxonomy_full[taxonomy_full["taxonomy_code"].astype(str).str.startswith("RSN")].copy()

    loc_series = _coerce_bool_series(taxonomy_all["Loc"])
    exp_series = _coerce_bool_series(taxonomy_all["Exp"])
    coplan_series = _coerce_bool_series(taxonomy_full["c_success"])

    return pd.DataFrame(
        [
            {
                "effort": "baseline_run4",
                "episodes": len(taxonomy_full),
                "status_ok_rate": 100.0,
                "loc_rate": loc_series.mean() * 100,
                "coplan_rate": coplan_series.mean() * 100,
                "exp_rate": exp_series.mean() * 100,
                "exp_scored_count": int(exp_series.notna().sum()),
            }
        ]
    )


def build_exp_annotation_table(
    metrics_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    exp_table = (
        metrics_df[
            [
                "effort",
                "task",
                "episode",
                "status",
                "gt_error_type",
                "gt_failure_reason",
                "pred_failure_reason",
                "pipeline_error",
            ]
        ]
        .drop_duplicates(subset=["effort", "task", "episode"])
        .sort_values(["effort", "task", "episode"])
        .reset_index(drop=True)
    )
    exp_table["human_score"] = pd.NA

    if output_path and os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        keep_cols = [col for col in ["effort", "task", "episode", "human_score"] if col in existing.columns]
        if keep_cols:
            exp_table = exp_table.drop(columns=["human_score"]).merge(
                existing[keep_cols],
                on=["effort", "task", "episode"],
                how="left",
            )
        if "human_score" not in exp_table.columns:
            exp_table["human_score"] = pd.NA

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        exp_table.to_csv(output_path, index=False)

    return exp_table


def summarize_metric_scaling(
    metrics_df: pd.DataFrame,
    exp_scores_df: Optional[pd.DataFrame] = None,
    baseline_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if metrics_df.empty:
        summary = pd.DataFrame(columns=["effort", "episodes", "status_ok_rate", "loc_rate", "coplan_rate"])
    else:
        summary = (
            metrics_df.groupby("effort", as_index=False)
            .agg(
                episodes=("episode", "count"),
                status_ok_rate=("status_ok", lambda s: s.mean() * 100),
                replay_available_rate=("replay_available", lambda s: s.mean() * 100),
                loc_rate=("loc_hit", lambda s: s.mean() * 100),
                coplan_rate=("coplan_success", _mean_optional_bool),
                coplan_scored_count=("coplan_success", _count_non_null),
            )
            .reset_index(drop=True)
        )

    if exp_scores_df is not None and not exp_scores_df.empty:
        exp_scores_df = exp_scores_df.copy()
        exp_scores_df = exp_scores_df.dropna(subset=["human_score"])
        if not exp_scores_df.empty:
            exp_scores_df["human_score"] = exp_scores_df["human_score"].astype(float)
            exp_summary = (
                exp_scores_df.groupby("effort", as_index=False)
                .agg(
                    exp_rate=("human_score", lambda s: s.mean() * 100),
                    exp_scored_count=("human_score", "count"),
                )
            )
            summary = summary.merge(exp_summary, on="effort", how="left")

    if "exp_rate" not in summary.columns:
        summary["exp_rate"] = float("nan")
    if "exp_scored_count" not in summary.columns:
        summary["exp_scored_count"] = 0
    if "coplan_scored_count" not in summary.columns:
        summary["coplan_scored_count"] = 0

    if baseline_summary_df is not None and not baseline_summary_df.empty:
        summary = pd.concat([baseline_summary_df, summary], ignore_index=True, sort=False)

    summary["_effort_order"] = summary["effort"].map(lambda effort: EFFORT_ORDER.get(str(effort), 999))
    summary = summary.sort_values(["_effort_order", "effort"]).drop(columns=["_effort_order"]).reset_index(drop=True)

    return summary


def metric_long_format(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "loc_rate": "Loc",
        "coplan_rate": "Co-plan",
        "exp_rate": "Exp",
    }
    melted = summary_df.melt(
        id_vars=[col for col in summary_df.columns if col not in metric_map],
        value_vars=list(metric_map.keys()),
        var_name="metric_key",
        value_name="score",
    )
    melted["metric"] = melted["metric_key"].map(metric_map)
    return melted.drop(columns=["metric_key"])
