"""
DARE-Bench offline evaluation — aligned with scripts/utils.py (datasci path).

Ground-truth file selection (same rules as utils.py when building each session's ``ground_truth`` path):

- **classification / regression + v1:** ``verify/simulated_pred_local.csv`` (reference executor
  output used as labels for IF-style scoring; generate via reference solution / simulation if missing).
- **classification / regression + v2:** ``verify/ground_truth.csv``.
- **time_series_analysis + v1:** ``verify/ground_truth_v1.csv``.
- **time_series_analysis + v2:** ``verify/ground_truth_v2.csv``.

Row alignment (same as _datasci_alignment_id_columns + merge in utils.py):
  - time_series_analysis + v2: join on all non-target columns in the GT file (e.g. Date, ts, or
    composite keys like date+states). Predictions must include those same columns; there is no row_id.
  - If ground_truth contains row_id (typical tabular v1): align on row_id.
  - Otherwise: non-target columns in GT.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score


def _datasci_alignment_id_columns(
    ground_truth_pd: pd.DataFrame,
    target_columns: List[str],
    task: str,
    ds_question_version: str,
) -> List[str]:
    """Aligned with utils._datasci_alignment_id_columns (v1/v2)."""
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    tset = set(target_columns)
    if task == "time_series_analysis" and ds_question_version == "v2":
        return [c for c in ground_truth_pd.columns if c not in tset]
    if "row_id" in ground_truth_pd.columns:
        return ["row_id"]
    return [c for c in ground_truth_pd.columns if c not in tset]


def get_datasci_metric_for_single_column(
    pred_col: List[Any],
    gt_col: List[Any],
    task: str,
    ds_question_version: str,
) -> float:
    """Aligned with utils.get_datasci_metric_for_single_column for v1/v2."""
    if len(pred_col) != len(gt_col):
        return 0.0

    if task == "classification":
        if ds_question_version == "v1":
            return float(gt_col == pred_col)
        if ds_question_version == "v2":
            try:
                return float(f1_score(gt_col, pred_col, average="macro"))
            except Exception:
                return 0.0
        raise NotImplementedError(f"Unsupported ds_question_version: {ds_question_version}")

    if task == "regression":
        if ds_question_version == "v1":
            return float(np.allclose(np.array(gt_col), np.array(pred_col)))
        if ds_question_version == "v2":
            try:
                return float(np.clip(r2_score(gt_col, pred_col), 0, 1))
            except Exception:
                return 0.0
        raise NotImplementedError(f"Unsupported ds_question_version: {ds_question_version}")

    if task == "time_series_analysis":
        try:
            return float(np.clip(r2_score(gt_col, pred_col), 0, 1))
        except Exception:
            return 0.0

    raise NotImplementedError(f"Evaluation for task {task} not implemented")


# Backward-compatible alias
get_metric_for_single_column = get_datasci_metric_for_single_column


def _resolve_ground_truth_path(
    verify_dir: str, ds_question_version: str, task: str
) -> Optional[str]:
    """
    Match scripts/utils.py (datasci ``process``) ground_truth_path selection.

    Tabular v1 uses simulated_pred_local.csv; tabular v2 uses ground_truth.csv.
    Time series v1 uses ground_truth_v1.csv; time series v2 uses ground_truth_v2.csv.
    """
    v = ds_question_version
    candidates: List[str]

    if task == "time_series_analysis":
        if v == "v1":
            candidates = [os.path.join(verify_dir, "ground_truth_v1.csv")]
        elif v == "v2":
            candidates = [os.path.join(verify_dir, "ground_truth_v2.csv")]
        else:
            candidates = [
                os.path.join(verify_dir, f"ground_truth_{v}.csv"),
                os.path.join(verify_dir, "ground_truth.csv"),
            ]
    elif task in ("classification", "regression"):
        if v == "v1":
            candidates = [os.path.join(verify_dir, "simulated_pred_local.csv")]
        elif v == "v2":
            candidates = [os.path.join(verify_dir, "ground_truth.csv")]
        else:
            candidates = [
                os.path.join(verify_dir, f"ground_truth_{v}.csv"),
                os.path.join(verify_dir, "ground_truth.csv"),
            ]
    else:
        candidates = [
            os.path.join(verify_dir, f"ground_truth_{v}.csv"),
            os.path.join(verify_dir, "ground_truth.csv"),
        ]

    return next((p for p in candidates if os.path.exists(p)), None)


def evaluate_prediction(
    prediction_path: str,
    ground_truth_path: str,
    metadata_path: str,
    ds_question_version: str = "v2",
    *,
    question_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate prediction.csv vs ground truth using the same merge logic as utils.get_final_result_for_eval.

    For time series v2, ``prediction.csv`` must include the same identifier columns as
    ``ground_truth_v2.csv`` (all columns except targets), not necessarily ``row_id``.
    """
    if question_version is not None:
        ds_question_version = question_version

    if not os.path.exists(prediction_path):
        return {"prediction_exist": 0.0, "final_score": 0.0}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    target_columns = metadata["question"]["target"]
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    task = metadata["question"]["problem_type"]

    ground_truth_pd = pd.read_csv(ground_truth_path)
    prediction = pd.read_csv(prediction_path)

    id_columns = _datasci_alignment_id_columns(
        ground_truth_pd, target_columns, task, ds_question_version
    )
    if not id_columns:
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "empty_id_columns_after_resolving_alignment",
            "id_columns": [],
        }
    if not all(c in prediction.columns for c in id_columns):
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "prediction_missing_id_columns",
            "id_columns": id_columns,
        }

    pred_cols = id_columns + [c for c in target_columns if c in prediction.columns]
    merged = ground_truth_pd.merge(
        prediction[pred_cols],
        on=id_columns,
        how="inner",
        suffixes=("_gt", "_pred"),
    )
    if len(merged) != len(ground_truth_pd):
        return {
            "prediction_exist": 1.0,
            "final_score": 0.0,
            "error": "row_count_mismatch_after_merge",
            "id_columns": id_columns,
            "ground_truth_rows": len(ground_truth_pd),
            "merged_rows": len(merged),
        }

    per_target_scores: Dict[str, float] = {}
    for col in target_columns:
        if col not in prediction.columns:
            per_target_scores[col] = 0.0
        else:
            gt_c, pr_c = f"{col}_gt", f"{col}_pred"
            per_target_scores[col] = get_datasci_metric_for_single_column(
                merged[pr_c].tolist(),
                merged[gt_c].tolist(),
                task,
                ds_question_version,
            )

    final_score = float(np.mean(list(per_target_scores.values())))
    return {
        "prediction_exist": 1.0,
        "final_score": final_score,
        "per_target_scores": per_target_scores,
        "id_columns": id_columns,
    }


def evaluate_batch(
    predictions_dir: str,
    databases_dir: str,
    question_list_path: str,
    ds_question_version: str = "v2",
    *,
    question_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    For each task, load ``prediction.csv`` and resolve the label file per task type and version
    (see module docstring: ``simulated_pred_local.csv`` vs ``ground_truth.csv`` vs ``ground_truth_v1/v2``).
    """
    if question_version is not None:
        ds_question_version = question_version

    with open(question_list_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results: Dict[str, Any] = {}
    for q in questions:
        folder_name = q["file_path"]
        prediction_path = os.path.join(predictions_dir, folder_name, "prediction.csv")
        verify_dir = os.path.join(databases_dir, folder_name, "verify")
        metadata_path = os.path.join(verify_dir, "all_metadata.json")

        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata not found for {folder_name}")
            continue

        task = q.get("task")
        if not task:
            with open(metadata_path, "r", encoding="utf-8") as mf:
                task = json.load(mf)["question"]["problem_type"]

        ground_truth_path = _resolve_ground_truth_path(verify_dir, ds_question_version, task)

        if not ground_truth_path:
            print(
                f"Warning: No expected label file for {folder_name} "
                f"(task={task}, version={ds_question_version}); "
                "see evaluation.py docstring (e.g. simulated_pred_local.csv for tabular v1)."
            )
            continue

        results[folder_name] = evaluate_prediction(
            prediction_path=prediction_path,
            ground_truth_path=ground_truth_path,
            metadata_path=metadata_path,
            ds_question_version=ds_question_version,
        )

    prediction_rates = [r["prediction_exist"] for r in results.values()]
    scores = [r["final_score"] for r in results.values() if r["prediction_exist"] > 0]

    return {
        "total_tasks": len(results),
        "prediction_rate": float(np.mean(prediction_rates)) if prediction_rates else 0.0,
        "average_score": float(np.mean(scores)) if scores else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DARE-Bench predictions (aligned with scripts/utils.py)"
    )
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--databases_dir", type=str, required=True)
    parser.add_argument("--question_list", type=str, required=True)
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help=(
            "Question version (ds_question_version). GT file: tabular v1 → "
            "simulated_pred_local.csv, tabular v2 → ground_truth.csv; "
            "TS v1 → ground_truth_v1.csv, TS v2 → ground_truth_v2.csv."
        ),
    )
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    out = evaluate_batch(
        predictions_dir=args.predictions_dir,
        databases_dir=args.databases_dir,
        question_list_path=args.question_list,
        ds_question_version=args.version,
    )

    print(f"\n{'=' * 60}")
    print("DARE-Bench Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Total tasks: {out['total_tasks']}")
    print(f"Prediction rate: {out['prediction_rate'] * 100:.2f}%")
    print(f"Average score: {out['average_score'] * 100:.2f}%")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.output}")
