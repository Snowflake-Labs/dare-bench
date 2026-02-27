"""
DARE-Bench Evaluation Script

This script provides functionality to evaluate model predictions against ground truth.
It supports classification, regression, and time series tasks.

Metrics:
- Classification-IF (v1): Exact match accuracy
- Classification_MM (v2): Macro F1 score  
- Regression-IF (v1): Exact match accuracy
- Regression-MM (v2): Clipped R² score (0-1)
- Time Series-XF/CF (v1/v2): Clipped R² score (0-1)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score
from pathlib import Path
from typing import Dict, List, Union, Optional


def get_metric_for_single_column(
    pred_col: List, 
    gt_col: List, 
    task: str, 
    question_version: str = "v2"
) -> float:
    """
    Calculate metric for a single target column.
    
    Args:
        pred_col: List of predictions
        gt_col: List of ground truth values
        task: One of "classification", "regression", "time_series_analysis"
        question_version: "v1" or "v2" (affects metric calculation)
    
    Returns:
        Score between 0 and 1
    """
    # Handle length mismatch
    if len(pred_col) != len(gt_col):
        return 0.0

    if task == "classification":
        if question_version in ["v1", "v3"]:
            # Exact match
            return float(gt_col == pred_col)
        elif question_version == "v2":
            try:
                # Macro F1 score
                reward = f1_score(gt_col, pred_col, average="macro")
                return reward
            except:
                return 0.0
        else:
            raise NotImplementedError(f"Unsupported question_version: {question_version}")

    elif task == "regression":
        if question_version in ["v1", "v3"]:
            # Exact match with tolerance
            return float(np.allclose(np.array(gt_col), np.array(pred_col)))
        elif question_version == "v2":
            try:
                # Clipped R² score (0 is worst, 1 is best)
                reward = np.clip(r2_score(gt_col, pred_col), 0, 1)
                return reward
            except:
                return 0.0
        else:
            raise NotImplementedError(f"Unsupported question_version: {question_version}")
            
    elif task == "time_series_analysis":
        try:
            # Clipped R² score
            reward = np.clip(r2_score(gt_col, pred_col), 0, 1)
            return reward
        except:
            return 0.0
    else:
        raise NotImplementedError(f"Evaluation for task {task} not implemented")


def evaluate_prediction(
    prediction_path: str,
    ground_truth_path: str,
    metadata_path: str,
    question_version: str = "v2"
) -> Dict[str, float]:
    """
    Evaluate a prediction file against ground truth.
    
    Args:
        prediction_path: Path to prediction.csv
        ground_truth_path: Path to ground_truth.csv
        metadata_path: Path to all_metadata.json
        question_version: "v1" or "v2"
    
    Returns:
        Dictionary with "prediction_exist" and "final_score"
    """
    # Check if prediction file exists
    if not os.path.exists(prediction_path):
        return {
            "prediction_exist": 0.0,
            "final_score": 0.0
        }
    
    # Load metadata to get target columns and task type
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    target_columns = metadata["question"]["target"]
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    
    task = metadata["question"]["problem_type"]
    
    # Load ground truth
    ground_truth_pd = pd.read_csv(ground_truth_path)
    ground_truth_pd.sort_values(by=["row_id"], inplace=True)
    ground_truth = {}
    for col in target_columns:
        ground_truth[col] = ground_truth_pd[col].tolist()
    
    # Load prediction
    prediction = pd.read_csv(prediction_path)
    
    # Check row_id alignment
    if "row_id" not in prediction.columns:
        return {"prediction_exist": 1.0, "final_score": 0.0}
    
    prediction.sort_values(by=["row_id"], inplace=True)
    if prediction["row_id"].tolist() != ground_truth_pd["row_id"].tolist():
        return {"prediction_exist": 1.0, "final_score": 0.0}
    
    # Calculate score for each target column
    per_target_scores = {}
    for col in target_columns:
        if col not in prediction.columns:
            per_target_scores[col] = 0.0  # Model didn't predict this column
        else:
            pred_col = prediction[col].tolist()
            gt_col = ground_truth[col]
            per_target_scores[col] = get_metric_for_single_column(
                pred_col, gt_col, task, question_version
            )
    
    # Return average of all target columns
    final_score = np.mean(list(per_target_scores.values()))
    return {
        "prediction_exist": 1.0,
        "final_score": final_score,
        "per_target_scores": per_target_scores
    }


def evaluate_batch(
    predictions_dir: str,
    databases_dir: str,
    question_list_path: str,
    question_version: str = "v2"
) -> Dict[str, Dict]:
    """
    Evaluate predictions for multiple tasks.
    
    Args:
        predictions_dir: Directory containing prediction.csv files (one per task folder)
        databases_dir: Directory containing database folders with ground_truth.csv
        question_list_path: Path to question_list.json
        question_version: "v1" or "v2"
    
    Returns:
        Dictionary mapping task folder names to evaluation results
    """
    with open(question_list_path, "r") as f:
        questions = json.load(f)
    
    results = {}
    for q in questions:
        folder_name = q["file_path"]
        
        prediction_path = os.path.join(predictions_dir, folder_name, "prediction.csv")
        ground_truth_path = os.path.join(databases_dir, folder_name, "verify", "ground_truth.csv")
        metadata_path = os.path.join(databases_dir, folder_name, "verify", "all_metadata.json")
        
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata not found for {folder_name}")
            continue
        
        result = evaluate_prediction(
            prediction_path=prediction_path,
            ground_truth_path=ground_truth_path,
            metadata_path=metadata_path,
            question_version=question_version
        )
        results[folder_name] = result
    
    # Calculate summary statistics
    prediction_rates = [r["prediction_exist"] for r in results.values()]
    scores = [r["final_score"] for r in results.values() if r["prediction_exist"] > 0]
    
    summary = {
        "total_tasks": len(results),
        "prediction_rate": np.mean(prediction_rates) if prediction_rates else 0.0,
        "average_score": np.mean(scores) if scores else 0.0,
        "results": results
    }
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate DARE-Bench predictions")
    parser.add_argument("--predictions_dir", type=str, required=True,
                        help="Directory containing prediction folders")
    parser.add_argument("--databases_dir", type=str, required=True,
                        help="Directory containing database folders")
    parser.add_argument("--question_list", type=str, required=True,
                        help="Path to question_list.json")
    parser.add_argument("--version", type=str, default="v2", choices=["v1", "v2"],
                        help="Question version (v1 or v2)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    
    args = parser.parse_args()
    
    results = evaluate_batch(
        predictions_dir=args.predictions_dir,
        databases_dir=args.databases_dir,
        question_list_path=args.question_list,
        question_version=args.version
    )
    
    print(f"\n{'='*60}")
    print(f"DARE-Bench Evaluation Results")
    print(f"{'='*60}")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Prediction rate: {results['prediction_rate']*100:.2f}%")
    print(f"Average score: {results['average_score']*100:.2f}%")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

