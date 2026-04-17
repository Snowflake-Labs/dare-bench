"""
DARE-Bench Reference Solution Generator

This script generates reference solutions for DARE-Bench tasks based on metadata.
The reference solution uses scikit-learn models with parameters specified in all_metadata.json.

The generated solution:
1. Loads train/validation data (CSV, SQLite, or Parquet)
2. Applies preprocessing (imputation, scaling, one-hot encoding)
3. Trains a specified scikit-learn model
4. Writes predictions to simulated_pred_local.csv (aligned with scripts/utils.py)

Optional execution (--execute): run the generated code either locally (cwd=source/, then move CSV to verify/)
or via HTTP sandbox (--use-sandbox), mirroring call_code_executor / get_simulate_results in utils.py.

Only ``problem_type`` **classification** and **regression** are supported. **time_series_analysis** (and any
other type) is skipped — this template matches tabular sklearn pipelines, not TS forecasting.
"""

import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from string import Template
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

OUTPUT_CSV_NAME = "simulated_pred_local.csv"

# Tabular ML only; time series uses a different benchmark setup
REFERENCE_SOLUTION_PROBLEM_TYPES = frozenset({"classification", "regression"})


def is_reference_solution_applicable(problem_type: str) -> bool:
    return problem_type.strip().lower() in REFERENCE_SOLUTION_PROBLEM_TYPES


def call_code_executor(
    data_path: str,
    code: str,
    url: str,
    save_file_type: str,
    timeout: int = 100,
) -> Dict[str, Any]:
    """
    Run generated code on remote sandbox; write fetch_files into verify/.
    Same contract as scripts/utils.py call_code_executor.
    """
    if requests is None:
        raise ImportError("requests is required for sandbox execution: pip install requests")

    encoded_files: Dict[str, str] = {}
    if save_file_type == "sqlite":
        files_to_load = ["train_v1_no_err.sqlite", "val_v1.sqlite"]
    elif save_file_type == "csv":
        files_to_load = ["train_v1_no_err.csv", "val_v1.csv"]
    elif save_file_type == "parquet":
        files_to_load = ["train_v1_no_err.parquet", "val_v1.parquet"]
    else:
        raise ValueError(f"Invalid save_file_type: {save_file_type}")

    for file_to_load in files_to_load:
        with open(os.path.join(data_path, "source", file_to_load), "rb") as f:
            content = f.read()
        encoded_files[file_to_load] = base64.b64encode(content).decode("utf-8")

    response = requests.post(
        url,
        json={
            "code": code,
            "language": "python",
            "files": encoded_files,
            "fetch_files": [OUTPUT_CSV_NAME],
            "compile_timeout": timeout,
            "run_timeout": timeout,
        },
        timeout=timeout + 60,
    )
    response.raise_for_status()
    payload = response.json()

    run_result = payload["run_result"]
    status = run_result.get("status")
    if status == "TimeLimitExceeded":
        raise TimeoutError(
            f"Code execution exceeded time limit ({run_result.get('execution_time')}s for {data_path})"
        )

    if OUTPUT_CSV_NAME not in payload.get("files", {}):
        raise ValueError(f"{OUTPUT_CSV_NAME} not returned by sandbox for {data_path}")

    verify_dir = os.path.join(data_path, "verify")
    os.makedirs(verify_dir, exist_ok=True)
    for file_name, file_content in payload["files"].items():
        with open(os.path.join(verify_dir, file_name), "wb") as f:
            f.write(base64.b64decode(file_content))

    return run_result


def run_reference_solution_local(database_path: str, code: str, timeout: int = 300) -> None:
    """Execute generated code with cwd=source/; move simulated_pred_local.csv to verify/."""
    source_dir = os.path.join(database_path, "source")
    verify_dir = os.path.join(database_path, "verify")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Missing source directory: {source_dir}")
    os.makedirs(verify_dir, exist_ok=True)

    script_path = os.path.join(source_dir, "_reference_solution_run.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=source_dir,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Local reference run timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise RuntimeError(f"Reference solution process exited with code {proc.returncode}")

    out_src = os.path.join(source_dir, OUTPUT_CSV_NAME)
    out_dst = os.path.join(verify_dir, OUTPUT_CSV_NAME)
    if not os.path.isfile(out_src):
        raise FileNotFoundError(f"Expected {out_src} after local run")

    shutil.move(out_src, out_dst)
    try:
        os.remove(script_path)
    except OSError:
        pass
    print(f"✓ {OUTPUT_CSV_NAME} saved to {out_dst}")


# Template for generating reference solution code
REFERENCE_SOLUTION_TEMPLATE = '''"""
Reference Solution for DARE-Bench Task
Generated from all_metadata.json parameters
"""

import json
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


def load_and_join(sqlite_path):
    """Load and join tables from a SQLite file on row_id."""
    conn = sqlite3.connect(sqlite_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    df = None
    for table in tables:
        df_tab = pd.read_sql_query(f"SELECT * FROM '{table}'", conn)
        if 'row_id' not in df_tab.columns:
            continue
        if df is None:
            df = df_tab
        else:
            df = df.merge(df_tab, on='row_id', how='inner')
    conn.close()
    return df


def train_predict_model(train_df, eval_df, feature_cols, model_type, column_type_inference, 
                        target_cols=['answer'], imputer_type="most_frequent", 
                        problem_type="classification", random_state=42):
    """Train a model and make predictions."""
    print(f"🚀 MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    # Check if target column exists in training data
    for target_column in target_cols:
        if target_column not in train_df.columns:
            print(f"❌ Target column '{target_column}' not found in training data!")
            print(f"Available columns in train_df: {list(train_df.columns)}")
            return None
    
    # Check if all feature columns exist in both datasets
    missing_features_train = [col for col in feature_cols if col not in train_df.columns]
    missing_features_eval = [col for col in feature_cols if col not in eval_df.columns]
    
    if missing_features_train:
        print(f"❌ Missing feature columns in train_df: {missing_features_train}")
        return None
    
    if missing_features_eval:
        print(f"❌ Missing feature columns in eval_df: {missing_features_eval}")
        return None
    
    formatted_targets = ", ".join("`{}`".format(col) for col in target_cols)
    print(f"✓ Target column {formatted_targets} found in training data")
    print(f"✓ Training dataset shape: {train_df.shape}")
    print(f"✓ Evaluation dataset shape: {eval_df.shape}")
    
    # Prepare training features and target
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_cols].copy()
    
    # Prepare evaluation features
    X_eval = eval_df[feature_cols].copy()
    
    # Check for null targets in training data
    null_targets_train = y_train.isnull().sum().sum()
    if null_targets_train > 0:
        print(f" Found {null_targets_train} null targets in training data - removing these rows")
        valid_indices = ~y_train.isnull().any(axis=1)  # make sure no null target row
        X_train = X_train[valid_indices]
        y_train = y_train[valid_indices]

    print(f"✓ Final training data: {X_train.shape[0]} rows")
    print(f"✓ Final evaluation data: {X_eval.shape[0]} rows")
    
    # Separate numeric and categorical features
    numeric_features = []
    categorical_features = []
    
    for col in feature_cols:
        if column_type_inference[col].lower() == "numerical":
            numeric_features.append(col)
        elif column_type_inference[col].lower() == "categorical":
            categorical_features.append(col)
    
    print(f"✓ Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"✓ Categorical features ({len(categorical_features)}): {categorical_features}")
    assert len(numeric_features) + len(categorical_features) == len(feature_cols)
    
    # Create preprocessing pipeline
    transformers = []
    
    if numeric_features:
        transformers.append(('num', Pipeline([
            ('imputer', SimpleImputer(strategy=imputer_type)),
            ('scaler', StandardScaler())
        ]), numeric_features))
    
    if categorical_features:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Choose model based on problem type
    if problem_type.lower() == "classification":
        if model_type == "LogisticRegression":
            model = LogisticRegression(random_state=random_state)
            if len(target_cols) > 1:
                model = MultiOutputClassifier(model)
        elif model_type == "DecisionTreeClassifier":
            model = DecisionTreeClassifier(random_state=random_state)
        elif model_type == "GaussianNB":
            model = GaussianNB()
            if len(target_cols) > 1:
                model = MultiOutputClassifier(model)
        elif model_type == "LinearSVC":
            model = LinearSVC(random_state=random_state)
            if len(target_cols) > 1:
                model = MultiOutputClassifier(model)
        elif model_type == "MLPClassifier":
            model = MLPClassifier(random_state=random_state)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        print(f"✓ Using {model_type} for classification")
        
    elif problem_type.lower() == "regression":
        if model_type == "LinearRegression":
            model = LinearRegression()
        elif model_type == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(random_state=random_state)
        elif model_type == "Ridge":
            model = Ridge(random_state=random_state)
        elif model_type == "Lasso":
            model = Lasso(random_state=random_state)
        elif model_type == "MLPRegressor":
            model = MLPRegressor(random_state=random_state)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        print(f"✓ Using {model_type} for regression")
    
    # Create full pipeline
    ml_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    print(f"📊 TRAINING MODEL...")
    try:
        ml_pipeline.fit(X_train, y_train)
        print(f"✓ Model trained successfully")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None
    
    # Make predictions
    print(f"🔮 MAKING PREDICTIONS...")
    try:
        y_pred = ml_pipeline.predict(X_eval)
        print(f"✓ Predictions completed")
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None
    
    y_pred_df = pd.DataFrame(y_pred, columns=y_train.columns)
    y_pred_df.insert(0, 'row_id', eval_df["row_id"].values)
    return {
        'pipeline': ml_pipeline,
        'predictions': y_pred_df,
        'problem_type': problem_type,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


# Parameters from all_metadata.json
feature_cols = $feature_cols
model_type = "$model_type"
column_type_inference = $column_type_inference
target_cols = $target_cols
imputer_type = "$imputer_type"
problem_type = "$problem_type"
random_state = $random_state
save_file_type = "$save_file_type"

# Load data based on file type
if save_file_type == 'sqlite':
    conn = sqlite3.connect("train_v1_no_err.sqlite")
    train_df = pd.read_sql("SELECT * FROM train_set", conn)
    train_df = train_df.replace({None: np.nan})
    conn.close()

    eval_df = load_and_join("val_v1.sqlite")
    eval_df = eval_df.replace({None: np.nan})
elif save_file_type == 'csv':
    train_df = pd.read_csv('train_v1_no_err.csv', keep_default_na=False, na_values=[""])
    eval_df = pd.read_csv('val_v1.csv', keep_default_na=False, na_values=[""])
elif save_file_type == 'parquet':
    train_df = pd.read_parquet('train_v1_no_err.parquet')
    train_df = train_df.replace({None: np.nan})
    eval_df = pd.read_parquet('val_v1.parquet')
    eval_df = eval_df.replace({None: np.nan})

# Train and predict
result = train_predict_model(
    train_df=train_df,
    eval_df=eval_df,
    feature_cols=feature_cols,
    model_type=model_type,
    column_type_inference=column_type_inference,
    target_cols=target_cols,
    imputer_type=imputer_type,
    problem_type=problem_type,
    random_state=random_state
)

# Save predictions (simulated_pred_local.csv — same name as utils.call_code_executor fetch_files)
result['predictions'].to_csv('simulated_pred_local.csv', index=False)
print("✓ Predictions saved to simulated_pred_local.csv")
'''


def generate_reference_solution_code(metadata_path: str) -> str:
    """
    Generate reference solution Python code from metadata.

    Only supports classification and regression (tabular). Raises ValueError for
    time_series_analysis and other problem types.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    question = all_metadata["question"]
    problem_type = question["problem_type"]
    if not is_reference_solution_applicable(problem_type):
        raise ValueError(
            f"Reference solution is only generated for classification/regression; "
            f"got problem_type={problem_type!r}"
        )

    feature_cols = question["selected_features"]
    model_type = question["model_type"]
    column_type_inference = question["column_type_inference"]
    target_cols = question["target"]
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    imputer_type = question["imputer_type"]
    random_state = question["random_state"]
    save_file_type = question["save_file_type"]
    
    code = Template(REFERENCE_SOLUTION_TEMPLATE).substitute(
        feature_cols=feature_cols,
        model_type=model_type,
        column_type_inference=column_type_inference,
        target_cols=target_cols,
        imputer_type=imputer_type,
        problem_type=problem_type,
        random_state=random_state,
        save_file_type=save_file_type
    )
    
    return code


def generate_reference_solution_for_task(
    database_path: str,
    output_path: Optional[str] = None,
    save_code: bool = False,
    execute: bool = False,
    use_sandbox: bool = False,
    sandbox_url: str = "http://localhost:8080/run_code",
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Generate and optionally execute reference solution for a single task.

    Args:
        database_path: Path to the task database folder (containing source/ and verify/)
        output_path: Where to save generated reference_solution.py when save_code=True
        save_code: Whether to save the generated code to reference_solution.py
        execute: If True, run the generated code (see use_sandbox)
        use_sandbox: If True, upload source files and run via HTTP (same as utils.call_code_executor).
            If False, run locally with cwd=source/, then move simulated_pred_local.csv to verify/.
        sandbox_url: Executor URL when use_sandbox=True
        timeout: Sandbox or local subprocess is expected to finish within this many seconds (sandbox uses it for compile/run_timeout)

    Returns:
        Dictionary with solution code, optional code_path, optional verify_output path,
        or ``skipped: True`` for time_series_analysis / unsupported problem_type.
    """
    metadata_path = os.path.join(database_path, "verify", "all_metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)
    problem_type = all_metadata["question"]["problem_type"]
    if not is_reference_solution_applicable(problem_type):
        print(f"⏭ Skipping (reference template is tabular only): {problem_type!r} — {database_path}")
        return {"skipped": True, "problem_type": problem_type, "code": None}

    save_file_type = all_metadata["question"]["save_file_type"]

    code = generate_reference_solution_code(metadata_path)

    result: Dict[str, Any] = {"skipped": False, "code": code}

    if save_code:
        code_path = os.path.join(output_path or database_path, "reference_solution.py")
        os.makedirs(os.path.dirname(code_path) or ".", exist_ok=True)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        result["code_path"] = code_path
        print(f"✓ Reference solution code saved to: {code_path}")

    if execute:
        if use_sandbox:
            print(f"Running reference solution on sandbox: {sandbox_url}")
            call_code_executor(database_path, code, sandbox_url, save_file_type, timeout=timeout)
            result["simulated_pred_path"] = os.path.join(database_path, "verify", OUTPUT_CSV_NAME)
            result["execution_mode"] = "sandbox"
        else:
            print("Running reference solution locally (cwd=source/)")
            run_reference_solution_local(database_path, code, timeout=timeout)
            result["simulated_pred_path"] = os.path.join(database_path, "verify", OUTPUT_CSV_NAME)
            result["execution_mode"] = "local"

    return result


def batch_generate_reference_solutions(
    databases_dir: str,
    question_list_path: str,
    output_dir: str,
    execute: bool = False,
    use_sandbox: bool = False,
    sandbox_url: str = "http://localhost:8080/run_code",
    timeout: int = 300,
) -> None:
    """
    Generate reference solution code for all tasks in question_list.json.

    Args:
        databases_dir: Directory containing database folders
        question_list_path: Path to question_list.json
        output_dir: Directory to save generated solution code
        execute, use_sandbox, sandbox_url, timeout: Passed to generate_reference_solution_for_task
    """
    with open(question_list_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for q in questions:
        folder_name = q["file_path"]
        database_path = os.path.join(databases_dir, folder_name)

        try:
            task_output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(task_output_dir, exist_ok=True)

            out = generate_reference_solution_for_task(
                database_path=database_path,
                output_path=task_output_dir,
                save_code=True,
                execute=execute,
                use_sandbox=use_sandbox,
                sandbox_url=sandbox_url,
                timeout=timeout,
            )
            if out.get("skipped"):
                skipped += 1
            else:
                success += 1
        except Exception as e:
            print(f"❌ Failed for {folder_name}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Reference Solution Generation Complete")
    print(f"{'='*60}")
    print(f"Success: {success}")
    print(f"Skipped (non-tabular): {skipped}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse

    default_url = os.environ.get("PYTHON_EXECUTOR_URL", "http://localhost:8080/run_code")

    parser = argparse.ArgumentParser(description="Generate DARE-Bench reference solutions")
    parser.add_argument("--database", type=str, default=None,
                        help="Path to a single database folder")
    parser.add_argument("--databases_dir", type=str, default=None,
                        help="Directory containing database folders (for batch mode)")
    parser.add_argument("--question_list", type=str, default=None,
                        help="Path to question_list.json (for batch mode)")
    parser.add_argument("--output_dir", type=str, default="./reference_solutions",
                        help="Output directory for generated solutions")
    parser.add_argument("--save_code", action="store_true",
                        help="Save generated code to file")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run generated code (local: cwd=source; sandbox: HTTP)",
    )
    parser.add_argument(
        "--use-sandbox",
        action="store_true",
        help="With --execute, run via HTTP executor instead of local Python",
    )
    parser.add_argument(
        "--sandbox-url",
        type=str,
        default=default_url,
        help="Executor URL for --use-sandbox (default: env PYTHON_EXECUTOR_URL or localhost:8080)",
    )
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout seconds for local run or sandbox compile/run limits")

    args = parser.parse_args()

    if args.database:
        result = generate_reference_solution_for_task(
            database_path=args.database,
            output_path=args.output_dir,
            save_code=args.save_code,
            execute=args.execute,
            use_sandbox=args.use_sandbox,
            sandbox_url=args.sandbox_url,
            timeout=args.timeout,
        )
        if result.get("skipped"):
            print(
                f"Nothing to do: problem_type={result.get('problem_type')!r} "
                "(reference solutions only for classification/regression)."
            )
        elif not args.save_code and not args.execute:
            print("\n" + "=" * 60)
            print("Generated Reference Solution Code:")
            print("=" * 60)
            print(result["code"])

    elif args.databases_dir and args.question_list:
        batch_generate_reference_solutions(
            databases_dir=args.databases_dir,
            question_list_path=args.question_list,
            output_dir=args.output_dir,
            execute=args.execute,
            use_sandbox=args.use_sandbox,
            sandbox_url=args.sandbox_url,
            timeout=args.timeout,
        )

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Generate solution for single task:")
        print("  python reference_solution.py --database ./databases/task_name --save_code")
        print("")
        print("  # Run locally (writes verify/simulated_pred_local.csv):")
        print("  python reference_solution.py --database ./databases/task --execute")
        print("")
        print("  # Run on sandbox:")
        print("  python reference_solution.py --database ./databases/task --execute --use-sandbox --sandbox-url http://host:8080/run_code")
        print("")
        print("  # Generate solutions for all tasks:")
        print("  python reference_solution.py --databases_dir ./databases --question_list ./question_list.json --output_dir ./solutions")
