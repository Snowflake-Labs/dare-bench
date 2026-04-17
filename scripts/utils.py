import os
import json
import logging
import re

from dataset_mapping import TEST_COLLECTION_MAPPING, TEST_FILE_MAPPING
import time
from string import Template
import numpy as np
import pandas as pd
import requests
import base64
import math
from sklearn.metrics import accuracy_score, f1_score, r2_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class FlushFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False, errors=None):
        # Use append mode by default to continue logging if file exists
        super().__init__(filename, mode, encoding, delay, errors)
    
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logger(log_path, config=None):
    # Get log level from config: Priority: config.log_level > environment > default INFO
    if config and hasattr(config, 'log_level'):
        level_name = config.log_level.upper()
    else:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger(f"ProcessLogger_{log_path}")
    logger.setLevel(level)

    logger.handlers = []

    # Check if log file already exists
    log_exists = os.path.exists(log_path)
    
    fh = FlushFileHandler(log_path)
    fh.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Add a separator and continuation message if log file already existed
    if log_exists:
        logger.info("=" * 80)
        logger.info("CONTINUING EXISTING LOG SESSION")
        logger.info("=" * 80)

    return logger


def cleanup_logger(logger):
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def setup_execution_logger(config, logger_name="ExecutionManager"):
    """
    Set up a centralized logger for execution manager operations.
    
    Args:
        config: Configuration object containing log_path and other settings
        logger_name: Name for the logger (default: "ExecutionManager")
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Determine log file path
    if hasattr(config, 'log_path') and config.log_path:
        # Use the configured log path directory
        log_dir = os.path.dirname(config.log_path) if os.path.dirname(config.log_path) else "."
        log_file = os.path.join(log_dir, "execution_manager.log")
    else:
        # Fallback to result path or current directory
        if hasattr(config, 'result_path') and config.result_path:
            log_dir = config.result_path
        else:
            log_dir = "."
        log_file = os.path.join(log_dir, "execution_manager.log")
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    
    # Get log level from config
    if hasattr(config, 'log_level'):
        level_name = config.log_level.upper()
    else:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    
    level = getattr(logging, level_name, logging.INFO)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create file handler with flush capability
    fh = FlushFileHandler(log_file)
    fh.setLevel(level)
    
    # Create console handler for real-time feedback
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Execution logger initialized. Log file: {log_file}")
    
    return logger

# ------------------------------------------------------------------------------------------------
# Datasci related functions start here
def call_code_executor(data_path, code, url, save_file_type, timeout=100):
    encoded_files = {}
    if save_file_type == 'sqlite':
        files_to_load = ["train_v1_no_err.sqlite", "val_v1.sqlite"]
    elif save_file_type == 'csv':
        files_to_load = ["train_v1_no_err.csv", "val_v1.csv"]
    elif save_file_type == 'parquet':
        files_to_load = ["train_v1_no_err.parquet", "val_v1.parquet"]
    else:
        raise ValueError(f"Invalid save_file_type: {save_file_type}")

    for file_to_load in files_to_load:
        with open(os.path.join(data_path, "source", file_to_load), 'rb') as f:
            content = f.read()
        base64_content = base64.b64encode(content).decode('utf-8')
        encoded_files[file_to_load] = base64_content
    response = requests.post(url, json={
        'code': code,
        'language': 'python',
        'files': {name: base64_content for name, base64_content in encoded_files.items()},
        "fetch_files": ["simulated_pred_local.csv", ],
        "compile_timeout": timeout,
        "run_timeout": timeout,
    })
   
    response = response.json()
    
    run_result = response["run_result"]
    status = run_result.get("status")
    if status == "TimeLimitExceeded":
        raise TimeoutError(f"Code execution exceeded time limit ({run_result.get('execution_time')}s for {data_path})")

    # run_result = {k: run_result[k] for k in ["stdout", "stderr",]}
    if "simulated_pred_local.csv" not in response["files"]:
        raise ValueError(f"simulated_pred_local.csv not generated for {data_path}")

    for file_name, file_content in response["files"].items():
        with open(os.path.join(data_path, "verify", file_name), 'wb') as f:
            f.write(base64.b64decode(file_content))

    return run_result

# Ground truth generation template for datasci dataset
GT_GENERATION_TEMPLATE = """
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

# Function to load and join tables from a SQLite file
def load_and_join(sqlite_path):
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

def train_predict_model(train_df, eval_df, feature_cols, model_type, column_type_inference, target_cols=['answer'], imputer_type="most_frequent", 
                       problem_type="classification", random_state=42):
    print(f"🚀 MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    # Check if target column exists in training data
    for target_coloumn in target_cols:
        if target_coloumn not in train_df.columns:
            print(f"❌ Target column '{target_coloumn}' not found in training data!")
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
    
    # Prepare evaluation features (and target if it exists)
    X_eval = eval_df[feature_cols].copy()
    
    # Check if target column exists in eval_df for evaluation
    has_eval_target = True
    for target_coloumn in target_cols:
        if target_coloumn not in eval_df.columns:
            has_eval_target = False
    if has_eval_target:
        y_eval = eval_df[target_cols].copy()
        print(f"✓ Target column found in evaluation data - will compute metrics")
    else:
        y_eval = None
        print(f" Target column not found in evaluation data - will only make predictions")
    
    # Check for null targets in training data
    null_targets_train = y_train.isnull().sum().sum()  # use two sum to get the total null number
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
        # Check data type in training data
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
    print(f"TRAINING MODEL...")
    try:
        ml_pipeline.fit(X_train, y_train)
        print(f"✓ Model trained successfully")
    except Exception as e:
        print(f" Error during training: {e}")
        return None
    
    # Make predictions
    print(f"MAKING PREDICTIONS...")
    try:
        y_pred = ml_pipeline.predict(X_eval)
        print(f"✓ Predictions completed")
    except Exception as e:
        print(f" Error during prediction: {e}")
        return None
    
    # Evaluate model (only if we have evaluation targets)
    if has_eval_target and y_eval is not None:
        print(f"ODEL EVALUATION")
        print("=" * 30)
        
        # Remove rows with null targets in evaluation for metrics
        valid_eval_mask = y_eval.notna()
        y_eval_clean = y_eval[valid_eval_mask]
        y_pred_clean = y_pred[valid_eval_mask]
        
        if len(y_eval_clean) == 0:
            print(" No valid evaluation targets found - skipping evaluation metrics")
        else:
            if problem_type.lower() == "classification":
                accuracy = accuracy_score(y_eval_clean, y_pred_clean)
                print(f"✓ Accuracy: {accuracy:.4f}")
                print(f"Classification Report:")
                print(classification_report(y_eval_clean, y_pred_clean))
                
                # Show sample predictions
                print(f"Sample Predictions:")
                for i in range(min(10, len(y_eval_clean))):
                    actual = y_eval_clean.iloc[i]
                    predicted = y_pred_clean[i]
                    status = "✓" if actual == predicted else "❌"
                    print(f"   {status} Row {i}: Actual={actual}, Predicted={predicted}")
            
            else:
                mse = mean_squared_error(y_eval_clean, y_pred_clean)
                r2 = r2_score(y_eval_clean, y_pred_clean)
                rmse = np.sqrt(mse)
                
                print(f"✓ R² Score: {r2:.4f}")
                print(f"✓ RMSE: {rmse:.4f}")
                print(f"✓ MSE: {mse:.4f}")
                
                # Show sample predictions
                print(f"Sample Predictions:")
                for i in range(min(10, len(y_eval_clean))):
                    actual = y_eval_clean.iloc[i]
                    predicted = y_pred_clean[i]
                    diff = abs(actual - predicted)
                    print(f"   Row {i}: Actual={actual:.3f}, Predicted={predicted:.3f}, Diff={diff:.3f}")
    else:
        print(f"EVALUATION SKIPPED - No target column in evaluation data")
        print(f"✓ Generated {len(y_pred)} predictions")
    
    y_pred_df = pd.DataFrame(y_pred, columns=y_train.columns)
    y_pred_df.insert(0, 'row_id', eval_df["row_id"].values)
    return {
        'pipeline': ml_pipeline,
        'predictions': y_pred_df,
        'eval_indices': X_eval.index,
        'problem_type': problem_type,
        'X_train': X_train,
        'y_train': y_train,
        'X_eval': X_eval,
        'y_eval': y_eval if has_eval_target else None,
        'has_eval_target': has_eval_target,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }    

feature_cols = $feature_cols
model_type = "$model_type"
column_type_inference = $column_type_inference
target_cols = $target_cols
imputer_type = "$imputer_type"
problem_type = "$problem_type"
random_state = $random_state
save_file_type = "$save_file_type"

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

result = train_predict_model(
    train_df = train_df,
    eval_df = eval_df,
    feature_cols = feature_cols,
    model_type = model_type,
    column_type_inference=column_type_inference,
    target_cols=target_cols,
    imputer_type=imputer_type,
    problem_type=problem_type,
    random_state=random_state
)

result['predictions'].to_csv('simulated_pred_local.csv', index=False)
"""

def get_simulate_results(data_path, url, timeout): 
    with open(os.path.join(data_path, "verify", "all_metadata.json"), "r") as f:
        all_metadata = json.load(f)

    feature_cols = all_metadata["question"]["selected_features"]
    model_type = all_metadata["question"]["model_type"]
    column_type_inference = all_metadata["question"]["column_type_inference"]
    target_cols = all_metadata["question"]["target"]
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    assert isinstance(target_cols, list), "Target column must be a list or string, but found : {}".format(target_cols)
    imputer_type = all_metadata["question"]["imputer_type"]
    problem_type = all_metadata["question"]["problem_type"]
    random_state = all_metadata["question"]["random_state"]
    save_file_type = all_metadata["question"]["save_file_type"]
    groud_truth_code = Template(GT_GENERATION_TEMPLATE).substitute(
        feature_cols=feature_cols,
        model_type=model_type,
        column_type_inference=column_type_inference,
        target_cols=target_cols,
        imputer_type=imputer_type,
        problem_type=problem_type,
        random_state=random_state,
        save_file_type=save_file_type
    )
    call_code_executor(data_path, groud_truth_code, url, save_file_type, timeout=timeout)

def skill_score(y_true, y_pred, clip=True):
    # if any nan in y_pred, return 0.0
    if any([math.isnan(x) for x in y_pred]):
        return 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse_model = np.mean((y_true - y_pred)**2)
    mse_base  = np.mean((y_true - np.mean(y_true))**2)
    skill = 1 - mse_model/mse_base
    return float(np.clip(skill, 0, 1)) if clip else skill

def get_datasci_metric_for_single_column(pred_col, gt_col, task, ds_question_version):
    # handle length mismatch
    if len(pred_col) != len(gt_col):
        return 0.0

    if task == "classification":
        if ds_question_version == "v1":
            return float(gt_col == pred_col)
        elif ds_question_version == "v2":
            try:
                # reward =  accuracy_score(session.ground_truth, prediction)
                reward = f1_score(gt_col, pred_col, average="macro")
                return reward
            except:
                return 0.0
        else:
            raise NotImplementedError(f"Unsupported ds_question_version: {ds_question_version}")

    elif task in ["regression"]:
        if ds_question_version == "v1":
            return float(np.allclose(np.array(gt_col), np.array(pred_col)))
        elif ds_question_version == "v2":
            try:
                # reward = skill_score(gt_col, pred_col)
                # we use clipped r2 score as the reward, 0 is the worst and 1 is the best
                reward = np.clip(r2_score(gt_col, pred_col), 0, 1)
                return reward
            except:
                return 0.0
        else:
            raise NotImplementedError(f"Unsupported ds_question_version: {ds_question_version}")
    elif task == "time_series_analysis":
        try:
            reward = np.clip(r2_score(gt_col, pred_col), 0, 1)
            return reward
        except:
            return 0.0
    else:
        raise NotImplementedError(f"Evaluation for task {task} not implemented")


def _datasci_alignment_id_columns(ground_truth_pd, target_columns, task: str, ds_question_version: str):
    """Align prediction vs GT — keep in sync with dare-bench/dare-bench/scripts/evaluation.py."""
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    tset = set(target_columns)
    if task == "time_series_analysis" and ds_question_version == "v2":
        return [c for c in ground_truth_pd.columns if c not in tset]
    if "row_id" in ground_truth_pd.columns:
        return ["row_id"]
    return [c for c in ground_truth_pd.columns if c not in tset]


def get_final_result_for_eval(session, config, eval_context=None):
    if config.dataset_name.startswith("datasci"):
        if not os.path.exists(os.path.join(config.cache_dir, "prediction.csv")):
            return {
                "prediction_exist": 0.0,
                "final_score": 0.0
            }
        else:
            # we actually store the prediction path rather than the prediction itself
            ground_truth_path = session.ground_truth
            target_columns = session.metadata["target"]
            if isinstance(target_columns, str):
                target_columns = [target_columns]
            with open(ground_truth_path, "r") as f:
                ground_truth_pd = pd.read_csv(f)

            with open(os.path.join(config.cache_dir, "prediction.csv")) as f:
                prediction = pd.read_csv(f)

            id_columns = _datasci_alignment_id_columns(
                ground_truth_pd, target_columns, session.task, config.ds_question_version
            )
            if not id_columns:
                return {"prediction_exist": 1.0, "final_score": 0.0}
            if not all(c in prediction.columns for c in id_columns):
                return {"prediction_exist": 1.0, "final_score": 0.0}

            pred_cols = id_columns + [c for c in target_columns if c in prediction.columns]
            merged = ground_truth_pd.merge(
                prediction[pred_cols],
                on=id_columns,
                how="inner",
                suffixes=("_gt", "_pred"),
            )
            if len(merged) != len(ground_truth_pd):
                return {"prediction_exist": 1.0, "final_score": 0.0}

            per_target_scores = {}
            for col in target_columns:
                if col not in prediction.columns:
                    per_target_scores[col] = 0.0  # model didn't predict this column
                else:
                    gt_c, pr_c = f"{col}_gt", f"{col}_pred"
                    pred_col = merged[pr_c].tolist()
                    gt_col = merged[gt_c].tolist()
                    per_target_scores[col] = get_datasci_metric_for_single_column(
                        pred_col, gt_col, session.task, config.ds_question_version
                    )
            # return the average of all target columns
            final_score = np.mean(list(per_target_scores.values()))
            return {
                "prediction_exist": 1.0,
                "final_score": final_score,
            }
                
    raise ValueError(f"get_final_result_for_eval: unsupported dataset {config.dataset_name!r}")

# ------------------------------------------------------------------------------------------------
# Datascience related functions end here

def load_content_filter_errors(filter_file_path):
    """
    Load the list of question IDs that cause Azure OpenAI Content Filter errors.
    
    Args:
        filter_file_path: Path to the file containing filtered question IDs
        
    Returns:
        set: Set of question IDs to filter out
    """
    filtered_ids = set()
    
    if os.path.exists(filter_file_path):
        with open(filter_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and header lines
                if line and not line.startswith('=') and not line.startswith('Question IDs'):
                    filtered_ids.add(line)
        print(f"Loaded {len(filtered_ids)} question IDs to filter out from {filter_file_path}")
    else:
        print(f"Content filter file {filter_file_path} not found. No filtering will be applied.")
    
    return filtered_ids

def get_processed_data(dataset_name, config):
    if not dataset_name.startswith("datasci"):
        raise ValueError(f"Only datasci datasets are supported; got {dataset_name!r}")

    filter_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "filter", "content_filter_errors.txt")
    filtered_question_ids = load_content_filter_errors(filter_file_path)

    assert "ds_question_version" in config, "ds_question_version is not set in config for datasci dataset"
    question_key = f"question_{config.ds_question_version}"

    # Unified bench layout: dare-bench/data/eval/{question_list.json,databases/<task>/...}
    if dataset_name == "datasci-eval":
        eval_root = getattr(config, "datasci_eval_root", None)
        if not eval_root:
            raise ValueError(
                "datasci_eval_root must be set when dataset_name=datasci-eval "
                "(directory containing question_list.json and databases/)."
            )
        db_base_path = os.path.join(eval_root, "databases")
    else:
        db_base_path = config.db_base_path

    def process(x):
        if x["task"] == "time_series_analysis":
            if question_key == "question_v1":
                ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "ground_truth_v1.csv")
                assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
                with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
                    all_metadata = json.load(f)
                needed_files = all_metadata['question']["needed_files_v1"]

            else:
                ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "ground_truth_v2.csv")
                assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
                with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
                    all_metadata = json.load(f)
                needed_files = all_metadata['question']["needed_files_v2"]

        elif question_key == "question_v1":
            verify_folder = os.path.join(db_base_path, x["file_path"], "verify")
            assert os.path.exists(verify_folder), f"Verify folder not found for {x['file_path']} with {verify_folder}"
            assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
            with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
                all_metadata = json.load(f)
            needed_files = all_metadata['question']["needed_files_v1"]
            ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "simulated_pred_local.csv")
            # We will simulate the results if the ground truth is not found or we force to simulate with the code executor for enforcing the environment discrepency
            if (not os.path.exists(ground_truth_path)) or config.force_simulate:
                print("start to simulate the results for ", x["file_path"])
                start_time = time.time()
                get_simulate_results(os.path.join(db_base_path, x["file_path"]), config.tool.python_executor.url, timeout=config.tool.python_executor.timeout)
                end_time = time.time()
                print(f"Simulating the results for {x['file_path']} took {end_time - start_time} seconds")
        elif question_key == "question_v2":
            ground_truth_path = os.path.join(db_base_path, x["file_path"], "verify", "ground_truth.csv")
            assert os.path.exists(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json")), f"all_metadata file not found for {x['file_path']}"
            with open(os.path.join(db_base_path, x["file_path"], "verify", "all_metadata.json"), "r") as f:
                all_metadata = json.load(f)
            needed_files = all_metadata['question']["needed_files_v2"]
        else:
            raise NotImplementedError(f"Question version {config.ds_question_version} not implemented")

        if "metadata.txt" not in needed_files:
            needed_files.append("metadata.txt")

        # try to read the ground truth file to ensure it exists, we only save the ground_truth_path and load during evaluation
        with open(ground_truth_path) as f:
            pd.read_csv(ground_truth_path)

        return [
            {
                "id": x["file_path"],
                "original_file_path": os.path.join(db_base_path, x["file_path"], "source"), # contains the the path to the original file
                "question": [x[question_key], ], # we will use [prompt, ] to make it compatible with multi-turn structure
                "metadata": {
                    "needed_files": needed_files,
                    "target": all_metadata["question"]["target"],
                },
                "task": x["task"],
                "tools": x["available_tools"],
                "ground_truth": ground_truth_path, # we will use the ground truth path to load the ground truth during evaluation
                "multi_turn": True,
            }
        ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset_name == "datasci-eval":
        eval_root = getattr(config, "datasci_eval_root", None)
        if not eval_root:
            raise ValueError(
                "datasci_eval_root must be set when dataset_name=datasci-eval "
                "(directory containing question_list.json and databases/)."
            )
        data_path = os.path.join(eval_root, "question_list.json")
    else:
        data_path = os.path.join(script_dir, "..", "data", TEST_FILE_MAPPING[dataset_name])
    data = load_file(data_path, sort_by_id=True)

    if dataset_name == "datasci-eval":
        problem_type = getattr(config, "datasci_problem_type", None)
        if not problem_type:
            raise ValueError(
                "datasci_problem_type must be set when dataset_name=datasci-eval "
                "(e.g. classification, regression, time_series_analysis — must match question_list.json \"task\")."
            )
        before = len(data)
        data = [x for x in data if x.get("task") == problem_type]
        print(
            f"datasci-eval: filtered question_list.json by datasci_problem_type={problem_type!r}: "
            f"{len(data)} tasks (from {before})."
        )

    filtered_data_info = {}
    if filtered_question_ids:
        for item in data:
            iid = item.get("id") or item.get("file_path")
            if iid and iid in filtered_question_ids:
                filtered_data_info[iid] = True
        print(f"Marked {len(filtered_data_info)} questions from {dataset_name} as filtered due to content filter errors")
        
    # if config.max_eval_samples == -1:
    #     processed_data = [process(x) for x in data]
    # else:
    #     processed_data = [process(x) for x in data[:config.max_eval_samples]]

    # optionally slice
    if config.max_eval_samples != -1:
        data = data[:config.max_eval_samples]

    processed_data = []
    # with ThreadPoolExecutor(max_workers=config.get("num_threads", 64)) as executor:
    #     # schedule all the jobs
    #     future_to_item = {executor.submit(process, x): x for x in data}
    #     for future in as_completed(future_to_item):
    #         try:
    #             processed_data.append(future.result())
    #         except Exception as e:
    #             item = future_to_item[future]
    #             print(f"Error processing {item['file_path']}: {e}")
    #             raise e
    with ThreadPoolExecutor(max_workers=config.get("num_threads", 64)) as executor:
        futures = {executor.submit(process, x): x for x in data}
        # wrap the as_completed iterator in tqdm, supplying total
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            item = futures[future]
            try:
                processed_data.append(future.result())
            except Exception as e:
                print(f"Error processing {item['file_path']}: {e}")
                raise
    # # use sequential now to avoid potential issues
    # for idx, x in enumerate(data):
    #     # try:
    #     print("Start to process", idx, x.get("file_path", x.get("id", "unknown")))
    #     out = process(x)
    #     processed_data.append(out)
    #     print("End of process", idx, x.get("file_path", x.get("id", "unknown")))
    #     # breakpoint()
    #     # except Exception as e:
    #     #     print(f"Error processing {x.get('file_path', x.get('id', 'unknown'))}: {e}")
    #     #     raise e
    for item_list in processed_data:
        for item in item_list:
            item["is_filtered"] = item["id"] in filtered_data_info
    return processed_data

 
def get_processed_data_dict(config):
    processed_data_dict = dict()
    if config.dataset_name in TEST_COLLECTION_MAPPING:
        dataset_names = TEST_COLLECTION_MAPPING[config.dataset_name]
    elif config.dataset_name in TEST_FILE_MAPPING:
        dataset_names = [config.dataset_name]
    else:
        raise ValueError("Dataset name not identified.")
    
    for dataset_name in dataset_names:
        processed_data = get_processed_data(dataset_name, config)
        processed_data_dict[dataset_name] = processed_data
    return processed_data_dict
        
                

def load_file(file_path, sort_by_id=False):
    """
    Load either:
      - A single JSON array/object file (e.g. "[ { ... }, { ... }, ... ]")
      - A JSONL file (one JSON object per line)
    """
    try:
        # First try to parse the whole file as one JSON document
        with open(file_path, 'r') as f:
            data = json.load(f)
        # If it's not a list, wrap it or raise
        if not isinstance(data, list):
            raise ValueError("Expected top-level list")
        result = data

    except (json.JSONDecodeError, ValueError):
        # Fallback: parse line-by-line as JSONL
        result = []
        with open(file_path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Bad JSON on line: {line!r}") from e
    try:
        if sort_by_id:
            result.sort(key=sort_key)
    except:
        print(f"Error sorting {file_path}, skipping sorting")

    return result

def sort_key(entry):
    """
    Index comes in two forms: TestCategory_Index or TestCategory_Index-FuncDocSubIndex-PromptSubIndex; both 0-indexed.

    TestCategory_Index: For example, `simple_20` means the 21st entry in the `simple` test category.

    TestCategory_Index-FuncDocSubIndex-PromptSubIndex is used when there are multiple prompts for a single function doc; this only happens in the live dataset.
    FuncDocSubIndex increments for each unique function doc.
    PromptSubIndex is per function doc. It resets to 0 for each function doc.
        For example, `live_simple_19-3-15` means the 20th entry in the `live_simple` test category.
        This entry has the 4th unique function doc and the 16th prompt for that function doc (there are at least 15 other prompts for this same function doc in this category).

    In either case, the universal index is enough to sort the entries.
    """
    eid = entry.get("id") or entry.get("file_path")
    if not eid:
        return ("", 0)
    parts = str(eid).rsplit("_", 1)
    if len(parts) < 2:
        return (str(eid), 0)
    test_category, index = parts[0], parts[1]
    # This handles the case where the index is in the form TestCategory_Index-FuncDocSubIndex-PromptSubIndex
    if "-" in index:
        index = index.split("-")[0]
    return (test_category, int(index))
