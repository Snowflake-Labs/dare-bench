"""
DARE-Bench Reference Solution Generator

This script generates reference solutions for DARE-Bench tasks based on metadata.
The reference solution uses scikit-learn models with parameters specified in all_metadata.json.

The generated solution:
1. Loads train/validation data (CSV, SQLite, or Parquet)
2. Applies preprocessing (imputation, scaling, one-hot encoding)
3. Trains a specified scikit-learn model
4. Makes predictions and saves to prediction.csv
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
from string import Template
from pathlib import Path
from typing import Dict, List, Optional, Any


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
        print(f"⚠ Found {null_targets_train} null targets in training data - removing these rows")
        return None
    
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

# Save predictions
result['predictions'].to_csv('prediction.csv', index=False)
print("✓ Predictions saved to prediction.csv")
'''


def generate_reference_solution_code(metadata_path: str) -> str:
    """
    Generate reference solution Python code from metadata.
    
    Args:
        metadata_path: Path to all_metadata.json
    
    Returns:
        Python code string that can be executed to generate predictions
    """
    with open(metadata_path, "r") as f:
        all_metadata = json.load(f)
    
    question = all_metadata["question"]
    
    feature_cols = question["selected_features"]
    model_type = question["model_type"]
    column_type_inference = question["column_type_inference"]
    target_cols = question["target"]
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    imputer_type = question["imputer_type"]
    problem_type = question["problem_type"]
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
    save_code: bool = False
) -> Dict[str, Any]:
    """
    Generate and optionally execute reference solution for a single task.
    
    Args:
        database_path: Path to the task database folder (containing source/ and verify/)
        output_path: Where to save prediction.csv (defaults to database_path/verify/)
        save_code: Whether to save the generated code to a file
    
    Returns:
        Dictionary with solution code and optionally predictions
    """
    metadata_path = os.path.join(database_path, "verify", "all_metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    code = generate_reference_solution_code(metadata_path)
    
    result = {"code": code}
    
    if save_code:
        code_path = os.path.join(output_path or database_path, "reference_solution.py")
        with open(code_path, "w") as f:
            f.write(code)
        result["code_path"] = code_path
        print(f"✓ Reference solution code saved to: {code_path}")
    
    return result


def batch_generate_reference_solutions(
    databases_dir: str,
    question_list_path: str,
    output_dir: str
) -> None:
    """
    Generate reference solution code for all tasks in question_list.json.
    
    Args:
        databases_dir: Directory containing database folders
        question_list_path: Path to question_list.json
        output_dir: Directory to save generated solution code
    """
    with open(question_list_path, "r") as f:
        questions = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    success = 0
    failed = 0
    
    for q in questions:
        folder_name = q["file_path"]
        database_path = os.path.join(databases_dir, folder_name)
        
        try:
            task_output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(task_output_dir, exist_ok=True)
            
            result = generate_reference_solution_for_task(
                database_path=database_path,
                output_path=task_output_dir,
                save_code=True
            )
            success += 1
        except Exception as e:
            print(f"❌ Failed for {folder_name}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Reference Solution Generation Complete")
    print(f"{'='*60}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
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
    
    args = parser.parse_args()
    
    if args.database:
        # Single task mode
        result = generate_reference_solution_for_task(
            database_path=args.database,
            output_path=args.output_dir,
            save_code=args.save_code
        )
        if not args.save_code:
            print("\n" + "="*60)
            print("Generated Reference Solution Code:")
            print("="*60)
            print(result["code"])
    
    elif args.databases_dir and args.question_list:
        # Batch mode
        batch_generate_reference_solutions(
            databases_dir=args.databases_dir,
            question_list_path=args.question_list,
            output_dir=args.output_dir
        )
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Generate solution for single task:")
        print("  python reference_solution.py --database ./databases/task_name --save_code")
        print("")
        print("  # Generate solutions for all tasks:")
        print("  python reference_solution.py --databases_dir ./databases --question_list ./question_list.json --output_dir ./solutions")

