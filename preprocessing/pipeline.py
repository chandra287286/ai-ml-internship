
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cleaned_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Features created during feature engineering
FEATURE_COLUMNS = [
    "question_word_count",
    "question_prompt_length_ratio"
]

# This dataset currently has no target column.
TARGET_COLUMN = "target"


def preprocess(df):
    """
    Clean missing values, engineer text-based features, and scale numeric inputs.
    Reuse the saved scaler when available, or fit and save one otherwise.
    Return the processed feature DataFrame X and target y (None if unavailable).
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df = df.copy()

    # 1. Remove columns with more than 40% missing values
    missing_percentage = df.isnull().mean() * 100
    columns_to_drop = missing_percentage[
        missing_percentage > 40
    ].index.tolist()

    df.drop(columns=columns_to_drop, inplace=True)

    # 2. Handle missing values
    for column in df.columns:
        if df[column].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
        else:
            mode_values = df[column].mode(dropna=True)

            if not mode_values.empty:
                fill_value = mode_values.iloc[0]
            else:
                fill_value = "Unknown"

            df[column] = df[column].fillna(fill_value)

    # 3. Check required text columns
    required_columns = ["test_question", "system_prompt"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")

    # 4. Engineer features
    question_text = df["test_question"].astype(str)
    prompt_text = df["system_prompt"].astype(str)

    df["question_char_count"] = question_text.str.len()
    df["prompt_char_count"] = prompt_text.str.len()

    df["question_word_count"] = (
        question_text.str.split().str.len()
    )

    df["question_prompt_length_ratio"] = (
        df["question_char_count"] /
        df["prompt_char_count"].replace(0, np.nan)
    )

    # Handle any undefined ratios
    df["question_prompt_length_ratio"] = (
        df["question_prompt_length_ratio"].fillna(0)
    )

    # 5. Select engineered numeric features
    X = df[FEATURE_COLUMNS].astype(float).copy()

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain infinite or invalid values.")

    # 6. Scale features
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

        if hasattr(scaler, "feature_names_in_"):
            saved_features = list(scaler.feature_names_in_)

            if saved_features != FEATURE_COLUMNS:
                raise ValueError(
                    "Saved scaler features do not match pipeline features."
                )

        X_scaled = scaler.transform(X)

    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        joblib.dump(scaler, SCALER_PATH)

    X = pd.DataFrame(
        X_scaled,
        columns=FEATURE_COLUMNS,
        index=df.index
    )

    # 7. Extract target if one exists
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None

    # 8. Final validation
    assert X.isnull().sum().sum() == 0
    assert np.isfinite(X.to_numpy()).all()

    return X, y


if __name__ == "__main__":

    print("========== PREPROCESSING PIPELINE ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load CSV and pass it to the reusable function
    raw_df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", raw_df.shape)

    X, y = preprocess(raw_df)

    print("\nPreprocessing completed successfully!")
    print("Processed feature shape:", X.shape)
    print("Features:", X.columns.tolist())

    print("\nProcessed feature preview:")
    print(X.head())

    if y is None:
        print("\nTarget: Not available in this dataset.")
    else:
        print("\nTarget column:", TARGET_COLUMN)

    print("\nMissing values in X:", X.isnull().sum().sum())
    print("Scaler saved at:", SCALER_PATH)
