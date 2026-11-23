
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "expanded_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# Input features
FEATURE_COLUMNS = [
    "question_word_count",
    "question_prompt_length_ratio"
]

# Target column
TARGET_COLUMN = "category"


def preprocess(df):
    """
    Clean missing values, engineer text-based features, and scale numeric inputs.
    Reuse the saved scaler when available, or fit and save one otherwise.
    Return processed features X and category labels y.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    df = df.copy()

    # 1. Drop columns with more than 40% missing values
    missing_percentage = df.isnull().mean() * 100
    columns_to_drop = missing_percentage[
        missing_percentage > 40
    ].index.tolist()

    df = df.drop(columns=columns_to_drop)

    # 2. Handle missing values
    for column in df.columns:
        if df[column].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(
                df[column].median()
            )
        else:
            mode_values = df[column].mode(dropna=True)
            fill_value = (
                mode_values.iloc[0]
                if not mode_values.empty
                else "Unknown"
            )
            df[column] = df[column].fillna(fill_value)

    # 3. Validate required columns
    required_columns = [
        "test_question",
        "system_prompt",
        TARGET_COLUMN
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")

    # 4. Remove rows with missing target labels
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    # 5. Engineer numeric features
    question_text = df["test_question"].astype(str)
    prompt_text = df["system_prompt"].astype(str)

    df["question_char_count"] = question_text.str.len()
    df["prompt_char_count"] = prompt_text.str.len()
    df["question_word_count"] = question_text.str.split().str.len()

    df["question_prompt_length_ratio"] = (
        df["question_char_count"] /
        prompt_text.str.len().replace(0, np.nan)
    ).fillna(0)

    # 6. Select features and target
    X = df[FEATURE_COLUMNS].astype(float).copy()
    y = df[TARGET_COLUMN].astype(str).copy()

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain invalid values.")

    # 7. Scale features
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

        if hasattr(scaler, "n_features_in_"):
            if scaler.n_features_in_ != len(FEATURE_COLUMNS):
                raise ValueError("Saved scaler has an incorrect feature count.")

        if hasattr(scaler, "feature_names_in_"):
            if list(scaler.feature_names_in_) != FEATURE_COLUMNS:
                raise ValueError("Saved scaler features do not match.")

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

    # 8. Final validation
    assert X.isnull().sum().sum() == 0
    assert np.isfinite(X.to_numpy()).all()
    assert len(X) == len(y)

    return X, y


if __name__ == "__main__":

    print("========== PREPROCESSING PIPELINE ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", raw_df.shape)

    X, y = preprocess(raw_df)

    print("\nPreprocessing completed successfully!")
    print("Processed feature shape:", X.shape)
    print("Features:", X.columns.tolist())

    print("\nProcessed feature preview:")
    print(X.head())

    print("\nTarget column:", TARGET_COLUMN)

    print("\nCategory distribution:")
    print(y.value_counts())

    print("\nMissing values in X:", X.isnull().sum().sum())
    print("Scaler saved at:", SCALER_PATH)
