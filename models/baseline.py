
from pathlib import Path
from datetime import date
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.pipeline import DATA_PATH, preprocess

MODEL_DIR = PROJECT_ROOT / "models"
PREDICTIONS_PATH = MODEL_DIR / "linear_predictions.csv"
RESULTS_PATH = MODEL_DIR / "results.csv"


def train_baseline():

    print("========== BASELINE MODEL TRAINING ==========")

    # 1. Load and preprocess dataset
    df = pd.read_csv(DATA_PATH)
    X_processed, y = preprocess(df)

    print("Dataset shape:", X_processed.shape)
    print("\nCategory distribution:")
    print(y.value_counts())

    # 2. Prepare text and numeric features
    X = pd.DataFrame(index=df.index)

    X["text"] = (
        df["system_prompt"].fillna("").astype(str)
        + " "
        + df["test_question"].fillna("").astype(str)
    )

    question_length = df["test_question"].fillna("").astype(str)
    prompt_length = df["system_prompt"].fillna("").astype(str)

    X["question_word_count"] = question_length.str.split().str.len()

    X["question_prompt_length_ratio"] = (
        question_length.str.len()
        / prompt_length.str.len().replace(0, 1)
    )

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 4. Prepare text and numeric transformations
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(ngram_range=(1, 2)),
                "text"
            ),
            (
                "numeric",
                StandardScaler(),
                [
                    "question_word_count",
                    "question_prompt_length_ratio"
                ]
            )
        ]
    )

    # 5. Build and train Logistic Regression
    model = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    # 6. Predict and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel: Logistic Regression")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy percentage: {accuracy * 100:.2f}%")

    # 7. Save predictions
    prediction_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions
    })

    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\nFirst 5 predictions vs actual:")
    print(prediction_df.head().to_string(index=False))

    # 8. Save baseline score
    results_df = pd.DataFrame([{
        "model_name": "LogisticRegression_TFIDF",
        "metric_name": "accuracy",
        "score": accuracy,
        "date": date.today().isoformat()
    }])

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n========== BASELINE COMPLETED ==========")
    print("Predictions saved:", PREDICTIONS_PATH)
    print("Results saved:", RESULTS_PATH)


if __name__ == "__main__":
    train_baseline()
