
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "expanded_dataset.csv"
RESULTS_PATH = PROJECT_ROOT / "models" / "results.csv"
RULES_PATH = PROJECT_ROOT / "models" / "decision_tree_rules.txt"
PLOT_PATH = PROJECT_ROOT / "models" / "feature_importances.png"


# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# 3. Create features
df["combined_text"] = (
    df["system_prompt"].fillna("") + " " +
    df["test_question"].fillna("")
)

df["question_word_count"] = (
    df["test_question"].fillna("").apply(
        lambda text: len(text.split())
    )
)

prompt_lengths = df["system_prompt"].fillna("").str.len()
question_lengths = df["test_question"].fillna("").str.len()

df["question_prompt_length_ratio"] = (
    question_lengths / prompt_lengths.replace(0, 1)
)

X = df[
    [
        "combined_text",
        "question_word_count",
        "question_prompt_length_ratio"
    ]
]

y = df["category"].astype(str)


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(ngram_range=(1, 2)),
            "combined_text"
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


# 6. Train shallow Decision Tree
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

model.fit(X_train, y_train)


# 7. Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n========== DECISION TREE RESULTS ==========")
print("Maximum depth:", 5)
print("Actual tree depth:",
      model.named_steps["classifier"].get_depth())
print("Accuracy:", round(accuracy, 4))
print("Accuracy percentage:", round(accuracy * 100, 2), "%")


# 8. Extract human-readable rules
feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

tree_classifier = model.named_steps["classifier"]

rules = export_text(
    tree_classifier,
    feature_names=list(feature_names),
    max_depth=5
)

with open(RULES_PATH, "w", encoding="utf-8") as file:
    file.write("DECISION TREE CLASSIFICATION RULES\n")
    file.write("Maximum depth: 5\n\n")
    file.write(rules)

print("\nRules saved:", RULES_PATH)
print("\n========== EXTRACTED RULES ==========")
print(rules)


# 9. Extract feature importances
importances = tree_classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

top_features = importance_df.head(10)

print("\n========== TOP 10 FEATURE IMPORTANCES ==========")
print(top_features.to_string(index=False))


# 10. Plot feature importances
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Decision Tree - Feature Importances")
plt.tight_layout()

plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print("\nFeature importance plot saved:", PLOT_PATH)


# 11. Compare with previous models
results_df = pd.read_csv(RESULTS_PATH)

print("\n========== MODEL COMPARISON ==========")
print(results_df[["model_name", "metric_name", "score"]].to_string(index=False))

# Remove an older Decision Tree result if rerunning
results_df = results_df[
    results_df["model_name"] != "DecisionTree_TFIDF"
]

new_result = pd.DataFrame([{
    "model_name": "DecisionTree_TFIDF",
    "metric_name": "accuracy",
    "score": accuracy,
    "date": date.today().isoformat()
}])

results_df = pd.concat(
    [results_df, new_result],
    ignore_index=True
)

results_df.to_csv(RESULTS_PATH, index=False)

print("\nResults updated:", RESULTS_PATH)
print("\n========== FINAL MODEL RESULTS ==========")
print(results_df.to_string(index=False))

print("\n========== TASK COMPLETED ==========")
