
from pathlib import Path
import sys
import pandas as pd

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.assistant_agent import AssistantAgent

# Define file paths
DATASET_PATH = PROJECT_ROOT / "data" / "sample_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "week1_results.csv"

def run_tests():

    # Load the dataset
    df = pd.read_csv(DATASET_PATH)

    print("========== WEEK 1 PROJECT TEST ==========")
    print("Dataset loaded successfully")
    print("Total questions:", len(df))

    # Select the first three test questions
    test_data = df.head(3)

    results = []

    for _, row in test_data.iterrows():

        question = str(row["test_question"])
        system_prompt = str(row["system_prompt"])

        # Create an assistant using the dataset prompt
        assistant = AssistantAgent(
            system_prompt=system_prompt
        )

        print("\n" + "=" * 50)
        print("Category:", row["category"])
        print("Question:", question)

        # Generate response using local Ollama
        response = assistant.chat(question)

        print("\nAssistant Response:")
        print(response)

        # Store actual results
        results.append({
            "id": row["id"],
            "category": row["category"],
            "test_question": question,
            "assistant_response": response
        })

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save results as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== TEST SUMMARY ==========")
    print("Questions tested:", len(results_df))
    print("Results saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    run_tests()