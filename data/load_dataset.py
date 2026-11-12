
from pathlib import Path
import pandas as pd

# Locate the CSV file relative to this Python script
DATASET_PATH = Path(__file__).parent / "sample_dataset.csv"

def load_dataset():
    # Load the CSV dataset
    df = pd.read_csv(DATASET_PATH)

    print("\n========== DATASET INFORMATION ==========")

    # Display number of rows and columns
    print("\nNumber of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    # Display column names
    print("\nColumn names:")
    print(df.columns.tolist())

    # Display data types
    print("\nData types:")
    print(df.dtypes)

    # Display first five rows
    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    # Check for missing values
    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    dataset = load_dataset()