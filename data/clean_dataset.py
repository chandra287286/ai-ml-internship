
from pathlib import Path
import pandas as pd

# File paths
DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "sample_dataset.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_dataset.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("========== ORIGINAL DATASET ==========")
print("Shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Document cleaning decisions
cleaning_report = []

# Drop columns with more than 40% missing values
missing_percentage = df.isnull().mean() * 100
columns_to_drop = missing_percentage[
    missing_percentage > 40
].index.tolist()

for column in columns_to_drop:
    cleaning_report.append(
        f"{column}: Dropped because more than 40% values were missing."
    )

df.drop(columns=columns_to_drop, inplace=True)

# Handle missing values in remaining columns
for column in df.columns:
    missing_count = df[column].isnull().sum()

    if missing_count == 0:
        cleaning_report.append(
            f"{column}: No missing values. No action required."
        )
        continue

    if pd.api.types.is_numeric_dtype(df[column]):
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

        cleaning_report.append(
            f"{column}: Filled {missing_count} missing values "
            f"using median ({median_value})."
        )

    else:
        mode_values = df[column].mode(dropna=True)

        if not mode_values.empty:
            fill_value = mode_values.iloc[0]
            strategy = f"mode ({fill_value})"
        else:
            fill_value = "Unknown"
            strategy = "Unknown"

        df[column] = df[column].fillna(fill_value)

        cleaning_report.append(
            f"{column}: Filled {missing_count} missing values "
            f"using {strategy}."
        )

# Display cleaning report
print("\n========== CLEANING REPORT ==========")
for item in cleaning_report:
    print("-", item)

# Verify no missing values remain
print("\nMissing values after cleaning:")
print(df.isnull().sum())

assert df.isnull().sum().sum() == 0, \
    "Cleaning failed: missing values still exist!"

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nSUCCESS: Zero missing values verified!")
print("Cleaned dataset saved to:", OUTPUT_FILE)
print("Final shape:", df.shape)
