import pandas as pd
import os


INPUT_FILE = "data/processed/nvidia_features.csv"

TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)


train_data = df.iloc[:train_end].copy()

validation_data = df.iloc[
    train_end:validation_end
].copy()

test_data = df.iloc[
    validation_end:
].copy()


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

train_data.to_csv(
    TRAIN_FILE,
    index=False
)

validation_data.to_csv(
    VALIDATION_FILE,
    index=False
)

test_data.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 70)

print(
    f"\nTotal rows: {len(df)}"
)

print(
    f"\nTraining rows: {len(train_data)}"
)

print(
    f"Validation rows: {len(validation_data)}"
)

print(
    f"Testing rows: {len(test_data)}"
)


print("\nDate ranges:")

print(
    f"\nTRAIN:"
)

print(
    train_data["Date"].iloc[0],
    "→",
    train_data["Date"].iloc[-1]
)

print(
    f"\nVALIDATION:"
)

print(
    validation_data["Date"].iloc[0],
    "→",
    validation_data["Date"].iloc[-1]
)

print(
    f"\nTEST:"
)

print(
    test_data["Date"].iloc[0],
    "→",
    test_data["Date"].iloc[-1]
)


print("\nTarget ranges:")

print(
    "\nTrain:",
    train_data["Target_Return"].min(),
    "→",
    train_data["Target_Return"].max()
)

print(
    "\nValidation:",
    validation_data["Target_Return"].min(),
    "→",
    validation_data["Target_Return"].max()
)

print(
    "\nTest:",
    test_data["Target_Return"].min(),
    "→",
    test_data["Target_Return"].max()
)


print("\nFiles saved:")
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(TEST_FILE)