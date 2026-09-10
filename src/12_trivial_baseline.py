import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------
# 1. Load labeled data
# ---------------------------------------------------------

FILE = "data/amazon_intent_200_auto_labeled.csv"

df = pd.read_csv(FILE)

print(f"Total examples: {len(df)}")


# ---------------------------------------------------------
# 2. Split the data
# ---------------------------------------------------------

_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["intent"]
)

y_test = test_df["intent"]


# ---------------------------------------------------------
# 3. Find the most common intent
# ---------------------------------------------------------

majority_intent = df["intent"].value_counts().idxmax()

print(f"Majority intent: {majority_intent}")


# ---------------------------------------------------------
# 4. Predict the same intent for every example
# ---------------------------------------------------------

predictions = [majority_intent] * len(y_test)


# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print()
print("=" * 60)
print("TRIVIAL BASELINE RESULTS")
print("=" * 60)

print(f"Accuracy: {accuracy:.3f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)