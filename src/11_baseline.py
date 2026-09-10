import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------
# 1. Load labeled data
# ---------------------------------------------------------

FILE = "data/amazon_intent_200_auto_labeled.csv"

df = pd.read_csv(FILE)

print(f"Total examples: {len(df)}")


# ---------------------------------------------------------
# 2. Prepare X and y
# ---------------------------------------------------------

X = df["customer_text"].fillna("")
y = df["intent"]


# ---------------------------------------------------------
# 3. Split into training and test data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")


# ---------------------------------------------------------
# 4. Convert text into numbers using TF-IDF
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10_000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ---------------------------------------------------------
# 5. Train Logistic Regression
# ---------------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# ---------------------------------------------------------
# 6. Predict
# ---------------------------------------------------------

predictions = model.predict(X_test_tfidf)


# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

print()
print("=" * 60)
print("BASELINE RESULTS")
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