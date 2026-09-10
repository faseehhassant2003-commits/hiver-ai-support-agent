import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Load Amazon historical conversations
# ---------------------------------------------------------

FILE = "data/amazon_pairs.csv"

df = pd.read_csv(FILE)

print(f"Historical conversations loaded: {len(df):,}")


# ---------------------------------------------------------
# 2. Prepare customer messages
# ---------------------------------------------------------

texts = df["customer_text"].fillna("")


# ---------------------------------------------------------
# 3. Convert messages into TF-IDF vectors
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=20_000
)

matrix = vectorizer.fit_transform(texts)

print(f"TF-IDF matrix created: {matrix.shape}")


# ---------------------------------------------------------
# 4. Function to find similar historical cases
# ---------------------------------------------------------

def find_similar_cases(query, top_k=3):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "score": float(scores[index]),
            "customer_text": df.iloc[index]["customer_text"],
            "amazon_reply": df.iloc[index]["amazon_reply"],
        })

    return results


# ---------------------------------------------------------
# 5. Test the retriever
# ---------------------------------------------------------

query = "i can't log into my account and need help resetting my password"

print()
print("=" * 70)
print("NEW CUSTOMER MESSAGE")
print(query)

print()
print("=" * 70)
print("MOST SIMILAR HISTORICAL CASES")

results = find_similar_cases(query, top_k=5)

for i, result in enumerate(results, start=1):

    print()
    print(f"CASE {i}")
    print(f"Similarity: {result['score']:.3f}")

    print("\nCustomer:")
    print(result["customer_text"])

    print("\nAmazon reply:")
    print(result["amazon_reply"])