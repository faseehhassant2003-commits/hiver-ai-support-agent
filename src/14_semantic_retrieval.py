import pandas as pd
from sentence_transformers import SentenceTransformer, util


FILE = "data/amazon_pairs.csv"

print("Loading Amazon conversations...")

df = pd.read_csv(FILE)

texts = df["customer_text"].fillna("").tolist()

print(f"Loaded {len(texts):,} customer messages.")

# Load a small, general-purpose embedding model.
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Embeddings ready!")


def find_similar_cases(query, top_k=5):
    """
    Find historical customer messages with similar meaning.
    """

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        query_embedding,
        embeddings
    )[0]

    top_results = scores.topk(k=top_k)

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices
    ):
        index = int(index)

        results.append({
            "score": float(score),
            "customer_text": df.iloc[index]["customer_text"],
            "amazon_reply": df.iloc[index]["amazon_reply"],
        })

    return results


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

query = "I can't log into my account and need to reset my password."

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