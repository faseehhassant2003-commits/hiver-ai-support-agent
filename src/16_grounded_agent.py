import json
import os
import pickle

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq


DATA_PATH = "data/amazon_pairs.csv"
EMBEDDING_PATH = "data/amazon_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"

INTENTS = [
    "delivery_missing",
    "order_preorder",
    "refund_payment",
    "return_cancellation",
    "account_prime",
    "technical_device",
    "prime_video_digital",
    "product_seller",
    "complaint_service",
    "general_other",
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Remove empty customer messages
    df = df.dropna(subset=["customer_text"]).reset_index(drop=True)

    return df


def load_or_create_embeddings(df, model):
    if os.path.exists(EMBEDDING_PATH):
        print("Loading saved embeddings...")
        embeddings = np.load(EMBEDDING_PATH)

        # Make sure the saved embeddings match the current dataset
        if len(embeddings) == len(df):
            print("Embeddings loaded successfully!")
            return embeddings

        print("Saved embeddings do not match dataset. Rebuilding...")

    print("Creating embeddings...")
    embeddings = model.encode(
        df["customer_text"].tolist(),
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(EMBEDDING_PATH, embeddings)

    print(f"Saved embeddings to {EMBEDDING_PATH}")

    return embeddings


def retrieve_cases(query, df, embeddings, model, top_k=5):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )[0]

    # Cosine similarity
    query_norm = np.linalg.norm(query_embedding)
    embedding_norms = np.linalg.norm(embeddings, axis=1)

    similarities = np.dot(embeddings, query_embedding) / (
        embedding_norms * query_norm + 1e-10
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "customer_text": df.iloc[index]["customer_text"],
                "amazon_reply": df.iloc[index]["amazon_reply"],
                "similarity": float(similarities[index]),
            }
        )

    return results


def decide_escalation(intent, top_similarity):
    high_risk_intents = {
        "refund_payment",
        "account_prime",
        "return_cancellation",
        "complaint_service",
    }

    minimum_similarity = 0.65

    if intent in high_risk_intents:
        return (
            "ESCALATE",
            f"High-risk intent: {intent} requires human handling.",
        )

    if top_similarity < minimum_similarity:
        return (
            "ESCALATE",
            f"Historical evidence is weak (top similarity={top_similarity:.3f}).",
        )

    return (
        "AUTO",
        f"Known low-risk intent with strong historical evidence "
        f"(top similarity={top_similarity:.3f}).",
    )


def call_groq(customer_message, retrieved_cases):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Set it in PowerShell before running the program."
        )

    client = Groq(api_key=api_key)

    evidence = ""

    for i, case in enumerate(retrieved_cases, start=1):
        evidence += f"""
CASE {i}
Similarity: {case["similarity"]:.3f}
Customer: {case["customer_text"]}
Historical Amazon reply: {case["amazon_reply"]}
"""

    prompt = f"""
You are an Amazon customer-support assistant.

Customer message:
{customer_message}

Historical evidence:
{evidence}

Choose exactly ONE intent from this list:
{INTENTS}

Rules:
- Classify based on the customer's main problem.
- Use the historical evidence to ground the answer.
- Do not invent policies, refunds, replacements, investigations,
  compensation, account actions, or URLs.
- Do not claim that you performed an action.
- Do not ask the customer to publicly share private information.
- Keep the reply concise and helpful.
- Return ONLY valid JSON.

JSON format:
{{
  "intent": "one_exact_intent_from_the_list",
  "reply": "customer-facing reply"
}}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a careful customer support assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    return json.loads(content)


def run_agent(customer_message):
    print("Loading Amazon historical conversations...")

    df = load_data()

    print(f"Loaded {len(df):,} historical customer messages.")

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = load_or_create_embeddings(df, model)

    print("Embeddings ready!")

    print("Retrieving similar historical cases...")

    retrieved_cases = retrieve_cases(
        customer_message,
        df,
        embeddings,
        model,
        top_k=5,
    )

    print("\nTOP HISTORICAL CASES")

    for i, case in enumerate(retrieved_cases, start=1):
        print(f"\n{i}. Similarity: {case['similarity']:.3f}")
        print(f"Customer: {case['customer_text']}")
        print(f"Amazon: {case['amazon_reply']}")

    result = call_groq(customer_message, retrieved_cases)

    top_similarity = retrieved_cases[0]["similarity"]

    decision, reason = decide_escalation(
        result["intent"],
        top_similarity,
    )

    final_result = {
        "intent": result["intent"],
        "decision": decision,
        "reason": reason,
        "reply": result["reply"],
    }

    print("\nRESULT:")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print('py src\\16_grounded_agent.py "your customer message"')
        sys.exit(1)

    customer_message = " ".join(sys.argv[1:])

    run_agent(customer_message)