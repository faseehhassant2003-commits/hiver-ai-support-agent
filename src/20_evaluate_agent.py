import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Make Windows console handle Unicode
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


GOLDEN_FILE = Path("data/golden_set.csv")
PAIRS_FILE = Path("data/amazon_pairs.csv")
EMBEDDINGS_FILE = Path("data/amazon_embeddings.npy")

OUTPUT_FILE = Path("data/evaluation_results.csv")

MODEL_NAME = "all-MiniLM-L6-v2"

# Full evaluation
TEST_LIMIT = None

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


def load_resources():
    print("=" * 70)
    print("LOADING EVALUATION RESOURCES")
    print("=" * 70)

    if not PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {PAIRS_FILE}"
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {EMBEDDINGS_FILE}"
        )

    print("Loading historical data...")
    pairs_df = pd.read_csv(PAIRS_FILE)

    pairs_df = pairs_df.dropna(
        subset=["customer_text"]
    ).reset_index(drop=True)

    print(f"Historical cases: {len(pairs_df):,}")

    print("Loading saved embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(embeddings) != len(pairs_df):
        raise ValueError(
            "Embedding count does not match historical data count."
        )

    print(f"Embeddings: {len(embeddings):,}")

    print("Loading semantic model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading Groq client...")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set."
        )

    client = Groq(api_key=api_key)

    print("All resources loaded.")
    print()

    return pairs_df, embeddings, model, client


def retrieve_cases(
    customer_message,
    pairs_df,
    embeddings,
    model,
    top_k=5,
):
    query_embedding = model.encode(
        [customer_message],
        convert_to_numpy=True,
    )[0]

    query_norm = np.linalg.norm(
        query_embedding
    )

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    similarities = np.dot(
        embeddings,
        query_embedding,
    ) / (
        embedding_norms * query_norm + 1e-10
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    cases = []

    for index in top_indices:
        cases.append(
            {
                "customer_text": pairs_df.iloc[index][
                    "customer_text"
                ],
                "amazon_reply": pairs_df.iloc[index][
                    "amazon_reply"
                ],
                "similarity": float(
                    similarities[index]
                ),
            }
        )

    return cases


def decide_escalation(
    intent,
    top_similarity,
):
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
            f"High-risk intent: {intent} "
            f"requires human handling.",
        )

    if top_similarity < minimum_similarity:
        return (
            "ESCALATE",
            f"Historical evidence is weak "
            f"(top similarity={top_similarity:.3f}).",
        )

    return (
        "AUTO",
        f"Known low-risk intent with strong "
        f"historical evidence "
        f"(top similarity={top_similarity:.3f}).",
    )


def call_groq(
    customer_message,
    retrieved_cases,
    client,
):
    evidence = ""

    for i, case in enumerate(
        retrieved_cases,
        start=1,
    ):
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

Intent definitions:

delivery_missing:
Late, missing, delayed, tracking, shipment, delivery,
delivered-but-not-received, or package arrival problems.

order_preorder:
Pre-orders, preorder availability, preorder delivery,
preorder codes, or questions specifically about a pre-ordered item.

refund_payment:
Refunds, charges, billing, payment, transaction,
cashback, or money-related problems.

return_cancellation:
Returning an item, cancelling an order, replacement requests,
or stopping/returning an order.

account_prime:
Account login, password, hacked account, locked account,
Prime membership/account access.

technical_device:
Technical problems with Alexa, Kindle, Fire TV, apps,
devices, software, settings, or device features.

prime_video_digital:
Prime Video, digital video content, Kindle digital content,
streaming, or digital-content viewing issues.

product_seller:
Product availability, seller issues, pricing, product listings,
authenticity, warranty from sellers, or product purchasing questions.

complaint_service:
Complaints specifically about customer service, support,
unresolved cases, rude agents, lack of response, or poor service.

general_other:
Anything that does not clearly fit another category,
including greetings, thanks, social comments, or unclear messages.

Rules:
- Classify the customer's main problem.
- Prefer a specific intent over general_other when the message clearly
  describes a known problem.
- Use the historical evidence to ground the answer.
- Do not invent policies or unsupported facts.
- Do not claim that you performed an action.
- Do not promise that another team will contact the customer.
- Do not ask the customer to provide private account, payment,
  order, password, phone, or email information publicly.
- Keep the reply concise and safe.
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
                "content": (
                    "You are a careful customer-support assistant."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    content = (
        response.choices[0]
        .message
        .content
        .strip()
    )

    # Robust JSON extraction
    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in Groq response."
        )

    json_text = content[start:end + 1]

    result = json.loads(json_text)

    intent = result.get("intent")
    reply = result.get("reply", "")

    if intent not in INTENTS:
        raise ValueError(
            f"Invalid intent returned: {intent}"
        )

    return {
        "intent": intent,
        "reply": reply,
    }


def main():
    if not GOLDEN_FILE.exists():
        print(
            f"ERROR: {GOLDEN_FILE} not found."
        )
        return

    golden_df = pd.read_csv(
        GOLDEN_FILE
    )

    required_columns = {
        "customer_tweet_id",
        "customer_text",
        "golden_intent",
    }

    missing = (
        required_columns
        - set(golden_df.columns)
    )

    if missing:
        print(
            f"ERROR: Missing columns: {missing}"
        )
        return

    blank_labels = (
        golden_df["golden_intent"]
        .isna()
        .sum()
    )

    if blank_labels:
        print(
            f"ERROR: {blank_labels} blank labels."
        )
        return

    if TEST_LIMIT is None:
        test_df = golden_df.copy()
    else:
        test_df = golden_df.head(
            TEST_LIMIT
        ).copy()

    print("=" * 70)
    print("HIVER AI SUPPORT AGENT EVALUATION")
    print("=" * 70)
    print(
        f"Golden examples available: {len(golden_df)}"
    )
    print(
        f"Examples being tested:     {len(test_df)}"
    )
    print()

    # Load everything only ONCE
    pairs_df, embeddings, model, client = (
        load_resources()
    )

    results = []

    for position, (_, row) in enumerate(
        test_df.iterrows(),
        start=1,
    ):
        customer_text = str(
            row["customer_text"]
        )
        golden_intent = str(
            row["golden_intent"]
        ).strip()

        print(
            f"[{position}/{len(test_df)}] "
            f"Tweet ID: {row['customer_tweet_id']}"
        )

        try:
            retrieved_cases = retrieve_cases(
                customer_text,
                pairs_df,
                embeddings,
                model,
                top_k=5,
            )

            prediction = call_groq(
                customer_text,
                retrieved_cases,
                client,
            )

            top_similarity = (
                retrieved_cases[0]["similarity"]
            )

            decision, reason = (
                decide_escalation(
                    prediction["intent"],
                    top_similarity,
                )
            )

            predicted_intent = prediction["intent"]
            reply = prediction["reply"]

            correct = (
                predicted_intent
                == golden_intent
            )

            error = ""

        except Exception as e:
            predicted_intent = "ERROR"
            decision = "ERROR"
            reason = ""
            reply = ""
            correct = False
            error = str(e)

        print(
            f"  Golden:    {golden_intent}"
        )
        print(
            f"  Predicted: {predicted_intent}"
        )
        print(
            f"  Decision:  {decision}"
        )
        print(
            f"  Correct:   {correct}"
        )

        if error:
            print(
                f"  Error:     {error}"
            )

        results.append(
            {
                "customer_tweet_id": row[
                    "customer_tweet_id"
                ],
                "customer_text": customer_text,
                "golden_intent": golden_intent,
                "predicted_intent": predicted_intent,
                "decision": decision,
                "reason": reason,
                "reply": reply,
                "correct": correct,
                "evaluation_error": error,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    valid = results_df[
        results_df["predicted_intent"]
        != "ERROR"
    ]

    errors = results_df[
        results_df["predicted_intent"]
        == "ERROR"
    ]

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total tested:           {len(results_df)}"
    )

    print(
        f"Successful predictions: {len(valid)}"
    )

    print(
        f"Evaluation errors:      {len(errors)}"
    )

    if len(valid) == 0:
        print(
            "\nNo valid predictions were produced."
        )
        print(
            f"Results saved to: {OUTPUT_FILE}"
        )
        return

    accuracy = accuracy_score(
        valid["golden_intent"],
        valid["predicted_intent"],
    )

    macro_f1 = f1_score(
        valid["golden_intent"],
        valid["predicted_intent"],
        average="macro",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("INTENT METRICS")
    print("=" * 70)

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        f"Macro F1: {macro_f1:.3f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            valid["golden_intent"],
            valid["predicted_intent"],
            zero_division=0,
        )
    )

    mistakes = valid[
        valid["golden_intent"]
        != valid["predicted_intent"]
    ]

    print()
    print("=" * 70)
    print(
        f"INTENT MISTAKES: {len(mistakes)}"
    )
    print("=" * 70)

    for _, row in mistakes.iterrows():
        print()
        print(
            "Tweet ID:",
            row["customer_tweet_id"],
        )
        print(
            "Customer:",
            row["customer_text"],
        )
        print(
            "Golden:",
            row["golden_intent"],
        )
        print(
            "Predicted:",
            row["predicted_intent"],
        )
        print(
            "Decision:",
            row["decision"],
        )

    print()
    print("=" * 70)
    print(
        f"Detailed results saved to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()