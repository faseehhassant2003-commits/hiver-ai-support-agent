import pandas as pd

INPUT = "data/intent_sample.csv"
OUTPUT = "data/training_data.csv"

df = pd.read_csv(INPUT)

labels = {
    1: "refund_payment",
    2: "product_seller",
    3: "complaint_service",
    4: "complaint_service",
    5: "general_other",
    6: "delivery_missing",
    7: "delivery_missing",
    8: "delivery_missing",
    9: "delivery_missing",
    10: "general_other",
    11: "return_cancellation",
    12: "return_cancellation",
    13: "product_seller",
    14: "delivery_missing",
    15: "delivery_missing",
    16: "general_other",
    17: "general_other",
    18: "refund_payment",
    19: "product_seller",
    20: "order_preorder",
    21: "account_prime",
    22: "delivery_missing",
    23: "prime_video_digital",
    24: "delivery_missing",
    25: "delivery_missing",
    26: "refund_payment",
    27: "delivery_missing",
    28: "return_cancellation",
    29: "delivery_missing",
    30: "complaint_service",
}

# Keep the first 30 examples
training = df.head(30).copy()

training["intent"] = [
    labels[i]
    for i in range(1, len(training) + 1)
]

training = training[
    ["customer_tweet_id", "customer_text", "amazon_reply", "intent"]
]

training.to_csv(OUTPUT, index=False)

print(f"Created training dataset: {len(training)} examples")
print(f"Saved to: {OUTPUT}")

print("\nIntent distribution:")
print(training["intent"].value_counts())