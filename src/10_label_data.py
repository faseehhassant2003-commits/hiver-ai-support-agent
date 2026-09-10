import pandas as pd
import os

INPUT = "data/intent_sample.csv"
OUTPUT = "data/labeled_training.csv"

INTENTS = {
    "1": "delivery_missing",
    "2": "order_preorder",
    "3": "refund_payment",
    "4": "return_cancellation",
    "5": "account_prime",
    "6": "technical_device",
    "7": "prime_video_digital",
    "8": "product_seller",
    "9": "complaint_service",
    "10": "general_other",
}

# ---------------------------------------------------------
# Load the intent-analysis sample
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# Load previous labels if they exist
# ---------------------------------------------------------

if os.path.exists(OUTPUT):
    labeled = pd.read_csv(OUTPUT)

    # Correct our first four labels
    corrections = {
        0: "refund_payment",
        1: "product_seller",
        2: "general_other",
        3: "complaint_service",
    }

    for row_number, correct_label in corrections.items():
        if row_number < len(labeled):
            labeled.loc[row_number, "intent"] = correct_label

    # Save corrections
    labeled.to_csv(OUTPUT, index=False)

    labeled_ids = set(labeled["customer_tweet_id"])

else:
    labeled = pd.DataFrame()
    labeled_ids = set()

# ---------------------------------------------------------
# Display instructions
# ---------------------------------------------------------

print("\nAmazonHelp Intent Labeling Tool")
print("=" * 50)

print("\nIntent choices:")

for number, intent in INTENTS.items():
    print(f"{number} = {intent}")

print("\nType 'q' to stop and save your progress.\n")

# ---------------------------------------------------------
# Label examples
# ---------------------------------------------------------

for _, row in df.iterrows():

    tweet_id = row["customer_tweet_id"]

    # Skip examples already labeled
    if tweet_id in labeled_ids:
        continue

    print("=" * 70)

    print("CUSTOMER MESSAGE:")
    print(str(row["customer_text"]).strip())

    print("\nAMAZON REPLY:")
    print(str(row["amazon_reply"]).strip())

    while True:

        choice = input("\nChoose intent (1-10): ").strip()

        if choice == "q":
            print("\nProgress saved.")
            print(f"Total labeled: {len(labeled):,}")
            raise SystemExit

        if choice in INTENTS:
            break

        print("Please enter a number from 1 to 10.")

    # -----------------------------------------------------
    # Save the new label
    # -----------------------------------------------------

    new_row = pd.DataFrame([{
        "customer_tweet_id": tweet_id,
        "customer_text": row["customer_text"],
        "amazon_reply": row["amazon_reply"],
        "intent": INTENTS[choice],
    }])

    labeled = pd.concat(
        [labeled, new_row],
        ignore_index=True
    )

    labeled.to_csv(
        OUTPUT,
        index=False
    )

    labeled_ids.add(tweet_id)

    print(f"Saved. Total labeled: {len(labeled):,}")

# ---------------------------------------------------------
# Finished
# ---------------------------------------------------------

print("\nAll examples have been labeled!")

print(f"Final dataset: {OUTPUT}")