import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/amazon_pairs.csv")
OUTPUT_FILE = Path("data/golden_set_review.csv")

TARGET_SIZE = 200


def main():
    df = pd.read_csv(INPUT_FILE)

    # Remove empty customer messages
    df = df.dropna(subset=["customer_text"]).copy()

    # Remove duplicate customer messages
    df = df.drop_duplicates(subset=["customer_text"])

    # Take a fresh reproducible sample
    golden = df.sample(
        n=min(TARGET_SIZE, len(df)),
        random_state=42
    ).copy()

    # Keep only the information needed for labeling
    golden = golden[
        ["customer_tweet_id", "customer_text"]
    ]

    # Empty column for your manual label
    golden["golden_intent"] = ""

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(OUTPUT_FILE, index=False)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Examples: {len(golden)}")


if __name__ == "__main__":
    main()