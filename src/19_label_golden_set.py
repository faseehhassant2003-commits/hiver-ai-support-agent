import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/golden_set_review.csv")
OUTPUT_FILE = Path("data/golden_set.csv")


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


def show_intents():
    print("\nChoose an intent:")
    for number, intent in INTENTS.items():
        print(f"{number} = {intent}")


def find_first_unlabeled_row(df):
    """
    Find the first row whose golden_intent is empty.
    """
    for i in range(len(df)):
        value = df.loc[i, "golden_intent"]

        if pd.isna(value):
            return i

        if str(value).strip() == "":
            return i

    return None


def main():
    # Check input file
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return

    # Load as string so labels such as 'account_prime' work correctly
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"golden_intent": "string"},
    )

    # Make sure the column exists
    if "golden_intent" not in df.columns:
        df["golden_intent"] = pd.Series(
            [""] * len(df),
            dtype="string"
        )

    # Convert the column to string type
    df["golden_intent"] = df["golden_intent"].astype("string")

    # Find where to resume
    start_index = find_first_unlabeled_row(df)

    # Everything already labeled
    if start_index is None:
        print("All examples have already been labeled!")
        print(f"Output file: {OUTPUT_FILE}")
        return

    print("=" * 70)
    print("GOLDEN SET LABELING")
    print("=" * 70)
    print(f"Total examples: {len(df)}")
    print("Type a number from 1-10.")
    print("Type S to skip this example.")
    print("Type Q to save and quit.")

    # Start labeling
    for i in range(start_index, len(df)):

        # Skip rows that already have a label
        existing = df.loc[i, "golden_intent"]

        if pd.notna(existing) and str(existing).strip() != "":
            continue

        tweet_id = df.loc[i, "customer_tweet_id"]
        text = str(df.loc[i, "customer_text"])

        print("\n" + "=" * 70)
        print(f"Example {i + 1} of {len(df)}")
        print(f"Tweet ID: {tweet_id}")
        print("=" * 70)
        print(text)
        print("=" * 70)

        while True:
            show_intents()

            choice = input("\nYour choice: ").strip().lower()

            # Quit and save
            if choice == "q":
                df.to_csv(OUTPUT_FILE, index=False)

                print("\n" + "=" * 70)
                print("PROGRESS SAVED")
                print("=" * 70)
                print(f"File: {OUTPUT_FILE}")
                return

            # Skip
            if choice == "s":
                print("Skipped.")
                break

            # Valid intent
            if choice in INTENTS:
                selected_intent = INTENTS[choice]

                df.loc[i, "golden_intent"] = selected_intent

                # Save after every label
                df.to_csv(OUTPUT_FILE, index=False)

                print(f"\nSaved: {selected_intent}")
                break

            print("\nInvalid choice.")
            print("Please enter 1-10, S, or Q.")

    # Final save
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Golden set saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()