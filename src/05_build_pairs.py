import pandas as pd

FILE = "data/twcs.csv"
OUTPUT = "data/amazon_pairs.csv"
BRAND = "AmazonHelp"

# ---------------------------------------------------------
# STEP 1: Find AmazonHelp tweets
# ---------------------------------------------------------

print("Step 1: Reading AmazonHelp tweets...")

amazon_tweets = {}

for chunk in pd.read_csv(FILE, chunksize=100_000):

    rows = chunk[chunk["author_id"] == BRAND]

    for _, row in rows.iterrows():

        if pd.isna(row["tweet_id"]):
            continue

        tweet_id = int(row["tweet_id"])

        amazon_tweets[tweet_id] = {
            "text": row["text"],
            "created_at": row["created_at"],
        }

print(f"AmazonHelp tweets: {len(amazon_tweets):,}")


# ---------------------------------------------------------
# STEP 2: Find customer tweets that AmazonHelp replied to
# ---------------------------------------------------------

print("Step 2: Finding customer -> AmazonHelp pairs...")

pairs = []

for chunk in pd.read_csv(FILE, chunksize=100_000):

    # Only customer tweets
    customers = chunk[chunk["inbound"] == True].copy()

    customers = customers.dropna(subset=["tweet_id"])

    # Customer tweet ID -> row
    customer_dict = {
        int(row["tweet_id"]): row
        for _, row in customers.iterrows()
    }

    # Look through AmazonHelp tweets that appear in this chunk.
    # We need the response relationship from the original dataset.
    for _, row in chunk.iterrows():

        if row["author_id"] != BRAND:
            continue

        if pd.isna(row["in_response_to_tweet_id"]):
            continue

        parent_id = int(row["in_response_to_tweet_id"])

        if parent_id in customer_dict:

            customer = customer_dict[parent_id]

            pairs.append({
                "customer_tweet_id": parent_id,
                "customer_text": customer["text"],
                "customer_time": customer["created_at"],
                "amazon_tweet_id": int(row["tweet_id"]),
                "amazon_reply": row["text"],
                "amazon_time": row["created_at"],
            })

    if len(pairs) >= 20_000:
        break


# ---------------------------------------------------------
# STEP 3: Save
# ---------------------------------------------------------

result = pd.DataFrame(pairs)

result.to_csv(OUTPUT, index=False)

print()
print(f"Pairs created: {len(result):,}")
print(f"Saved to: {OUTPUT}")