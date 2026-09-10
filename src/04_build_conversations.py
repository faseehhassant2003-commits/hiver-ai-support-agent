import pandas as pd

FILE = "data/twcs.csv"
BRAND = "AmazonHelp"
OUTPUT = "data/amazon_conversations.csv"

# ---------------------------------------------------------
# STEP 1: Get all AmazonHelp tweet IDs
# ---------------------------------------------------------

print("Reading AmazonHelp tweets...")

amazon_ids = set()

for chunk in pd.read_csv(FILE, chunksize=100_000):
    amazon_rows = chunk[chunk["author_id"] == BRAND]

    for tweet_id in amazon_rows["tweet_id"].dropna():
        amazon_ids.add(int(tweet_id))

print(f"AmazonHelp tweets: {len(amazon_ids):,}")


# ---------------------------------------------------------
# STEP 2: Find customer tweets that reply to AmazonHelp
# ---------------------------------------------------------

print("Finding customer messages...")

customer_rows = []

for chunk in pd.read_csv(FILE, chunksize=100_000):

    inbound = chunk[chunk["inbound"] == True].copy()

    inbound = inbound.dropna(subset=["in_response_to_tweet_id"])

    inbound["parent_id"] = inbound["in_response_to_tweet_id"].astype(int)

    matching = inbound[inbound["parent_id"].isin(amazon_ids)]

    if len(matching) > 0:
        customer_rows.append(
            matching[
                [
                    "tweet_id",
                    "author_id",
                    "created_at",
                    "text",
                    "in_response_to_tweet_id",
                ]
            ]
        )

customers = pd.concat(customer_rows, ignore_index=True)

print(f"Customer messages found: {len(customers):,}")


# ---------------------------------------------------------
# STEP 3: Save a manageable sample
# ---------------------------------------------------------

# We don't need all 100k+ conversations right now.
# Take a random sample for exploration.

sample = customers.sample(
    n=min(20_000, len(customers)),
    random_state=42
)

sample.to_csv(OUTPUT, index=False)

print(f"Saved {len(sample):,} conversations to:")
print(OUTPUT)