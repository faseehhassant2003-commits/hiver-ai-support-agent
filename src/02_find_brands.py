import pandas as pd

FILE = "data/twcs.csv"
BRAND = "AmazonHelp"

print("Step 1: Finding AmazonHelp tweet IDs...")

brand_tweet_ids = set()

for chunk in pd.read_csv(FILE, chunksize=100_000):
    brand_rows = chunk[chunk["author_id"] == BRAND]

    for tweet_id in brand_rows["tweet_id"]:
        brand_tweet_ids.add(str(int(tweet_id)))

print(f"AmazonHelp tweets found: {len(brand_tweet_ids):,}")

print("\nStep 2: Finding customer tweets that reply to AmazonHelp...")

customer_messages = 0

for chunk in pd.read_csv(FILE, chunksize=100_000):
    inbound_rows = chunk[chunk["inbound"] == True]

    for value in inbound_rows["in_response_to_tweet_id"].dropna():
        if str(int(value)) in brand_tweet_ids:
            customer_messages += 1

print(f"Customer messages replying to AmazonHelp: {customer_messages:,}")