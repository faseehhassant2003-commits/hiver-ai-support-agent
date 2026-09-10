import pandas as pd

FILE = "data/twcs.csv"
BRAND = "AmazonHelp"

# First collect AmazonHelp tweet IDs
brand_tweet_ids = set()

for chunk in pd.read_csv(FILE, chunksize=100_000):
    brand_rows = chunk[chunk["author_id"] == BRAND]

    for tweet_id in brand_rows["tweet_id"].dropna():
        brand_tweet_ids.add(str(int(tweet_id)))

# Now find customer tweets replying to AmazonHelp
samples = []

for chunk in pd.read_csv(FILE, chunksize=100_000):
    inbound = chunk[chunk["inbound"] == True].copy()

    inbound["parent_id"] = (
        inbound["in_response_to_tweet_id"]
        .dropna()
        .astype("int64")
        .astype(str)
    )

    matching = inbound[inbound["parent_id"].isin(brand_tweet_ids)]

    for _, row in matching.iterrows():
        samples.append(row)

        if len(samples) >= 50:
            break

    if len(samples) >= 50:
        break

result = pd.DataFrame(samples)

print("\nSample Amazon customer messages:\n")

for i, text in enumerate(result["text"], start=1):
    print(f"{i}. {text}")