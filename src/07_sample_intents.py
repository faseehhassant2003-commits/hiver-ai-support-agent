import pandas as pd

FILE = "data/amazon_pairs.csv"
OUTPUT = "data/intent_sample.csv"

df = pd.read_csv(FILE)

# Take a balanced random sample across the dataset.
sample = df.sample(
    n=min(200, len(df)),
    random_state=42
)

sample[[
    "customer_tweet_id",
    "customer_text",
    "amazon_reply"
]].to_csv(OUTPUT, index=False)

print(f"Total pairs: {len(df):,}")
print(f"Intent-analysis sample: {len(sample):,}")
print(f"Saved to: {OUTPUT}")