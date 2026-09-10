import pandas as pd

FILE = "data/amazon_pairs.csv"

df = pd.read_csv(FILE)

print(f"Total pairs: {len(df):,}")

print("\nShowing 10 customer -> AmazonHelp pairs:\n")

for i, row in df.head(10).iterrows():
    print("=" * 70)
    print(f"PAIR {i + 1}")
    print("\nCUSTOMER:")
    print(str(row["customer_text"]).strip())
    print("\nAMAZON:")
    print(str(row["amazon_reply"]).strip())
    print()