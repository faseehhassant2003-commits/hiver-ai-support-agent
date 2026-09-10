import pandas as pd

INPUT = "data/intent_sample.csv"
OUTPUT = "data/intent_review.txt"

df = pd.read_csv(INPUT)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        f.write("=" * 70 + "\n")
        f.write(f"EXAMPLE {i + 1}\n")
        f.write("\nCUSTOMER:\n")
        f.write(str(row["customer_text"]).strip() + "\n")
        f.write("\nAMAZON REPLY:\n")
        f.write(str(row["amazon_reply"]).strip() + "\n\n")

print(f"Created review file with {len(df)} examples:")
print(OUTPUT)