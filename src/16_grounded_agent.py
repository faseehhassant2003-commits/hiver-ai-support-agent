import os
import json
import pandas as pd

from groq import Groq
from sentence_transformers import SentenceTransformer, util


# =========================================================
# 1. Configuration
# =========================================================

DATA_FILE = "data/amazon_pairs.csv"
MODEL_NAME = "openai/gpt-oss-20b"

customer_message = "My package says delivered but I never received it."


# =========================================================
# 2. Allowed intents
# =========================================================

ALLOWED_INTENTS = [
    "delivery_missing",
    "order_preorder",
    "refund_payment",
    "return_cancellation",
    "account_prime",
    "technical_device",
    "prime_video_digital",
    "product_seller",
    "complaint_service",
    "general_other",
]


# =========================================================
# 3. Check API key
# =========================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found."
    )

client = Groq(api_key=api_key)


# =========================================================
# 4. Load historical data
# =========================================================

print("Loading Amazon historical conversations...")

df = pd.read_csv(DATA_FILE)

df["customer_text"] = df["customer_text"].fillna("")
df["amazon_reply"] = df["amazon_reply"].fillna("")

texts = df["customer_text"].tolist()

print(f"Loaded {len(texts):,} historical customer messages.")


# =========================================================
# 5. Semantic embeddings
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Creating embeddings...")

embeddings = embedding_model.encode(
    texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Embeddings ready!")


# =========================================================
# 6. Retrieve similar historical cases
# =========================================================

query_embedding = embedding_model.encode(
    customer_message,
    convert_to_tensor=True
)

scores = util.cos_sim(
    query_embedding,
    embeddings
)[0]

top_k = 5

top_results = scores.topk(k=top_k)

historical_cases = []

for score, index in zip(
    top_results.values,
    top_results.indices
):

    index = int(index)

    historical_cases.append({
        "similarity": float(score),
        "customer": df.iloc[index]["customer_text"],
        "amazon": df.iloc[index]["amazon_reply"],
    })


# =========================================================
# 7. Build evidence
# =========================================================

evidence = ""

for i, case in enumerate(historical_cases, start=1):

    evidence += f"""
CASE {i}
Similarity: {case['similarity']:.3f}

Customer:
{case['customer']}

Historical Amazon reply:
{case['amazon']}
----------------------------------------
"""


# =========================================================
# 8. LLM instructions
# =========================================================

system_prompt = f"""
You are a careful Amazon customer-support assistant.

You must:

1. Classify the customer's primary intent.
2. Decide AUTO or ESCALATE.
3. Give a short reason.
4. Write a customer-facing reply.

The intent MUST be exactly one of:

{", ".join(ALLOWED_INTENTS)}

The decision MUST be exactly one of:

AUTO
ESCALATE

Rules:

- Use historical Amazon replies as your main evidence.
- Do not invent Amazon policies.
- Do not invent URLs.
- Do not promise refunds, replacements, investigations,
  compensation, or account actions unless supported by evidence.
- Do not request private account/order information publicly.
- If the issue requires private account/order investigation,
  prefer ESCALATE.
- If the evidence is weak or unclear, prefer ESCALATE.
- Keep the reply concise.
- Do not mention that you are an AI.
- Do not mention the historical cases.

Return ONLY valid JSON:

{{
  "intent": "one_allowed_intent",
  "decision": "AUTO or ESCALATE",
  "reason": "short explanation",
  "reply": "customer-facing response"
}}
"""


user_prompt = f"""
Customer message:

{customer_message}

Historical Amazon support evidence:

{evidence}
"""


# =========================================================
# 9. Call Groq
# =========================================================

print("Asking Groq...")

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ],
    temperature=0.1,
)


# =========================================================
# 10. Parse response
# =========================================================

raw_answer = response.choices[0].message.content.strip()

try:
    result = json.loads(raw_answer)

except json.JSONDecodeError:
    print("\nGroq returned invalid JSON:")
    print(raw_answer)
    raise RuntimeError("Invalid JSON returned by Groq.")


# =========================================================
# 11. Validate output
# =========================================================

if result.get("intent") not in ALLOWED_INTENTS:
    raise ValueError(
        f"Invalid intent returned: {result.get('intent')}"
    )

if result.get("decision") not in ["AUTO", "ESCALATE"]:
    raise ValueError(
        f"Invalid decision returned: {result.get('decision')}"
    )


# =========================================================
# 12. Display
# =========================================================

print()
print("=" * 70)
print("HIVER GROUNDED AI AGENT")
print("=" * 70)

print("\nCUSTOMER:")
print(customer_message)

print("\nRESULT:")
print(json.dumps(
    result,
    indent=2,
    ensure_ascii=False
))