import os

from groq import Groq


# ---------------------------------------------------------
# 1. Read the API key
# ---------------------------------------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Set it in PowerShell first."
    )


# ---------------------------------------------------------
# 2. Create Groq client
# ---------------------------------------------------------

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"


# ---------------------------------------------------------
# 3. Customer message
# ---------------------------------------------------------

customer_message = (
    "My package says delivered but I never received it."
)


# ---------------------------------------------------------
# 4. Ask the LLM to classify the intent and decide
#    whether a human should handle the case.
# ---------------------------------------------------------

prompt = f"""
You are an Amazon customer-support assistant.

Customer message:
{customer_message}

Classify the message into exactly one of these intents:

- delivery_missing
- order_preorder
- refund_payment
- return_cancellation
- account_prime
- technical_device
- prime_video_digital
- product_seller
- complaint_service
- general_other

Then decide whether the case should be:

- AUTO
- ESCALATE

Use ESCALATE when solving the issue likely requires private
account/order access or human investigation.

Return exactly this format:

Intent: <intent>
Decision: <AUTO or ESCALATE>
Reason: <short reason>
Reply: <customer-facing reply>
"""


# ---------------------------------------------------------
# 5. Call Groq
# ---------------------------------------------------------

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a careful customer-support assistant. "
                "Do not invent account-specific information."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
    temperature=0.2,
)


# ---------------------------------------------------------
# 6. Print result
# ---------------------------------------------------------

answer = response.choices[0].message.content

print("\n" + "=" * 70)
print("HIVER AI AGENT")
print("=" * 70)

print(answer)