# ---------------------------------------------------------
# Hiver Escalation Policy
# ---------------------------------------------------------

# Intents where a human is usually safer because the case
# may require account access, payment investigation,
# exceptions, or case-specific judgment.
HIGH_RISK_INTENTS = {
    "refund_payment",
    "account_prime",
    "return_cancellation",
    "complaint_service",
}

# If semantic retrieval is weak, we do not want the AI
# confidently answering from unrelated historical examples.
MIN_SIMILARITY = 0.65


def decide_escalation(intent: str, top_similarity: float):
    """
    Decide whether the case should be AUTO handled
    or ESCALATED to a human.

    Returns:
        decision, reason
    """

    # Rule 1: High-risk intents
    if intent in HIGH_RISK_INTENTS:
        return (
            "ESCALATE",
            f"Intent '{intent}' may require private information, "
            "account access, payment investigation, or human judgment."
        )

    # Rule 2: Weak historical evidence
    if top_similarity < MIN_SIMILARITY:
        return (
            "ESCALATE",
            f"Historical evidence is weak "
            f"(top similarity={top_similarity:.3f})."
        )

    # Rule 3: Otherwise allow automatic handling
    return (
        "AUTO",
        f"Known low-risk intent with strong historical evidence "
        f"(top similarity={top_similarity:.3f})."
    )


# ---------------------------------------------------------
# Test examples
# ---------------------------------------------------------

tests = [
    ("delivery_missing", 0.73),
    ("refund_payment", 0.82),
    ("account_prime", 0.75),
    ("technical_device", 0.71),
    ("prime_video_digital", 0.70),
    ("delivery_missing", 0.48),
]

print("=" * 70)
print("ESCALATION POLICY TEST")
print("=" * 70)

for intent, similarity in tests:

    decision, reason = decide_escalation(
        intent,
        similarity
    )

    print()
    print(f"Intent: {intent}")
    print(f"Similarity: {similarity:.3f}")
    print(f"Decision: {decision}")
    print(f"Reason: {reason}")