# Hiver AI Support Agent

An AI customer-support agent built for the Hiver SDE Intern take-home assignment.

The system focuses on AmazonHelp conversations from the Customer Support on Twitter dataset.

## What it does

The agent:

1. Classifies a customer message into a defined support intent.
2. Retrieves similar historical Amazon support conversations.
3. Generates a grounded customer-facing response.
4. Decides whether the case should be AUTO handled or ESCALATED to a human.

## Current pipeline

Customer message
    ↓
Intent classification
    ↓
Semantic retrieval
    ↓
Historical Amazon responses
    ↓
Groq LLM
    ↓
Grounded reply
    ↓
AUTO / ESCALATE

## Dataset

Primary dataset:
Customer Support on Twitter

The original `twcs.csv` file is intentionally not included in this repository because it is large.

Place the dataset at:

data/twcs.csv

The project extracts AmazonHelp customer-support pairs into:

data/amazon_pairs.csv

## Setup

Create a Python environment and install dependencies:

```bash
py -m pip install -r requirements.txt