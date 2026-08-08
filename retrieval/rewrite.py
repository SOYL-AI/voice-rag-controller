import sys
from pathlib import Path
from typing import List


_REPO_ROOT = Path(__file__).resolve().parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm.client import make_client
from llm.config import load_config


SYSTEM_PROMPT = """
You are a query rewriting assistant.

Rewrite the user's latest question into a standalone search query.

Rules:

- Read the conversation history.
- If the latest question already makes sense by itself,
  return it unchanged.
- Otherwise rewrite it so it is understandable without
  conversation history.
- Never answer the question.
- Return ONLY the rewritten query.
"""


_client = None
_cfg = load_config()


def get_client():
    global _client

    if _client is None:
        _client = make_client()

    return _client


def build_messages(history: List[str], current_query: str):

    conversation = "\n".join(history)

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"""
Conversation History:

{conversation}

Latest User Question:

{current_query}

Rewrite the latest question into a standalone search query.

Return ONLY the rewritten query.
"""
        }
    ]


def rewrite_query(history: List[str], current_query: str):

    client = get_client()

    kwargs = {}

    if _cfg["backend"] == "mock":
        # Deterministic response for local development.
        kwargs["mock_answer"] = current_query

    generation = client.generate(
        build_messages(history, current_query),
        **kwargs,
    )

    return generation.text.strip()


if __name__ == "__main__":

    history = [
        "Does the hotel have a pool?",
        "Yes.",
        "What time does it open?",
        "8 AM."
    ]

    print(rewrite_query(history, "What about kids?"))