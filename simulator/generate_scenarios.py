import json
import re
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "retrieval" / "corpus"
OUTPUT_FILE = Path(__file__).parent / "scenarios.json"

IGNORED_DOCUMENTS = {
    "03_early_checkin_free.txt",
    "52_early_checkin_paid.txt",

    "21_breakfast_v1.txt",
    "58_breakfast_v2.txt",

    "31_pool_v1.txt",
    "59_pool_v2.txt",

    "47_parking_v1.txt",
    "60_parking_v2.txt",

    "53_pet_policy_allowed.txt",
    "54_pet_policy_not_allowed.txt",
}

def extract_goal(filename: str) -> str:
    """
    Convert filename into a readable goal.
    Example:
    01_checkin_policy.txt -> Checkin Policy
    """
    name = re.sub(r"^\d+_", "", filename)
    name = name.replace(".txt", "")
    name = name.replace("_", " ")
    return name.title()


def extract_answer(text: str) -> str:
    """
    Extract the actual policy paragraph.

    Document format:

    Title

    Hotel Name

    Policy text
    """

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Most corpus files follow:
    #   0 -> Title
    #   1 -> Hotel name
    #   2 -> Actual policy
    if len(paragraphs) >= 3:
        return paragraphs[2]

    # Fallback for shorter documents.
    return paragraphs[-1]


def generate_question(goal: str) -> str:

    goal = goal.lower()

    if "checkin" in goal:
        return "What is the hotel's check-in policy?"

    if "checkout" in goal:
        return "What is the hotel's check-out policy?"

    if "airport" in goal:
        return "Does the hotel provide airport pickup service?"

    if "luggage" in goal:
        return "Can I store my luggage at the hotel?"

    if "currency" in goal:
        return "Does the hotel provide currency exchange?"

    if "complaint" in goal:
        return "How do I report a complaint?"

    if "lost" in goal:
        return "What should I do if I lose an item?"

    if "concierge" in goal:
        return "What concierge services are available?"

    return f"Tell me about {goal}."


def main():

    scenarios = []

    txt_files = sorted(CORPUS_DIR.glob("*.txt"))

    for idx, file in enumerate(txt_files, start=1):
        if file.name in IGNORED_DOCUMENTS:
            continue

        text = file.read_text(encoding="utf-8")

        goal = extract_goal(file.name)

        question = generate_question(goal)

        answer = extract_answer(text)

        scenarios.append({

            "id": idx,

            "document": file.name,

            "source": file.name,

            "goal": goal,

            "question": question,

            "expected_answer": answer,

            "requires_lookup": True,

            "retrieval_needed": True,

            "history": []

        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(scenarios, f, indent=2)

    print(f"Generated {len(scenarios)} scenarios.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()