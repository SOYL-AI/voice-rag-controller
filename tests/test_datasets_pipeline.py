import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_dataset_pipeline():

    scenarios = ROOT / "simulator" / "scenarios.json"

    assert scenarios.exists()

    data = json.loads(scenarios.read_text(encoding="utf-8"))

    required_fields = {
        "id",
        "question",
        "expected_answer",
        "retrieval_needed",
        "source",
        "history",
    }

    assert len(data) > 0

    for item in data:

        assert required_fields.issubset(item.keys())

        assert isinstance(item["retrieval_needed"], bool)

        assert isinstance(item["history"], list)

        assert item["expected_answer"].strip() != ""