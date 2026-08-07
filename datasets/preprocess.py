import json
from pathlib import Path

from loader import DatasetLoader


OUTPUT_DIR = Path(__file__).parent / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


class DatasetPreprocessor:

    def __init__(self):
        self.loader = DatasetLoader()

    def preprocess_spokenwoz(self):

        raw_data = self.loader.load_spokenwoz()

        processed = []

        for dialogue_id, dialogue in raw_data.items():

            goal = dialogue.get("goal", {})
            log = dialogue.get("log", [])

            history = []

            for turn in log:

                speaker = turn.get("tag", "")
                text = turn.get("text", "").strip()

                if not text:
                    continue

                processed.append({
                    "dataset": "spokenwoz",
                    "dialogue_id": dialogue_id,
                    "goal": goal,
                    "speaker": speaker,
                    "text": text,
                    "history": history.copy()
                })

                history.append({
                    "speaker": speaker,
                    "text": text
                })

        output_file = OUTPUT_DIR / "spokenwoz_processed.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2)

        print(f"Saved {len(processed)} dialogue turns.")
    def preprocess_spoken_squad(self):

        squad_folder = self.loader.load_spoken_squad()

        print(f"Spoken-SQuAD located at:")
        print(squad_folder)

        # We'll process this dataset next.


if __name__ == "__main__":

    processor = DatasetPreprocessor()

    processor.preprocess_spokenwoz()
    processor.preprocess_spoken_squad()