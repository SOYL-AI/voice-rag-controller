import json
from pathlib import Path

import pandas as pd

from loader import DatasetLoader


OUTPUT_DIR = Path(__file__).parent / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)


class BenchmarkBuilder:

    def __init__(self):
        self.loader = DatasetLoader()

    def build_spokenwoz(self):

        data = self.loader.load_spokenwoz()

        benchmark = []

        for dialogue_id, dialogue in data.items():

            history = []

            for turn in dialogue["log"]:

                text = turn.get("text", "").strip()

                if not text:
                    continue

                benchmark.append({
                    "dataset": "spokenwoz",
                    "dialogue_id": dialogue_id,
                    "speaker": turn.get("tag"),
                    "query": text,
                    "history": history.copy()
                })

                history.append(text)

        return benchmark

    def build_spoken_squad(self):

        squad_folder = self.loader.load_spoken_squad()

        benchmark = []

        parquet_files = sorted(squad_folder.glob("*.parquet"))

        for file in parquet_files:

            df = pd.read_parquet(file)

            for _, row in df.iterrows():

                benchmark.append({

                    "dataset": "spoken_squad",

                    "query": row["instruction"],

                    "answer": row["answer"],

                    "history": []

                })

        return benchmark

    def build(self):

        benchmark = []

        benchmark.extend(self.build_spokenwoz())

        benchmark.extend(self.build_spoken_squad())

        output = OUTPUT_DIR / "benchmark.json"

        with open(output, "w", encoding="utf-8") as f:

            json.dump(benchmark, f, indent=2)

        print(f"Saved {len(benchmark)} benchmark samples.")

if __name__ == "__main__":

    builder = BenchmarkBuilder()

    builder.build()