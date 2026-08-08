import json
from pathlib import Path


BASE_DIR = Path(__file__).parent

SPOKENWOZ_DIR = BASE_DIR / "spokenwoz"
SPOKEN_SQUAD_DIR = BASE_DIR / "spoken_squad"


class DatasetLoader:
    def __init__(self):
        self.spokenwoz_path = SPOKENWOZ_DIR
        self.spoken_squad_path = SPOKEN_SQUAD_DIR

    def load_spokenwoz(self):
        """
        Loads SpokenWOZ dataset.
        Returns the raw JSON.
        """

        data_file = self.spokenwoz_path / "data.json"

        if not data_file.exists():
            raise FileNotFoundError(
                 "SpokenWOZ dataset not found.\n\n"
                f"Expected file:\n{data_file}\n\n"
                "Expected directory structure:\n"
                "datasets/\n"
                "└── spokenwoz/\n"
                "    └── data.json\n\n"
                "Please download/setup the SpokenWOZ dataset before running the benchmark."
            )

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def load_spoken_squad(self):
        """
        Returns the Spoken-SQuAD dataset directory.
        We'll preprocess it later.
        """

        data_folder = self.spoken_squad_path / "data"

        if not data_folder.exists():
            raise FileNotFoundError(
                 "Spoken-SQuAD dataset not found.\n\n"
                f"Expected directory:\n{data_folder}\n\n"
                "Expected directory structure:\n"
                "datasets/\n"
                "└── spoken_squad/\n"
                "    └── data/\n\n"
                "Please download/setup the Spoken-SQuAD dataset before running the benchmark."
            )

        return data_folder


if __name__ == "__main__":

    loader = DatasetLoader()

    spokenwoz = loader.load_spokenwoz()
    print(f"Loaded SpokenWOZ conversations: {len(spokenwoz)}")

    spoken_squad = loader.load_spoken_squad()
    print(f"Spoken-SQuAD folder: {spoken_squad}")