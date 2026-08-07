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
                f"SpokenWOZ data.json not found at {data_file}"
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
                f"Spoken-SQuAD data folder not found at {data_folder}"
            )

        return data_folder


if __name__ == "__main__":

    loader = DatasetLoader()

    spokenwoz = loader.load_spokenwoz()
    print(f"Loaded SpokenWOZ conversations: {len(spokenwoz)}")

    spoken_squad = loader.load_spoken_squad()
    print(f"Spoken-SQuAD folder: {spoken_squad}")