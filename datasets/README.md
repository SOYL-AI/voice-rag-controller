# Public Benchmark Datasets

This project uses two public datasets for benchmarking:

- SpokenWOZ
- Spoken-SQuAD

## Expected directory structure

```text
datasets/
├── spokenwoz/
│   └── data.json
├── spoken_squad/
│   └── data/
```

## Setup

1. Download the SpokenWOZ dataset.
2. Place `data.json` inside `datasets/spokenwoz/`.
3. Download the Spoken-SQuAD dataset.
4. Place its extracted `data/` folder inside `datasets/spoken_squad/`.
5. Run the preprocessing/benchmark scripts.

The loader validates this directory structure and will display a helpful error if the datasets are missing.