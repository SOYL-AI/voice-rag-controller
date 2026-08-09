import pandas as pd

file = "spoken_squad/data/test-00000-of-00021.parquet"

df = pd.read_parquet(file)

print(df.columns)

print()

print(df.iloc[0])