import pandas as pd

df = pd.read_csv(
    "data/featured/selected_features.csv",
    nrows=1_000_000
)

df.to_parquet(
    "data/featured/training_sample.parquet",
    index=False
)

print("Done")