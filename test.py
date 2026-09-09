import pandas as pd

df = pd.read_csv(
    "data/raw/02-14-2018.csv",
    nrows=5,
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

print(df.head())
print(df.shape)