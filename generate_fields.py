import pandas as pd

df = pd.read_csv("data/featured/selected_features.csv", nrows=1)

for col in df.columns:
    if col not in ["Label", "event_timestamp"]:
        print(f'Field(name="{col}", dtype=Float64),')