from pathlib import Path
import pandas as pd

raw = Path("data/raw")

dfs = {}

for file in sorted(raw.glob("*.csv")):
    df = pd.read_csv(file, nrows=1)
    dfs[file.name] = list(df.columns)

base = dfs["02-14-2018.csv"]

for name, cols in dfs.items():
    if len(cols) != len(base):
        print("\n", "=" * 60)
        print(name)

        extra = [c for c in cols if c not in base]
        missing = [c for c in base if c not in cols]

        print("Extra columns:")
        for c in extra:
            print("  ", c)

        print("\nMissing columns:")
        for c in missing:
            print("  ", c)