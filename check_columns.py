from pathlib import Path
import pandas as pd

raw_folder = Path("data/raw")

print("=" * 60)
print("Checking CSV files...")
print("=" * 60)

for file in sorted(raw_folder.glob("*.csv")):
    try:
        df = pd.read_csv(file, nrows=5, low_memory=False)
        print(f"{file.name:<20} -> {len(df.columns)} columns")
    except Exception as e:
        print(f"{file.name:<20} -> ERROR: {e}")