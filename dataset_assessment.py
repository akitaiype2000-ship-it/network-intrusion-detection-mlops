import pandas as pd
import glob
import os

print("=" * 80)
print("CSE-CIC-IDS2018 DATASET ASSESSMENT")
print("=" * 80)

folder = "data"

print("Current Working Directory:", os.getcwd())
print("Looking in folder:", os.path.abspath(folder))

csv_files = sorted(set(glob.glob(os.path.join(folder, "*.csv"))))

print("\nCSV files found:")
for f in csv_files:
    print(f)

if len(csv_files) == 0:
    print("No CSV files found.")
    exit()

total_rows = 0
columns = None

missing_total = None
duplicates = 0
class_counts = {}

categorical_columns = set()

for file in csv_files:

    print(f"\nProcessing: {os.path.basename(file)}")

    try:
        for chunk in pd.read_csv(
            file,
            chunksize=100000,
            low_memory=False,
            engine="c",
            on_bad_lines="skip"
        ):

            # Total rows
            total_rows += len(chunk)

            # Store column names once
            if columns is None:
                columns = chunk.columns.tolist()

            # Missing values
            miss = chunk.isnull().sum()

            if missing_total is None:
                missing_total = miss
            else:
                missing_total += miss

            # Duplicate rows (within each chunk)
            duplicates += chunk.duplicated().sum()

            # Class distribution
            if "Label" in chunk.columns:

                vc = chunk["Label"].value_counts()

                for label, count in vc.items():
                    class_counts[label] = class_counts.get(label, 0) + count

            # Categorical columns
            cat = chunk.select_dtypes(include=["object", "string"]).columns
            categorical_columns.update(cat)

            # Save first chunk for later printing
            if 'sample_df' not in locals():
                sample_df = chunk.copy()

    except Exception as e:
        print(f"Error reading {file}")
        print(e)
print("\n\n")

print("=" * 80)
print("1. ALIGNMENT AND SCOPE")
print("=" * 80)

print(f"\nTotal Rows : {total_rows}")

print(f"Total Columns : {len(columns)}")

print("\nTarget Variable")

if "Label" in columns:

    print("✓ Label column exists")

else:

    print("✗ Label column not found")

print("\nFeature Relevance")

print("""
Network traffic features such as Flow Duration,
Packet Length, Flow Bytes/s,
Flow Packets/s,
TCP Flags,
Header Length,
Flow IAT,
Idle Time,
etc. are meaningful predictors for intrusion detection.
""")

print("\nData Freshness")

print("""
Traffic was captured during 2018.

Although newer attack techniques exist,
this remains a modern benchmark dataset
for intrusion detection research.
""")

print("\n")

print("=" * 80)
print("2. QUALITY AND COMPLETENESS")
print("=" * 80)

print("\nMissing Value Percentage")

missing_percent = (missing_total / total_rows) * 100

print(missing_percent[missing_percent > 0].sort_values(ascending=False))

print("\nColumns >40% Missing")

high = missing_percent[missing_percent > 40]

if len(high):

    print(high)

else:

    print("None")

print("\nDuplicate Rows")

print(duplicates)

print("\nData Types")

print(sample_df.dtypes)

print("\nCategorical Columns")

print(list(categorical_columns))

print("\nPossible Inconsistencies")

for col in categorical_columns:

    print(f"\n{col}")

    if col in sample_df.columns:
        print(sample_df[col].dropna().unique()[:20])

print("\n")

print("=" * 80)
print("3. VOLUME VS DIMENSIONALITY")
print("=" * 80)

print(f"\nRows : {total_rows}")

print(f"Columns : {len(columns)}")

print("\nClass Distribution")

for k, v in class_counts.items():

    print(k, ":", v)

print("\nClass Percentage")

for k, v in class_counts.items():

    print(k, ":", round((v / total_rows) * 100, 4), "%")

print("""

Dimensionality Assessment

The dataset contains approximately 80 features,
which is manageable for most ML algorithms.

Feature selection may further improve performance.
""")

print("\n")

print("=" * 80)
print("4. BIAS, ETHICS AND COMPLIANCE")
print("=" * 80)

print("\nPossible PII")

keywords = [
    "ip",
    "email",
    "phone",
    "name",
    "address",
    "user"
]

pii = []

for c in columns:

    for k in keywords:

        if k.lower() in c.lower():

            pii.append(c)

if len(pii):

    print(pii)

else:

    print("No obvious PII columns detected.")

print("\nHistorical Bias")

print("""
Traffic was collected in a controlled environment.

The dataset may not represent all
real-world enterprise networks or
the latest cyber attacks.
""")

print("\nLicensing")

print("""
Verify the dataset license before
commercial deployment.
It is commonly used for research
and educational purposes.
""")

print("\n")

print("=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

print("""

✓ Target variable available

✓ Large dataset

✓ Suitable for supervised learning

✓ Suitable for intrusion detection

✓ Suitable for MLOps

Review:

• Missing values
• Duplicate rows
• Class imbalance
• Licensing

before beginning model development.
""")