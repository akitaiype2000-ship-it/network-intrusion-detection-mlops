import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

writer = None

for chunk in pd.read_csv(
    "data/featured/selected_features.csv",
    chunksize=250000,
    low_memory=False
):
    # Convert every numeric column to float64
    for col in chunk.columns:
        if col != "event_timestamp":
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float64")

    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            "data/featured/selected_features.parquet",
            table.schema
        )

    writer.write_table(
        table,
        row_group_size=250000
    )

writer.close()
print("Done!")