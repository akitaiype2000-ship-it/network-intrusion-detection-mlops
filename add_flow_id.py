import pandas as pd

input_file = "data/featured/selected_features.csv"
output_file = "data/featured/selected_features_temp.csv"

chunksize = 100000
flow_id = 0
first_chunk = True

for chunk in pd.read_csv(input_file, chunksize=chunksize, low_memory=False):
    chunk.insert(0, "flow_id", range(flow_id, flow_id + len(chunk)))
    flow_id += len(chunk)

    chunk.to_csv(
        output_file,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )

    first_chunk = False
    print(f"Processed {flow_id:,} rows")

print("Done!")