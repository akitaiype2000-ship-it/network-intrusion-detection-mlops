import pyarrow.parquet as pq

table = pq.read_table(
    "data/featured/selected_features.parquet",
    columns=None
)

df = table.to_pandas().sample(
    n=self.config.sample_size,
    random_state=42
)