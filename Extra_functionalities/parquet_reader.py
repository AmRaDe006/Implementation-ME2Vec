import pandas as pd

# replace `file_path` with the path to your actual .parquet file
file_path = "saved_data/spec_init_emb.parquet"

# read the .parquet file into a pandas DataFrame
df = pd.read_parquet(file_path)

# view the entire DataFrame
print(df)
