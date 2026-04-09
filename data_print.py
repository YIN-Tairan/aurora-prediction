import pandas as pd
import numpy as np

# Load the Parquet file
df = pd.read_parquet('omni_ready_for_pytorch.parquet')

# Print column titles (headers)
print("Column titles:", df.columns.tolist())

# Print the first 50 rows
print(df.head(50))

# Count rows containing inf or -inf
inf_count = df.isin([np.inf, -np.inf]).any(axis=1).sum()
nan_count = df.isin([np.nan]).any(axis=1).sum()
print(f"Number of rows containing inf or -inf: {inf_count}")
print(f"Number of rows containing NaN: {nan_count}")
print(f"Total rows: {len(df)}")

# Find positions of the first 50 NaN values
nan_mask = df.isna()
nan_positions = nan_mask.stack()
nan_indices = nan_positions[nan_positions].index.tolist()[:50]
print("Positions of the first 50 NaN values (row, column):")
for pos in nan_indices:
    print(pos)

# Check for values like 9999 or 99999
has_9999 = (df == 9999).any().any()
has_99999 = (df == 99999).any().any()
print(f"Contains 9999: {has_9999}")
print(f"Contains 99999: {has_99999}")