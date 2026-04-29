from dask import delayed
import pandas as pd
import dask.dataframe as dd
import os

FILE_DIR = os.path.dirname(__file__)

@delayed
def read_feather(path):
    return pd.read_feather(path, use_threads=True)

feather_path = os.path.join(FILE_DIR, "data", "proofread_connections_783.feather")
ddf = dd.from_delayed(read_feather(feather_path))
ddf.info(verbose=True, memory_usage=True)

# First attempt ~16 seconds on local machine to get to this point
# Second attempt onwards ~5 seconds

# Get neuropil list
neuropils = sorted(ddf['neuropil'].unique().compute().tolist())
print(neuropils)
print(len(neuropils))

