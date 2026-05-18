""" Create test cases for performance analyses 

This is an adhoc script to subset the connectome feather file into several
differently sized test files.

"""
import os
import pandas as pd
from dask import dataframe as ddf
from dask import delayed

ROOT_DIR = os.path.dirname(__file__)
FEATHER = os.path.join(ROOT_DIR, "data", "proofread_connections_783.feather")


def load_connectome() -> ddf.DataFrame:
    """ Parse connectome feather file into dask dataframe """
    global FEATHER
    
    @delayed
    def read_feather(path):
        return pd.read_feather(path, use_threads=True)
    
    connectome = ddf.from_delayed(read_feather(FEATHER))
    return connectome



def main():
    connectome = load_connectome()
    # View which brain regions have the most and fewest edges
    region_counts = connectome.groupby("neuropil").count().compute()
    print(region_counts.sort_values(by="pre_pt_root_id"))

if __name__ == "__main__":
    main()