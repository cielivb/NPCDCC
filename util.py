""" Utility functions 
These functions are not required in the performance testing pipeline but are
used in Jupyter notebook 
"""
import numpy as np
import pandas as pd
from datetime import datetime
from dask import dataframe as ddf
import fast_hdbscan


def do_hdbscan(connectome, minsize):
    """ Attach HBSCAN cluster IDs to connectome 
    Requires computing all XYZ coordinates in the connectome file, so may scale
    poorly to larger datasets, but will work okay on the 915 MB parquet dataset.
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Doing HDBSCAN ...")
    # Do topological coordinate clustering
    coords = connectome[["x", "y", "z"]].to_dask_array(lengths=True).compute()
    clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=30, 
                                     cluster_selection_epsilon=0.5)
    cluster_ids = clusterer.fit_predict(coords)
    
    # To avoid index alignment issues, compute the connectome index, make a pd
    # df using cluster_ids with the same index as connectome, then turn into a 
    # dask dataframe and merge back onto connectome. The cluster_ids df has the 
    # same number of rows as the connectome df so shouldn't misalign.
    index = connectome.index.compute()
    id_df = pd.DataFrame({"pre": index.get_level_values(0), 
                          "post": index.get_level_values(1), 
                          "hdbscan_id": cluster_ids})
    id_ddf = ddf.from_pandas(id_df, npartitions=connectome.npartitions)
    connectome = connectome.merge(
        id_ddf, left_index=True, right_on=["pre","post"], how="inner")

    # Replace noise (-1 labels) with NaN
    connectome["hdbscan_id"] = connectome["hdbscan_id"].replace(-1, np.nan)
    
    print(f"{datetime.now().strftime("%H:%M:%S")} HDBSCAN complete")
    return connectome.persist()
    
    
def downsample(connectome, n):
    """ Sample approximately n rows from connectome dataframe. 
    Useful for keeping RAM in check during plotting and stats.
    """
    row_count = connectome.shape[0].compute()
    if row_count > n:
        return connectome
    frac = n / row_count
    sample = connectome.sample(frac=frac).persist()
    return sample
    