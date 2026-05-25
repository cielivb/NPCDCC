""" Utility functions 
These functions are not required in the performance testing pipeline but are
used in Jupyter notebook 
"""

import hdbscan
from dask import dataframe as ddf


def hdbscan(connectome, minsize):
    """ Attach HBSCAN cluster IDs to connectome 
    Requires computing all XYZ coordinates in the connectome file, so may scale
    poorly to larger datasets, but will work okay on the 915 MB parquet dataset.
    """
    # Do topological coordinate clustering
    coords = connectome[["x", "y", "z"]].to_dask_array(lengths=True).compute()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=minsize,
                                cluster_selection_epsilon=0.5)
    cluster_ids = clusterer.fit_predict(coord_array)
    
    # Attach cluster ids and replace noise (-1 labels) with NaN
    id_df = ddf.from_array(cluster_ids, columns=["hdbscan_id"])
    id_df = id_df.reset_index(drop=True)
    connectome = connectome.assign(hdbscan_id=id_df["hdbscan_id"])
    connectome["hdbscan_id"] = connectome["hdbscan_id"].replace(-1, np.nan)
    

def louvain(connectome, minsize):
    """ Attach Louvain-derived community IDs to connectome 
    Must compute data to add to igraph and run Louvain algorithm
    """
    pass


def leiden(connectome, minsize):
    """..."""
    pass