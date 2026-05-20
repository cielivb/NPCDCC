""" Create test cases for performance analyses 

This is an adhoc script to subset the connectome feather file into several
differently sized test files.

"""
import os
import pandas as pd
from dask import dataframe as ddf
from dask import delayed

import detect_communities

ROOT_DIR = os.path.dirname(__file__)
FEATHER = os.path.join(ROOT_DIR, "data", "proofread_connections_783.feather")
OUT_DIR = os.path.join(ROOT_DIR, "data")
METADATA_FILE = os.path.join(OUT_DIR, "test_metadata.txt")


def load_connectome() -> ddf.DataFrame:
    """ Parse connectome feather file into dask dataframe """
    global FEATHER
    
    @delayed
    def read_feather(path):
        return pd.read_feather(path, use_threads=True)
    
    connectome = ddf.from_delayed(read_feather(FEATHER))
    return connectome


def write_metadata(sub_connectome, test_id):
    global METADATA_FILE
    """ Write test connectome file metadata 
    E.g., number of nodes, number of edges, node:edge ratio, neuropils """
    num_nodes = detect_communities.get_all_nodes(
        sub_connectome, 
        node_cols=["pre_pt_root_id","post_pt_root_id"]).count().compute()
    num_edges = sub_connectome["pre_pt_root_id"].count().compute()
    node_edge_ratio = num_nodes/num_edges
    neuropils = list(sub_connectome["neuropil"].unique().compute())
    with open(METADATA_FILE, "a") as mfile:
        mfile.write(f"Testfile Metadata: {test_id}.feather\n")
        mfile.write(f"Number of nodes: {num_nodes}\n")
        mfile.write(f"Number of edges: {num_edges}\n")
        mfile.write(f"Node-edge ratio: {node_edge_ratio}\n")
        mfile.write(f"Neuropils: {neuropils}\n\n")
    

def write_subset_file(sub_connectome, test_id):
    """ Write feather file containing sub-connectome. 
    These tests should fit in memory so am using non-streaming conversion. 
    Saving as feather file to match full dataset file type. """
    global OUT_DIR
    filename = os.path.join(OUT_DIR, f"{test_id}.feather")
    pd_connectome = sub_connectome.compute()
    pd_connectome.to_feather(filename)


def main():
    """ Subset connectome into 4 different test sizes .
    Tests are based on brain regions just in case there is a relatively high
    bridge edge to cluster edge ratio across the connectome, in which case
    random selection of test edges will likely yield non-representatively fast
    results compared to the algorithm's expected speed with real connectome data.
    """
    connectome = load_connectome()

    # Combine BU_L and BU_R to get BU with 1346+1989 = 3,335 total edges
    tiny = connectome[connectome["neuropil"].str.startswith("BU_")]
    
    # Combine LOP_L and LOP_R with 447405+613882 = 1,061,287 total edges
    small = connectome[connectome["neuropil"].str.startswith("LOP_")]
    
    # Combine ME_L and ME_R to get ME with 2426813+2598890 = 5,025,703 total edges
    medium = connectome[connectome["neuropil"].str.startswith("ME_")]    
    
    # Combine everything but ME_L, ME_R, LOP_L, & LOP_R for 10,761,007 total edges
    omit = {"ME_L", "ME_R", "LOP_L", "LOP_R"}
    large = connectome[~connectome["neuropil"].isin(omit)]
    
    # Write subset data
    subsets = [tiny,small,medium,large]
    test_ids = ["tiny","small","medium","large"]
    for sub_connectome, test_id in zip(subsets, test_ids):
        write_metadata(sub_connectome, test_id)
        write_subset_file(sub_connectome, test_id)


if __name__ == "__main__":
    main()