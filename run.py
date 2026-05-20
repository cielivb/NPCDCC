""" 
Computing Excitatory-Inhibitory Neurotransmitter Ratios of Drosophila 
Connectome Communities

Program Author: Ciel Baumann


--- DATA

Dataset: FlyWire Whole-brain Connectome Connectivity Data
Dataset Retrieved From: https://zenodo.org/records/10676866
Dataset Version: 783.0
Dataset Published By: Flywire Consortium

Data Files used:
- proofread_connections_783.feather
- proofread_root_ids_783.npy

Dataset Citation (APA):
FlyWire Consortium. (2024). FlyWire Whole-brain Connectome Connectivity Data 
  (783.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10676866
  
  
--- USAGE

TODO



--- CONTENTS

TODO


"""

import dask
import numpy as np
import os
import pandas as pd
from dask import bag as db
from dask import dataframe as ddf
from dask import delayed
from dask.distributed import Client
from dask.distributed import LocalCluster
from time import sleep

import detect_communities
import preprocess

CLIENT = None # Assigned at bottom of script

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MAIN_FILE = os.path.join(DATA_DIR, "proofread_connections_783.parquet")
COORD_FILE = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")



################################## LOAD ########################################

def read_parquet_w_id(path):
    """ Create dataframe from parquet file at path with attribute ID """
    file_id = os.path.splitext(os.path.basename(path))
    raw_df = ddf.read_parquet(path)
    raw_df.attrs["id"] = file_id
    return raw_df


def remove_self_edges(raw):
    """ Remove any edges to self - they are not needed for this analysis and I
     have not danger-proofed the pipeline from them """
    edges_to_remove = raw[raw["pre"] == raw["post"]]
    merged = raw.merge(edges_to_remove, on=["pre","post"], how="left", indicator=True)
    connectome = merged[merged["_merge"] == "left_only"].persist()
    connectome.id = raw.id
    return connectome


def load_connectomes() -> list[ddf.DataFrame]:
    """ Parse connectome feather files in data directory into dask dataframes """
    global FILES, CLIENT
    raw_dfs = [CLIENT.submit(read_parquet_w_id(f)) for f in FILES]
    raw_dfs = CLIENT.gather(raw_dfs)
    connectomes = [CLIENT.submit(remove_self_edges(df)) for df in raw_dfs]
    connectomes = CLIENT.gather(connectomes)
    print(connectomes[0].attrs)
    return connectomes





################################# ANALYSIS #####################################


def aggregate_cluster_data(tagged_connectome_df):
    """ Generate the following metrics for each cluster:
    
    Cluster ID, number of nodes, number of edges, number of synapses, 
    dominant neuropil, neuropil purity (i.e., the largest proportion of edges in
    a given neuropil), average GABA probabilities + variance, average acetylcholine 
    probabilities + variance, average other neurotransmitter probabilities +
    variance. 
    """
    raise NotImplementedError


def aggregate_neuropil_data(tagged_connectome_df):
    """ Generate the following metrics for each neuropil: 
    
    Neuropil, number of clusters, number of edges assigned to clusters, number
    of synapses assigned to clusters, number of edges not assigned to clusters,
    number of synapses not assigned to clusters, percentage of edges assigned
    to clusters, percentage of edges not assigned to clusters, percentage of
    synapses assigned to clusters, percentage of synapses not assigned to
    clusters, average GABA probs + var for clustered edges vs unclustered edges,
    average acetylchole probs + var for clustered vs unclustered edges, and
    average other neurotransmitter probs + var for clustered vs unclustered edges.
    
    Neuropils with L and R portions should have L and R portion summary stats
    in addition to collated summary stats.
    """
    raise NotImplementedError


def do_stats(aggregated):
    """ Run a statistical analysis on the clusters and report the results. 
    Only clusters with neuropil purity >= 98% will be used in brain region
    comparisons.
    """
    raise NotImplementedError


def make_graphs(aggregated):
    """ Generate supporting graphs """
    raise NotImplementedError







################################### MAIN #######################################

def needs_preprocessing():
    """ Check if data is ready for pipelining """
    global MAIN_FILE, COORD_FILE
    if not os.path.exists(MAIN_FILE) or not os.path.exists(COORD_FILE):
        return True
    return False


def main():
    """ Run the full statistical analysis pipeline from loading to reporting """
    global MAIN_FILE, COORD_FILE
    
    # Preprocess files (if applicable)
    if needs_preprocessing():
        print("Preprocessing datasets ...")
        preprocess.run(MAIN_FILE, COORD_FILE)
    print("Datasets ready for pipelining")

    # Load connectomes
    print("\nLoading connectomes ...")
    connectome_dfs = load_connectomes() # List of dataframes
    print(f"{len(connectome_dfs)} connectomes loaded into dask dataframes")
    
    # Identify clusters
    print(f"\nIdentifying clusters (this may take a while) ...")
    tagged = [CLIENT.submit(detect_clusters.run, df) for df in connectome_dfs]
    tagged = CLIENT.gather(tagged) # List of dataframes tagged with cluster IDs
    print("\nClusters identified; all dataframes tagged")
    for df in tagged:
        print(f"\n{df.id}")
        print(df.head(10))
        
    # Do statistical analyses
    print(f"Doing statistical analyses ...")
    do_stats(tagged_connectome_df)
    print(f"Statistical analyses complete")
    
    # Create supporting visuals
    print(f"Generating brain maps ...")
    make_graphs(tagged_connectome_df)
    print(f"Brain maps generated")


if __name__ == "__main__":
    # Set-up dask cluster and client.
    # My local machine has 16 GB RAM.
    cluster = LocalCluster(
        n_workers=4,
        processes=True,
        threads_per_worker=1,
        memory_limit = "3GB",
        dashboard_address=":8787"
    )
    CLIENT = Client(cluster)
    dask.config.set({"dataframe.shuffle.method": "tasks"})    
    main()