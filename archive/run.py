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
- flywire_synapses_783.feather

Dataset Citation (APA):
FlyWire Consortium. (2024). FlyWire Whole-brain Connectome Connectivity Data 
  (783.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10676866
  
  
--- USAGE

TODO



--- CONTENTS

TODO


"""
import argparse
import dask
import logging
import igraph as ig
import hdbscan
import numpy as np
import os
import pandas as pd
from cdlib import algorithms
from dask import bag as db
from dask import dataframe as ddf
from dask import delayed
from dask.distributed import Client
from dask.distributed import LocalCluster
from datetime import datetime
from time import sleep

import make_brain_map


CLIENT = None # Assigned in start_cluster()
dask.config.set({"dataframe.shuffle.method": "p2p"})

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULT_DIR = os.path.join(ROOT_DIR, "results")

p = argparse.ArgumentParser()
p.add_argument("-c", "--cores", help="Number of cores to use", type=int, required=True)
p.add_argument("-f", "--file", help="Parquet file to run through pipeline", required=True)
p.add_argument("-m", "--min", help="Minimum number of nodes", type=int, default=30)
ARGS = p.parse_args()

logging.basicConfig(level = logging.DEBUG)
LOGGER = logging.getLogger(__name__)
logging.getLogger("distributed.shuffle").setLevel(logging.ERROR)
logging.getLogger("fsspec").setLevel(logging.ERROR)



### Admin -----------------------------------------------------------------

def create_session_id(file, num_cores):
    """ Derive session id from filename, number of cores, and datetime """
    datetime_id = datetime.now().strftime("%Y%m%d%H%M")
    filename = os.path.basename(file.removesuffix(".parquet"))
    if "_" in filename:
        filename = "full"
    session_id = f"{datetime_id}_{num_cores}_{filename}"
    LOGGER.info(f"Session ID: {session_id}")
    return session_id


def initialise_log_file(outdir: str):
    """ Add file handler that maps to log file in output directory to logger """
    log_path = os.path.join(outdir, "log.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    

def start_cluster(num_cores):
    """ Create dask CLIENT and start cluster with 1 worker per core """
    global CLIENT
    LOGGER.info("Starting cluster ...")    
    CLIENT = Client(processes=False, threads_per_worker=num_cores)




### File I/O --------------------------------------------------------------
    

def load_connectome(file) -> ddf.DataFrame:
    """ Load parquet connectome file into dask dataframe """
    connectome = ddf.read_parquet(file)
    connectome = connectome.rename(columns = {"pre_pt_root_id": "pre",
                                              "post_pt_root_id": "post"})
    return connectome


def write_tagged_connectome(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Write tagged connectome to parquet file """
    outfile = os.path.join(outdir, "tagged.parquet")    
    LOGGER.info(f"Writing tagged connectome to {outfile}...")
    tagged_connectome.to_parquet(outfile)
    
    
def load_coord_file():
    """ Load in edge coordinate data from 12.7 GB parquet file """
    global DATA_DIR
    coord_file_path = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")
    edge_coords = ddf.read_parquet( # Don't read in neurotransmitter prob cols
        coord_file_path, columns=["pre_pt_root_id", "post_pt_root_id", 
                                  "pre_pt_position_x", "pre_pt_position_y", 
                                  "pre_pt_position_z", "post_pt_position_x", 
                                  "post_pt_position_y", "post_pt_position_z"])
    edge_coords = edge_coords.rename(
        columns={"pre_pt_root_id":"pre", "post_pt_root_id": "post"})
    edge_coords = edge_coords.repartition(partition_size="150MB")
    return edge_coords



### Detect topological clusters -------------------------------------------

def cluster(connectome):
    """ Turn each neural connection into a coord then cluster """
    # Extract coordinates as a NumPy array
    coord_array = connectome[["x", "y", "z"]].to_dask_array(lengths=True).compute()

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0.5)
    labels = clusterer.fit_predict(coord_array)

    # Attach cluster ids and replace noise (-1 labels) with NaN
    id_df = ddf.from_array(labels, columns=["hdbscan_id"]).reset_index(drop=True)
    connectome = connectome.assign(hdbscan_id=id_df["hdbscan_id"])
    connectome["hdbscan_id"] = connectome["hdbscan_id"].replace(-1, np.nan)

    return connectome.persist()


### Get communities -------------------------------------------------------

def stream_into_graph(connectome, mapping):
    # Use a streaming approach instead of computing directly to avoid a sudden 
    # RAM spike (probably not too big of a deal with my dataset but could be
    # helpful for a larger dataset, especially considering parquet is often
    # compressed and loading it into pandas uncompresses it)
    num_nodes = mapping.count().compute()["node_id"].item()
    # Safe to use undirected because interested in physical connections, 
    # not information flow
    g = ig.Graph(directed=False) 
    g.add_vertices(num_nodes)
    g.vs["name"] = list(range(num_nodes))  # let name = node id
    
    for partition in connectome.to_delayed():
        pdf = partition.compute()
        pre, post = pdf["pre_key"].tolist(), pdf["post_key"].tolist()
        syn_count = pdf["syn_count"].to_list()
            
        # Add edges then assign edge weights (=syn_count) to newly added edges
        g.add_edges(list(zip(pre, post)))
        g.es[-len(syn_count):]["weight"] = syn_count # es = 'edge sequence'
    return g


def get_communities_list(g, method):
    if method == "leiden":
        # Considered a more 'robust' version of louvain, less prone to over-
        # fitting and favours smaller communities
        communities = algorithms.leiden(g, weights=g.es["weight"])
    elif method == "louvain":
        # Typically favours larger communities at the expense of hiding legit
        # sub-communities by pulling them into one big agglomerated community
        communities = algorithms.louvain(g, weight="weight")
    communities_list = communities.communities # List of lists
    return (method, communities_list)


def make_community_df(community_tup, minsize):
    # Turn sufficiently large lists/communities into dask dataframes
    method, communities_list = community_tup[0], community_tup[1]
    community_dfs = []
    for community in communities_list:
        if len(community) >= minsize:
            new_df = ddf.from_pandas(pd.DataFrame(community, columns=["node_id"]))
            community_dfs.append(new_df)
    
    # Create dataframe of node_ids and community_ids
    comm_ids = range(0, len(community_dfs))
    for comm_df, comm_id in zip(community_dfs, comm_ids):
        comm_df[f"{method}_id"] = comm_id
    community_df = ddf.concat(community_dfs).persist()
    return community_df


def tag_connectome(connectome, communities, minsize):
    """ Attach community ID to each edge in connectome """
    global CLIENT
    community_df_f = [CLIENT.submit(make_community_df, communities[0], minsize),
                      CLIENT.submit(make_community_df, communities[1], minsize)]
    community_dfs = CLIENT.gather(community_df_f)
    
    # Merge community_dfs with connectome to tag edges with community IDs
    tagged1 = connectome.merge(
        community_dfs[0], left_on="pre", right_on="node_id", how="left").drop(
            columns=["pre_key", "post_key"]).persist()
    tagged2 = tagged1.merge(
        community_dfs[1], left_on="pre", right_on="node_id", how="left").drop(columns = ["node_id_x", "node_id_y"]).persist()
    return tagged2


def attach_community_ids(connectome, mapping, minsize):
    """ Use Louvain and Leiden algorithms - not sure which will be better """
    global CLIENT
    g = stream_into_graph(connectome, mapping)
    get_comms_f = [CLIENT.submit(get_communities_list, g, "louvain"),
                   CLIENT.submit(get_communities_list, g, "leiden")]
    communities_lists = CLIENT.gather(get_comms_f) # list of tuples (method, list of lists)
    tagged = tag_connectome(connectome, communities_lists, minsize)
    return tagged



### Ops -------------------------------------------------------------------

def get_all_node_ids(df, node_cols=["pre","post"]):
    """ Return a dataframe containing every unique node id in the dataframe """
    node_cols = [df[col].rename("node_id").to_frame() for col in node_cols]
    all_nodes = ddf.concat(node_cols).drop_duplicates().repartition(partition_size="150MB").reset_index(drop=True)
    return all_nodes


def map_nodes(connectome):
    """ Create a mapping from connectome node IDs to IDs of form 0, 1, 2, ... 
    The Leiden algorithm requires node IDs to be in contiguous form
    starting from node 0. The node IDs in the connectome dataset neither start
    from zero nor are contiguous.
    """
    # Get key-node_id mapping
    pdf = get_all_node_ids(connectome).compute().reset_index(drop=True)
    pdf["key"] = range(len(pdf))
    mapping = ddf.from_pandas(pdf, npartitions=connectome.npartitions).persist()
    
    merged = connectome.merge(mapping, left_on="pre", right_on="node_id", how="inner").persist()
    merged = merged.rename(columns={"key": "pre_key"}).drop(columns=["node_id"]).persist()
    merged = merged.merge(mapping, left_on="post", right_on="node_id", how="inner").persist()
    merged = merged.rename(columns={"key": "post_key"}).drop(columns=["node_id"]).persist()
    return merged.reset_index(drop=True).persist(), mapping


def attach_coords(connectome):
    """ Merge connectome with coord dataframe and calculated midpoint coords 
    
    "synapses were identified with two points, one in each neuron" (Zenodo). 
    Take the mean of these two coordinates to use as true synapse coordinate.
    """
    coord_df = load_coord_file()
    merged = connectome.merge(coord_df, on=["pre","post"], how="inner").persist()
    merged["x"] = merged["pre_pt_position_x"] + merged["post_pt_position_x"] / 2
    merged["y"] = merged["pre_pt_position_y"] + merged["post_pt_position_y"] / 2
    merged["z"] = merged["pre_pt_position_z"] + merged["post_pt_position_z"] / 2
    merged = merged.drop(columns=["pre_pt_position_x", "post_pt_position_x",
                                  "pre_pt_position_y", "post_pt_position_y",
                                  "pre_pt_position_z", "post_pt_position_z"])
    return merged.persist()

    
def normalise_neurotransmitter_probs(tagged):
    """ Sum probabilities of neurotransmitters that are not inherently excitatory
    or regulatory together - only interested in excitatory-inhibitory dynamics
    in this analysis. The sums of neurotransmitter probabilities are sometimes
    just a few decimal places out from being exactly 1, so normalise as well.
    Removes edges with all NaN neurotransmitter probabilities. """
    print(tagged)    
    tagged = tagged.rename(
        columns={"gaba_avg": "gaba", "ach_avg": "ach", "glut_avg": "glut", 
                 "oct_avg": "oct", "ser_avg": "ser", "da_avg": "da"})
    other_nt = ["glut", "oct", "ser", "da"]
    other_sum = tagged[other_nt].sum(axis=1)
    tagged["other"] = other_sum
    tagged = tagged.drop(columns = other_nt)
    tagged["total_prob"] = tagged[["gaba", "ach", "other"]].sum(axis=1)
    tagged["gaba"] = tagged["gaba"] / tagged["total_prob"]
    tagged["ach"] = tagged["ach"] / tagged["total_prob"]
    tagged["other"] = tagged["other"] / tagged["total_prob"]
    tagged = tagged.drop(columns=["total_prob"])
    tagged = tagged.dropna(subset=["gaba", "ach", "other"]) # Drop NaNs
    print(tagged)
    return tagged.persist()



################################### MAIN #######################################

def report(session_id, duration, minsize):
    """ Write performance data to performances file """
    global RESULT_DIR
    perf_file = os.path.join(RESULT_DIR, "performances.txt")
    with open(perf_file, 'a') as file:
        file.write(f"{session_id}: time = {duration}, minsize = {minsize}\n")


def main():
    """ Run the full statistical analysis pipeline from loading to reporting """
    global ARGS, CLIENT
    # Set-up testing / debugging stuff
    session_id = create_session_id(ARGS.file, ARGS.cores)
    outdir = os.path.join(RESULT_DIR, session_id)
    os.makedirs(outdir, exist_ok = True)
    initialise_log_file(outdir)
    start_time = datetime.now() # Start timing actual pipeline
    
    # Start of pipeline
    start_cluster(ARGS.cores)
    LOGGER.info("Loading connectome ...")
    connectome = load_connectome(ARGS.file)
    LOGGER.info("Repartitioning connectome ...")
    connectome = connectome.repartition(partition_size="150MB")
    LOGGER.info("Attaching coordinates ...")
    connectome = attach_coords(connectome)    
    #LOGGER.info("Mapping nodes ...")
    #mapped_connectome, mapping = map_nodes(connectome)
    
    LOGGER.info("Clustering  ...")
    tagged_connectome = cluster(connectome)
    #LOGGER.info("Getting community IDs ...")
    #tagged_connectome = attach_community_ids(
    #    mapped_connectome, mapping.persist(), ARGS.min)

    LOGGER.info("Normalising neurotransmitter probabilities ...")
    connectome = normalise_neurotransmitter_probs(connectome)
    
    # Now that communities have been assigned, it would be nice to cluster
    # communities together where they are topologically close to each other,
    # but that is machine learning and thus outside the scope of this project.
    
    # Brain mapping heavy on GPU, and pyvista rendering is not threadsafe
    LOGGER.info("Preparing brain map plotter ...")
    plotter_louvain = make_brain_map.prepare_plotter(connectome, "louvain")
    plotter_leiden = make_brain_map.prepare_plotter(connectome, "leiden")
    LOGGER.info("Shutting down cluster ...")
    CLIENT.close() # disconnect from CLIENT
    LOGGER.info("Generating brain map ...")
    make_brain_map.save(plotter_louvain, "louvain", outdir)
    make_brain_map.save(plotter_leiden, "leiden", outdir)
    # End of pipeline


    # Stop timing and report duration
    LOGGER.info(f"End of pipeline! Results available in {outdir}")
    end_time = datetime.now()
    duration = end_time - start_time
    report(session_id, duration, ARGS.min)


if __name__ == "__main__":
    main()