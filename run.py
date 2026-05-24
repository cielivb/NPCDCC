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
import os
import pandas as pd
from dask import bag as db
from dask import dataframe as ddf
from dask import delayed
from dask.distributed import Client
from dask.distributed import LocalCluster
from datetime import datetime
from time import sleep

import detect_communities
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
logging.getLogger("distributed.shuffle").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)



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
    """ Create dask client and start cluster with 1 worker per core """
    global CLIENT
    LOGGER.info("Starting cluster ...")    
    cluster = LocalCluster(
        n_workers=num_cores,
        processes=True,
        threads_per_worker=1,
        memory_limit = "2GB",
        dashboard_address=":8787"
    )
    CLIENT = Client(cluster)




### File I/O --------------------------------------------------------------
    

def load_connectome(file) -> ddf.DataFrame:
    """ Load parquet connectome file into dask dataframe """
    LOGGER.info(f"Loading connectome ({file}) ...")
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
        file, columns=["pre_pt_root_id", "post_pt_root_id", 
                       "pre_pt_position_x", "pre_pt_position_y", 
                       "pre_pt_position_z", "post_pt_position_x", 
                       "post_pt_position_y", "post_pt_position_z"])
    edge_coords = edge_coords.rename(cols=["pre","post"])
    return edge_coords


### Get communities -------------------------------------------------------
    
def attach_community_ids(connectome, minsize):
    """ Use Leiden algorithm """
    # Use a streaming approach instead of computing directly to avoid a sudden 
    # RAM spike (probably not too big of a deal with my dataset but could be
    # helpful for a larger dataset, especially considering parquet is often
    # compressed and loading it into pandas uncompresses it)
    g = ig.Graph(directed=True)
    for partition in connectome.to_delayed():
        pdf = partition.compute()
        pre, post = pdf["pre"].tolist(), pdf["post"].tolist()
        syn_count = pdf["syn_count"].to_list()
        
        # Add nodes if required. This works because nodes/vertices must be and 
        # are labelled 0, 1, ..., n.
        max_node = max(max(pre), max(post))
        if max_node >= g.vcount():
            g.add_vertices(max_node + 1 - g.vcount())
            
        # Add edges then assign edge weights (=syn_count) to newly added edges
        g.add_edges(list(zip(pre, post)))
        g.es[-len(syn_count):]["weight"] = syn_count # es = 'edge sequence'
    
    # Run the Leiden community detection algorithm and extract communities
    communities = algorithms.leiden(g, weights=g.es["weight"])
    communities_list = communities.communities # List of lists
    
    # Turn sufficiently large lists/communities into dask dataframes
    community_dfs = []
    for community in communities_list:
        if len(community) >= minsize:
            new_df = ddf.from_pandas(pd.DataFrame(community, columns=["node_id"]))
            community_dfs.append(new_df)
    
    # Create dataframe of node_ids and community_ids
    comm_ids = range(0, len(community_dfs))
    for comm_df, comm_id in zip(community_dfs, comm_ids):
        comm_df["community_id"] = comm_id
    community_df = ddf.concat(community_dfs)
    
    # Merge community_df with connectome to tag edges with community IDs
    tagged = connectome.merge(
        community_df, left_index=True, right_on="node_id", how="left")
    
    return tagged
        


### Ops -------------------------------------------------------------------


def attach_coords(connectome):
    """ Merge connectome with coord dataframe and calculated midpoint coords 
    
    "synapses were identified with two points, one in each neuron" (Zenodo). 
    Take the mean of these two coordinates to use as true synapse coordinate.
    """
    coord_df = load_coord_file()
    merged = connectome.merge(coord_df, on=["pre","post"], how="inner")
    merged["x"] = merged["pre_pt_position_x"] + merged["post_pt_position_x"] / 2
    merged["y"] = merged["pre_pt_position_y"] + merged["post_pt_position_y"] / 2
    merged["z"] = merged["pre_pt_position_z"] + merged["post_pt_position_z"] / 2
    merged = merged.drop(cols=["pre_pt_position_x", "post_pt_position_x",
                               "pre_pt_position_y", "post_pt_position_y",
                               "pre_pt_position_z", "post_pt_position_z"])
    return merged

    
def normalise_neurotransmitter_probs(tagged):
    # Sum probabilities of neurotransmitters that are not inherently excitatory
    # or regulatory together - only interested in excitatory-inhibitory dynamics
    # in this analysis. The sums of neurotransmitter probabilities are sometimes
    # just a few decimal places out from being exactly 1, so normalise as well.
    LOGGER.info("Normalising neurotransmitter probabilities ...")
    other_nt = ["glut", "oct", "ser", "da"]
    other_sum = tagged[other_nt].sum(axis=1)
    tagged["other"] = other_sum
    tagged = tagged.drop(columns = other_nt)
    tagged["total_prob"] = tagged[["gaba", "ach", "other"]].sum(axis=1)
    tagged["gaba"] = tagged["gaba"] / tagged["total_prob"]
    tagged["ach"] = tagged["ach"] / tagged["total_prob"]
    tagged["other"] = tagged["other"] / tagged["total_prob"]
    return tagged



################################### MAIN #######################################

def report_duration(session_id, duration):
    """ Write duration to duration file """
    global RESULT_DIR
    duration_file = os.path.join(RESULT_DIR, "durations.txt")
    with open(duration_file, 'a') as file:
        file.write(f"{session_id}: {duration}\n")


def main():
    """ Run the full statistical analysis pipeline from loading to reporting """
    global ARGS
    # Set-up testing / debugging stuff
    session_id = create_session_id(ARGS.file, ARGS.cores)
    outdir = os.path.join(RESULT_DIR, session_id)
    os.makedirs(outdir, exist_ok = True)
    initialise_log_file(outdir)
    start_time = datetime.now() # Start timing actual pipeline
    
    # Start of pipeline
    connectome = load_connectome(ARGS.file).set_index("pre", drop = False)
    connectome = connectome.repartition(100)    
    trimmed = connectome[["pre", "post", "syn_count"]]
    tagged_connectome = attach_community_ids(trimmed, ARGS.min)
    # Probably another filtering step here?
    connectome = attach_coords(tagged_connectome)
    make_brain_map.make_brain_map(connectome)
    # End of pipeline

    
    # Stop timing and report duration
    LOGGER.info(f"End of pipeline! Results available in {outdir}")
    end_time = datetime.now()
    duration = end_time - start_time
    report_duration(session_id, duration)


if __name__ == "__main__":
    main()