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
import argparse
import dask
import numpy as np
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


CLIENT = None # Assigned in start_cluster()

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULT_DIR = os.path.join(ROOT_DIR, "results")


def parse_args():
    """ Process user input """
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--cores", "Number of cores to use", type=int, required=True)
    p.add_argument("-f", "--file", "Parquet file to run through pipeline", required=True)
    p.add_argument("-m", "--min", "Minimum community size", type=int, default=30)
    p.add_argument("-k", "--madk", "MAD outlier detection K", type=float, default=2.5)
    args = p.parse_args()
    return args


def create_session_id(file, num_cores):
    """ Derive session id from filename, number of cores, and datetime """
    datetime_id = datetime.now().strftime("%Y%m%d%H%M")
    filename = os.path.basename(file)
    if "_" in filename:
        filename = "full"
    session_id = f"{datetime_id}_{num_cores}_{filename}"
    return session_id


def start_cluster(num_cores):
    """ Create dask client and start cluster with 1 worker per core """
    global CLIENT
    cluster = LocalCluster(
        n_workers=num_cores,
        processes=True,
        threads_per_worker=1,
        memory_limit = "2GB",
        dashboard_address=":8787"
    )
    CLIENT = Client(cluster)
    dask.config.set({"dataframe.shuffle.method": "tasks"})    


def load_connectome(file) -> ddf.DataFrame:
    """ Load parquet connectome file into dask dataframe """
    connectome = ddf.read_parquet(file).persist()
    return connectome


def write_tagged_connectome(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Write tagged connectome to feather file/s """
    outfile = os.path.join(outdir, "tagged.parquet")
    tagged_connectome.to_parquet(outfile)


def write_cluster_data(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Generate and write the following metrics for each cluster:
    
    Cluster ID, number of nodes, number of edges, number of synapses, 
    dominant neuropil, neuropil purity (i.e., the largest proportion of edges in
    a given neuropil), average GABA probabilities + variance, average acetylcholine 
    probabilities + variance, average other neurotransmitter probabilities +
    variance. 
    """
    raise NotImplementedError


def write_neuropil_data(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Generate and write the following metrics for each neuropil: 
    
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


def do_stats(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Run a statistical analysis on the clusters and report the results. 
    Only clusters with neuropil purity >= 98% will be used in brain region
    comparisons.
    """
    raise NotImplementedError


def make_graphs(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Generate supporting bar charts """
    raise NotImplementedError


def make_brain_maps(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Use PyVista to generate before and after brain map images. """
    raise NotImplementedError






################################### MAIN #######################################

def report_duration(session_id, duration):
    """ Write duration to duration file """
    global RESULT_DIR
    duration_file = os.path.join(RESULT_DIR, "durations.txt")
    with open(duration_file, 'a') as file:
        file.write(f"{session_id}: {duration}\n")


def main():
    """ Run the full statistical analysis pipeline from loading to reporting """
    # Set-up performance testing stuff
    args = parse_args()
    session_id = create_session_id(args.file, args.cores)
    outdir = os.path.join(RESULT_DIR, session_id)
    os.mkdir(outdir)
    
    # Start timing
    start_time = datetime.now()
    
    # Set up cluster, load data, and identify clusters
    start_cluster(args.cores)
    connectome = load_connectome(args.file)
    tagged = detect_communities.run(connectome, args.min, args.madk)
    
    # Write tagged data, perform analyses, and generate visuals. None of these
    # tasks depend on the completion of any other of these tasks.
    w1_f = CLIENT.submit(write_tagged_connectome, tagged, outdir)
    w2_f = CLIENT.submit(write_cluster_data, tagged, outdir)
    w3_f = CLIENT.submit(write_neuropil_data, tagged, outdir)
    stats_f = CLIENT.submit(do_stats, tagged, outdir)
    graphs_f = CLIENT.submit(make_graphs, tagged, outdir)
    bm_f = CLIENT.submit(make_brain_maps, tagged, outdir)
    futures = [w1_f, w2_f, w3_f, stats_f, graphs_f, bm_f]
    status = futures.gather() # Block until are tasks are done
    
    # Stop timing and report duration
    end_time = datetime.now()
    duration = end_time - start_time
    report_duration(session_id, duration)


if __name__ == "__main__":
    main()