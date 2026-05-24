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
p.add_argument("-m", "--min", help="Minimum number of edges", type=int, default=30)
p.add_argument("-k", "--madk", help="MAD outlier detection K", type=float, default=2.5)
ARGS = p.parse_args()

logging.basicConfig(level = logging.DEBUG)
LOGGER = logging.getLogger(__name__)
logging.getLogger("distributed.shuffle").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

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


def write_community_data(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Generate and write the following metrics for each community:
    
    Community ID, number of nodes, number of edges, number of synapses, 
    dominant neuropil, neuropil purity (i.e., the largest proportion of edges in
    a given neuropil), average GABA probabilities + variance, average acetylcholine 
    probabilities + variance, average other neurotransmitter probabilities +
    variance. 
    
    Tagged connectome has columns : pre, post, syn_count, neuropil, gaba_avg,
    ach_avg, glut_avg, oct_avg, ser_avg, da_avg, community_id
    
    """
    df = tagged_connectome
    
    easy_aggs = {"pre": "count", "syn_count": "sum",
                 "gaba_avg": ["mean", "var", "min", "max"],
                 "ach_avg": ["mean", "var", "min", "max"],
                 "glut_avg": ["mean", "var", "min", "max"],
                 "oct_avg": ["mean", "var", "min", "max"],
                 "ser_avg": ["mean", "var", "min", "max"],
                 "da_avg": ["mean", "var", "min", "max"]}
    group1 = df.groupby("community_id").agg(easy_aggs).rename(
        columns={"num_edges": "pre"})
    
    # Calculate dominant neuropil - the neuropil that accounts for the largest
    # proportion of edges in the community, and calculate neuropil purity, the 
    # maximum proportion of the community's edges belonging to a single neuropil,
    # i.e., the proportion of edges of the community belonging to the dominant 
    # neuropil.
    def get_dominant_neuropil(df: pd.DataFrame):
        total_edges = df["num_edges"].sum()
        max_edges_i = df["num_edges"].idxmax()
        dominant_neuropil = df.loc[max_edges_i]["neuropil"]
        prop = df.loc[max_edges_i]["num_edges"] / total_edges # purity
        return pd.from_dict({"community_id": community_id, 
                             "dominant": dominant_neuropil, 
                             "purity": prop})
    neuropil_counts = df.groupby(
        ["community_id", "neuropil"])["pre"].count().rename("num_edges")
    group2 = neuropil_counts.groupby("community_id").apply(get_dominant_neuropil)
    
    # Get number of nodes/neurons per community
    communities = list(df["community_id"].drop_duplicates().compute())
    @delayed
    def get_num_nodes_by_community(community_id):
        subset = df[df["community_id"] == community_id]
        num_nodes = detect_communities.get_num_nodes(subset) # Returns computed int
        return pd.from_dict({"community_id": community_id, "num_nodes": num_nodes})
    tasks = [get_num_nodes_by_community(comm) for comm in communities]
    group3 = ddf.from_delayed(tasks)
    
    # Merge the 3 groups then write to csv file
    result = group1.merge(group2, on="community_id", how="inner")
    result = result.merge(group3, on="community_id", how="inner")
    outfile = os.path.join(outdir, "summary_stats_communities.csv")
    result.to_csv(outfile, write_index=True)


def write_neuropil_data(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Generate and write the following metrics for each neuropil: 
    
    Neuropil, number of communities, number of edges assigned to communities, number
    of synapses assigned to communities, number of edges not assigned to communities,
    number of synapses not assigned to communities, percentage of edges assigned
    to communities, percentage of edges not assigned to communities, percentage of
    synapses assigned to communities, percentage of synapses not assigned to
    communities, average GABA probs + var for community edges vs other edges,
    average acetylchole probs + var for community edges vs other edges, and
    average other neurotransmitter probs + var for community vs other edges.
    
    Neuropils with L and R portions should have L and R portion summary stats
    in addition to collated summary stats.
    
    Tagged connectome has columns : pre, post, syn_count, neuropil, gaba_avg,
    ach_avg, glut_avg, oct_avg, ser_avg, da_avg, community_id
    
    """
    df = tagged_connectome
    df["assigned"] = df[df["community_id"].notna()]
    df["base_neuropil"] = df["neuropil"].str.split("_").str[0]
    df["hemisphere"] = df["neuropil"].str.split("_").str[1] # L or R
    
    # Get overall neuropil summary statistics
    nt_cols = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]
    easy_aggs = {"pre": "count", "syn_count": "sum", 
                   **{nt: ["mean", "var"] for nt in nt_cols}}
    overall_stats = df.groupby("neuropil").agg(easy_aggs)
    
    # Get community and bridge stats
    comm_stats = df[df["assigned"]].groupby("neuropil").agg(
        base_aggs).add_prefix("comm_")
    bridge_stats = df[~df["assigned"]].groupby("neuropil").agg(
        base_aggs).add_prefix("bridge_")
    
    # Merge overall stats, community stats, and bridge stats columnwise
    stat_df = ddf.concat([overall_stats, comm_stats, bridge_stats], axis=1)
    
    # Compute proportion of edges and synapses assigned to communities
    stat_df["prop_edges_comm"] = stat_df["comm_pre_count"] / stat_df["pre_count"]
    stat_df["prop_synapses_comm"] = stat_df["comm_syn_count_sum"] / stat_df["syn_count_sum"]

    # Aggregate left and right hemisphere neuropils into single neuropil summary
    # and add to stat_df
    combined = df.groupby("base_neuropil").agg(easy_aggs)
    stat_df = ddf.concat([stat_df, combined])

    # Write to file
    outfile = os.path.join(outdir, "summary_stats_neuropils.csv")    
    stat_df.to_csv(outfile, write_index=True)

    


def do_stats(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Run a statistical analysis on the communities and report the results. 
    Only communities with neuropil purity >= 98% will be used in brain region
    comparisons.
    """
    raise NotImplementedError


def make_graphs(tagged_connectome: ddf.DataFrame, outdir: str):
    """ Generate supporting bar charts """
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
    global ARGS
    # Set-up testing / debugging stuff
    session_id = create_session_id(ARGS.file, ARGS.cores)
    outdir = os.path.join(RESULT_DIR, session_id)
    os.makedirs(outdir, exist_ok = True)
    initialise_log_file(outdir)
    start_time = datetime.now() # Start timing actual pipeline
    
    # Set up cluster, load data, and identify communities
    start_cluster(ARGS.cores)
    connectome = load_connectome(ARGS.file)
    tagged = detect_communities.run(connectome, ARGS)
    
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
    
    LOGGER.info("Making brain map ...")
    make_brain_map.make_brain_map(tagged, coord_dir, outdir)
    
    # Write tagged data, perform analyses, and generate visuals. None of these
    # tasks depend on the completion of any other of these tasks.
    #w1_f = CLIENT.submit(write_tagged_connectome, tagged, outdir)
    #w2_f = CLIENT.submit(write_community_data, tagged, outdir)
    #w3_f = CLIENT.submit(write_neuropil_data, tagged, outdir)
    #stats_f = CLIENT.submit(do_stats, tagged, outdir)
    #graphs_f = CLIENT.submit(make_graphs, tagged, outdir)
    #bm_f = CLIENT.submit(make_brain_maps, tagged, outdir)
    #futures = [w1_f, w2_f, w3_f, stats_f, graphs_f, bm_f]
    #status = CLIENT.gather(futures) # Block until are tasks are done
    
    # Stop timing and report duration
    LOGGER.info(f"End of pipeline! Results available in {outdir}")
    end_time = datetime.now()
    duration = end_time - start_time
    report_duration(session_id, duration)


if __name__ == "__main__":
    main()