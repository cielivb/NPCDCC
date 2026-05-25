""" Main pipeline file - use for performance testing """

from datetime import datetime
import os

from dask import dataframe as ddf

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
COORD_FILE = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")
MAIN_FILE = os.path.join(DATA_DIR, "proofread_connections_783.parquet")


def load_connectome(file):
    """ Load parquet connectome file into dask dataframe """
    global MAIN_FILE
    if not file: file = MAIN_FILE
    print(f"{datetime.now().strftime("%H:%M:%S")} Loading connectome ...")
    connectome = ddf.read_parquet(file)
    connectome = connectome.rename(
        columns = {"pre_pt_root_id": "pre", "post_pt_root_id": "post",
                   "gaba_avg": "gaba", "ach_avg": "ach", "glut_avg": "glut",
                   "oct_avg": "oct", "ser_avg": "ser", "da_avg": "da"}).persist()
    print(f"{datetime.now().strftime("%H:%M:%S")} Connectome loaded")
    return connectome


def load_coord_file():
    """ Load in edge coordinate data from 12.7 GB parquet file """
    global DATA_DIR
    coord_file_path = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")
    edge_coords = ddf.read_parquet( # Don't read in neurotransmitter prob cols!
        coord_file_path, columns=["pre_pt_root_id", "post_pt_root_id", 
                                  "pre_pt_position_x", "pre_pt_position_y", 
                                  "pre_pt_position_z", "post_pt_position_x", 
                                  "post_pt_position_y", "post_pt_position_z"])
    edge_coords = edge_coords.rename(
        columns={"pre_pt_root_id":"pre", "post_pt_root_id": "post"})
    return edge_coords


def attach_coords(connectome):
    """ Add x, y, z coordinates to each connectome edge. 
    
    In the coordinate file, each edge has a column containing the 'pre' neuron's
    synapse coordinates, and a column containing the 'post' neuron's synapse 
    coordinates. There is an entry for every synapse. 
    
    This function first calculates the midpoint x,y,z coordinates for every 
    synapse, then attaches the average x,y,z coordinates across all synapses 
    for every neural connection to each edge in connectome.
    """
    coord_df = load_coord_file()
    print(f"{datetime.now().strftime("%H:%M:%S")} Attaching coordinates ...")
    merged = connectome.merge(coord_df, on=["pre","post"], how="inner").persist()
    
    # Get x, y, z coordinates for each synapse
    merged["x"] = merged["pre_pt_position_x"] + merged["post_pt_position_x"] / 2
    merged["y"] = merged["pre_pt_position_y"] + merged["post_pt_position_y"] / 2
    merged["z"] = merged["pre_pt_position_z"] + merged["post_pt_position_z"] / 2
    merged = merged.drop(columns=["pre_pt_position_x", "post_pt_position_x",
                                  "pre_pt_position_y", "post_pt_position_y",
                                  "pre_pt_position_z", "post_pt_position_z"])
    
    # Groupby 'pre' and 'post' then average x, y, z coordinates to get an 
    # approximate coordinate corresponding to a neural connection location.
    grouped = merged.groupby(
        ["pre", "post", "syn_count", "neuropil", "gaba", "ach", 
         "glut", "oct", "ser", "da"])[["x","y","z"]].mean().persist()
    
    print(f"{datetime.now().strftime("%H:%M:%S")} Coordinates attached")
    return grouped
    

def normalise_nt_probs(connectome):
    """ Ensure probabilities sum to 1 for each edge, discarding NaN rows """
    print(f"{datetime.now().strftime("%H:%M:%S")} Normalising neurotransmitter probabilities ...")
    other_cols = ["glut", "oct", "ser", "da"]
    connectome["other"] = connectome[other_cols].sum(axis=1)
    connectome["total_prob"] = connectome[["gaba", "ach", "other"]].sum(axis=1)
    connectome["gaba"] = connectome["gaba"] / connectome["total_prob"]
    connectome["ach"] = connectome["ach"] / connectome["total_prob"]
    connectome["other"] = connectome["other"] / connectome["total_prob"]
    connectome = connectome.drop(columns=["total_prob"])
    connectome = connectome.dropna(subset=["gaba", "ach", "other"]).persist() # Drop NaNs
    print(f"{datetime.now().strftime("%H:%M:%S")} Neurotransmitter probabilities normalised")
    return connectome
    