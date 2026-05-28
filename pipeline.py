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
    connectome = ddf.read_parquet(file).repartition(partition_size="200MB")
    connectome = connectome.rename(
        columns = {"pre_pt_root_id": "pre", "post_pt_root_id": "post",
                   "gaba_avg": "gaba", "ach_avg": "ach", "glut_avg": "glut",
                   "oct_avg": "oct", "ser_avg": "ser", "da_avg": "da"}).persist()
    connectome["neuropil"] = connectome["neuropil"].cat.as_known()
    print(f"{datetime.now().strftime("%H:%M:%S")} Connectome loaded")
    return connectome


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
    connectome = connectome.drop(columns=other_cols)
    connectome = connectome.dropna().persist(subset=["gaba", "ach", "other"]) # Drop NaNs
    print(f"{datetime.now().strftime("%H:%M:%S")} Neurotransmitter probabilities normalised")
    return connectome


def load_coord_file():
    """ Load in edge coordinate data from 12.7 GB parquet file. 
    It is safe to omit neurotransmitter probabilities because each edge has
    the same neurotransmitter probability, and these probabilities have already
    been normalised in the preceeding normalise_nt_probs step.
    """
    global DATA_DIR
    coord_file_path = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")
    edge_coords = ddf.read_parquet( # Don't read in neurotransmitter prob cols!
        coord_file_path, columns=["pre_pt_root_id", "post_pt_root_id", 
                                  "pre_pt_position_x", "pre_pt_position_y", 
                                  "pre_pt_position_z", "post_pt_position_x", 
                                  "post_pt_position_y", "post_pt_position_z"])
    edge_coords = edge_coords.rename(
        columns={"pre_pt_root_id":"pre", "post_pt_root_id": "post"})
    edge_coords = edge_coords.repartition(partition_size="200MB")
    return edge_coords


def attach_synapse_coords(connectome):
    """ Add x, y, z coordinates for every synaptic connection in connectome.
    
    In the coordinate file, each synapse has a column containing the 'pre' neuron's
    synapse coordinates, and a column containing the 'post' neuron's synapse 
    coordinates.
    
    This function calculates the midpoint x,y,z coordinates for every 
    synapse, and uses the product as the synapse's coordinates.
    
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Attaching coordinates ...")    
    coord_df = load_coord_file()
    merged = connectome.merge(coord_df, on=["pre","post"], how="inner").persist()
    
    # Get x, y, z coordinates for each synapse
    merged["x"] = merged["pre_pt_position_x"] + merged["post_pt_position_x"] / 2
    merged["y"] = merged["pre_pt_position_y"] + merged["post_pt_position_y"] / 2
    merged["z"] = merged["pre_pt_position_z"] + merged["post_pt_position_z"] / 2
    merged = merged.drop(columns=["pre_pt_position_x", "post_pt_position_x",
                                  "pre_pt_position_y", "post_pt_position_y",
                                  "pre_pt_position_z", "post_pt_position_z",
                                  "syn_count"])
    
    print(f"{datetime.now().strftime("%H:%M:%S")} Coordinates attached")
    return merged
    

def condense(connectome):
    """ Extract edges and averaged synaptic xyz coords for use in HDBSCAN.
    The full set of synapses is not necessary for HDBSCAN. """
    print(f"{datetime.now().strftime("%H:%M:%S")} Condensing synapses to neural connections ...")
    # Groupby 'pre' and 'post' then average x, y, z coordinates to get an 
    # approximate coordinate corresponding to a neural connection location.
    condensed = connectome.groupby(["pre", "post"])[["x","y","z"]].mean().persist()
    print(f"{datetime.now().strftime("%H:%M:%S")} Condensed synapse coordinates to neural connections")
    return condensed


def extend(condensed, connectome):
    """ Assign synapses to same clusters as parent neurons.
    condensed contains 'pre','post','hdbscan_id'. 
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Extending cluster IDs to synapses ...")
    extended = connectome.merge(
        condensed, left_on=["pre", "post"], right_on=["pre", "post"], 
        how="inner")
    extended = extended.drop(columns = ["x_x", "y_x", "z_x"]).rename(
        columns = {"x_y": "x", "y_y": "y", "z_y": "z"}).persist()
    print(f"{datetime.now().strftime("%H:%M:%S")} Extended cluster IDs to synapses")
    return extended


def get_neuropil_summary_stats(connectome):
    """ Aggregate synapse data by neuropil. 
    For each neuropil, get synapse count, average neurotransmitter probabilities,
    and neurotransmitter probability variances.
    """
    print(f"{datetime.now().strftime("%H:%M:%S")} Aggregating neuropil data ...")
    trimmed = connectome.drop(columns = ["pre","post","x","y","z"])
    # Get aggregated data (neurotransmitter means and variances)
    agg_dict = {"gaba": ["mean", "var"],
                "ach": ["mean", "var"],
                "other": ["mean", "var"]}
    aggregated = trimmed.groupby(["neuropil"]).agg(agg_dict)
    aggregated.columns = [f"{col}_{stat}" if stat else col
        for col, stat in aggregated.columns.to_flat_index()]
    aggregated = aggregated.reset_index(drop=False).set_index(
        "neuropil", drop=True).persist()
    
    # Get synapse counts per neuropil
    counts = trimmed.groupby(["neuropil"]).count().reset_index(
        drop=False).set_index("neuropil", drop=True)
    counts = counts.drop(columns = ["gaba", "ach"]).rename(
        columns={"other": "neuropil_size"}).persist()
    
    # Combine aggregated data into one dataframe
    merged_aggregated = counts.merge(aggregated, on=["neuropil"], how="inner")
    merged_aggregated = merged_aggregated.set_index("neuropil", drop=False).persist()
    print(f"{datetime.now().strftime("%H:%M:%S")} Neuropil data aggregated")
    return merged_aggregated


def do_stats(r_path, result_dir):
    """ Take a sample then spawn a subprocess that runs the R stats script """
    sample = util.downsample(connectome, 1_000_000)
    try:
        subprocess.run(
            [r_path, "statistical_analysis.R", result_dir], 
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print("An error occurred in statistical_analysis.R")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
    

def main():
    pass