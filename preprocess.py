""" Preprocess raw data 

Convert raw data from feather files to parquet files, and generate scalability 
test files of varying problem sizes.

This script should only need to be ran once.

"""
import dask
import os
import pyarrow
import pyarrow.feather as feather
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from dask import dataframe as ddf
from dask.distributed import Client
from dask.distributed import get_client
from datetime import datetime

CLIENT = None # Assigned in run()

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MAIN_FILE_RAW = os.path.join(DATA_DIR, "proofread_connections_783.feather")
COORD_FILE_RAW = os.path.join(DATA_DIR, "flywire_synapses_783.feather")
MAIN_FILE = os.path.join(DATA_DIR, "proofread_connections_783.parquet")
COORD_FILE = os.path.join(DATA_DIR, "flywire_synapses_783.parquet")

METADATA_FILE = os.path.join(DATA_DIR, "metadata.txt")


############################### PREPROCESS #####################################

def _preprocess_main_file(raw_path):
    """ Convert to parquet and create scalability test files """
    global MAIN_FILE
    if not os.path.exists(MAIN_FILE):
        _feather_to_parquet(file_to_convert=raw_path, destination=MAIN_FILE)


def _preprocess_coord_file(raw_path):
    """ Convert coordinate file to parquet file """
    global COORD_FILE
    if not os.path.exists(COORD_FILE):
        _feather_to_parquet(file_to_convert=raw_path, destination=COORD_FILE)


def _feather_to_parquet(file_to_convert, destination):
    """ Convert feather file to parquet file
    WARNING: brings the whole feather file into memory. This hogs about 11 GB 
    memory max (inspected via Dask Dashboard)"""
    print(f"{datetime.now().strftime("%H:%M:%S")} Converting {file_to_convert} to parquet file ...")
    table = feather.read_table(file_to_convert)
    pq.write_table(table, destination, compression=None, 
                   row_group_size=800_000)
    print(f"{datetime.now().strftime("%H:%M:%S")} {file_to_convert} converted to parquet")


def _get_all_node_ids(df, node_cols=["pre","post"]):
    """ Return a dataframe containing every unique node id in the dataframe """
    node_cols = [df[col].rename("node_id").to_frame() for col in node_cols]
    all_nodes = ddf.concat(node_cols).drop_duplicates().repartition(npartitions=1).reset_index(drop=True)
    return all_nodes


def _write_test_metadata(sub_connectome, test_id):
    """ Write test connectome file metadata
    E.g., number of nodes, number of edges, node:edge ratio, neuropils """
    global METADATA_FILE    
    num_nodes = _get_all_node_ids(sub_connectome, 
        node_cols=["pre_pt_root_id","post_pt_root_id"]).count()
    num_edges = sub_connectome["pre_pt_root_id"].count()
    neuropils = sub_connectome["neuropil"].unique()
    
    num_nodes, num_edges, neuropils = dask.compute(
        num_nodes, num_edges, neuropils)
    node_edge_ratio = num_nodes/num_edges    
    
    with open(METADATA_FILE, "a") as mfile:
        mfile.write(f"Testfile Metadata: {test_id}.feather\n")
        mfile.write(f"Number of nodes: {num_nodes}\n")
        mfile.write(f"Number of edges: {num_edges}\n")
        mfile.write(f"Node-edge ratio: {node_edge_ratio}\n")
        mfile.write(f"Neuropils: {list(neuropils)}\n\n")


def _write_subset_file(sub_connectome, test_id):
    """ Write parquet file containing sub-connectome. 
    These tests should fit in memory so will compute to pandas first to avoid
    saving them in their own directories """
    global DATA_DIR
    filename = os.path.join(DATA_DIR, f"{test_id}.parquet")
    pd_sub_connectome = sub_connectome.compute()
    pd_sub_connectome.to_parquet(filename, index=False, compression=None)
    
    
def _make_tests():
    """ Create subsets of main file of varying sizes for performance analysis """
    global MAIN_FILE
    print(f"{datetime.now().strftime("%H:%M:%S")} Generating test parquet files ...")
    connectome = ddf.read_parquet(MAIN_FILE)
    connectome = connectome.categorize(columns=["neuropil"]).persist()    
    
    # Subset dataframe into different sizes
    tiny = connectome.map_partitions(
        lambda df: df[df["neuropil"].str.startswith("BU_")])
    small = connectome.map_partitions(
        lambda df: df[df["neuropil"].str.startswith("LOP_")])   
    medium = connectome.map_partitions(
        lambda df: df[df["neuropil"].str.startswith("ME_")])    
    omit = {"ME_L", "ME_R", "LOP_L", "LOP_R"}
    large = connectome.map_partitions(
        lambda df: df[~df["neuropil"].isin(omit)])
    
    # Write subset data 
    subsets = [tiny, small, medium, large]
    test_ids = ["tiny", "small", "medium", "large"]
    futures = []
    for sub_connectome, test_id in zip(subsets, test_ids):
        futures.append(CLIENT.submit(_write_subset_file, sub_connectome, test_id))
        _write_test_metadata(sub_connectome, test_id)
    CLIENT.gather(futures)
    print(f"{datetime.now().strftime("%H:%M:%S")} Generated test parquet files")


################################### MAIN #######################################

def _is_downloaded(file1, file2):
    if os.path.exists(file1) and os.path.exists(file2):
        return True
    return False


def _all_tests_exist():
    global DATA_DIR
    test_ids = ["tiny", "small", "medium", "large"]
    test_files = [os.path.join(DATA_DIR, f"{_id}.parquet") for _id in test_ids]
    all_exist = all(os.path.exists(file) for file in test_files)
    return True if all_exist else False


def run():
    """ Convert raw data """
    global MAIN_FILE_RAW, COORD_FILE_RAW, CLIENT
    print("Notice: this may take about 10 minutes if this is the first time "
          "running preprocess.py. ")
    if not _is_downloaded(MAIN_FILE_RAW, COORD_FILE_RAW):
        raise Exception("Cannot find raw data files in the data directory")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_preprocess_main_file, MAIN_FILE_RAW)
        f2 = pool.submit(_preprocess_coord_file, COORD_FILE_RAW)
        file_paths = f1.result()
        coord_path = f2.result()
        
    CLIENT = get_client()
    if not _all_tests_exist():
        _make_tests()
    
    print(f"\n{datetime.now().strftime("%H:%M:%S")} Preprocessing complete!")