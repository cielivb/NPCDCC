""" Exploratory Data Analysis

Generate descriptive statistics for the following data files:
data/proofread_connections_783.feather
data/per_neuron_neuropil_count_post_783.feather
data/per_neuron_neuropil_count_pre_783.feather

"""

import dask.bag as db
import dask.dataframe as dd
import dask.delayed
import pyarrow.feather as feather
import pandas as pd

import os


FILE_DIR = os.path.dirname(__file__)


@dask.delayed
def feather_to_pandas(chunk):
    """ Convert feather file chunk to pandas dataframe """
    return chunk.to_pandas()


def get_chunks(chunk_size, path):
    """ Return list of feather chunks (data views - not fully in memory) """
    table = feather.read_table(path)
    
    @dask.delayed
    def get_chunk(start_row):
        end_row = min(start_row + chunk_size, table.num_rows)
        chunk = table.slice(start_row, end_row - start_row)
        return chunk 
    
    chunks = [get_chunk(_) for _ in range(0, table.num_rows, chunk_size)]
    return chunks


@dask.delayed
def load_dask_dataframe(path):
    chunks = get_chunks(chunk_size=1000, path=path)
    pd_dfs = [feather_to_pandas(chunk) for chunk in chunks]
    df = dd.from_delayed(pd_dfs)
    return df


def get_dask_dataframes():
    global FILE_DIR
    data_dir = os.path.join(FILE_DIR, "data")
    paths = [os.path.join(data_dir, "proofread_connections_783.feather"),
             os.path.join(data_dir, "per_neuron_neuropil_count_pre_783.feather"),
             os.path.join(data_dir, "per_neuron_neuropil_count_post_783.feather")]
    dfs = [load_dask_dataframe(path) for path in paths]
    return dfs


@dask.delayed
def get_stat_dict(df):
    """ Return a dictionary of summary statistics for a dask dataframe """
    stats = {}
    stats['dtypes'] = df.dtypes
    stats['describe'] = df.describe(include="all")
    return stats


def get_descriptive_stats(dask_dfs):
    """ Compute summary statistics for each dask dataframe in dask_dfs """
    stats = [get_stat_dict(df) for df in dask_dfs]
    return dask.compute(*stats)


def write_descriptive_stats(stats):
    global FILE_DIR
    outfile = os.path.join(FILE_DIR, "results", "exploratory_data_analysis.txt")
    with open(outfile, 'w') as outfile:
        for i, df_stats in enumerate(stats):
            outfile.write(f"\nDataframe {i+1} Exploratory Data Analysis\n")
            for key, value in df_stats.items():
                outfile.write(f"\n{key}\n")
                outfile.write(f"{value}\n")


def explore():
    dask_dfs = get_dask_dataframes()
    stats = get_descriptive_stats(dask_dfs)
    write_descriptive_stats(stats)


if __name__ == '__main__':
    explore()