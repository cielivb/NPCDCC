""" Demo of splitting a dask dataframe into sub-dataframes by neuropil """

from dask import delayed
import pandas as pd
import dask.dataframe as dd
import os

FILE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(FILE_DIR)

@delayed
def read_feather(path):
    return pd.read_feather(path, use_threads=True)

feather_path = os.path.join(ROOT_DIR, "data", "proofread_connections_783.feather")
ddf = dd.from_delayed(read_feather(feather_path))


def split_main_df_by_neuropil(main_df):
    """"""
    neuropil_list = list(ddf["neuropil"].unique().compute())
    df_list = map(lambda neuropil: main_df[main_df["neuropil"] == neuropil], neuropil_list)
    df_dict = {neuropil: df for neuropil, df in zip(neuropil_list, df_list)}
    return df_dict # In actual project, should wrap in a dask bag

grouped_dfs = split_main_df_by_neuropil(ddf)

# Print average gaba probability for each edge in left medulla
print(grouped_dfs["ME_L"][["neuropil", "gaba_avg"]].compute())