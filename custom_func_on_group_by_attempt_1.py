from dask import delayed
import pandas as pd
import dask.dataframe as dd
import os

FILE_DIR = os.path.dirname(__file__)

@delayed
def read_feather(path):
    return pd.read_feather(path, use_threads=True)

feather_path = os.path.join(FILE_DIR, "data", "proofread_connections_783.feather")
ddf = dd.from_delayed(read_feather(feather_path))
#ddf.info(verbose=True, memory_usage=True)



def my_custom_func(sub_ddf):
    """ Remove some columns and add a column to the ddf """
    #print(sub_ddf.info(verbose=True, memory_usage=True))
    new_sub_ddf = sub_ddf[["pre_pt_root_id", "post_pt_root_id"]]
    new_sub_ddf["new_col"] = sub_ddf["gaba_avg"] + sub_ddf["ach_avg"]
    return new_sub_ddf

meta = {"pre_pt_root_id": int, "post_pt_root_id": int, "new_col": float}
results = ddf.groupby("neuropil").apply(my_custom_func, meta=meta)
results.info(verbose=True, memory_usage=True)

print(type(results))