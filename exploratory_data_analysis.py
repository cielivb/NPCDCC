import dask.dataframe as dd
import dask.delayed
import pyarrow.feather as feather
import pandas as pd


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


chunks = get_chunks(chunk_size=1000, path="data/proofread_connections_783.feather")
pd_dfs = [feather_to_pandas(chunk) for chunk in chunks]
df1 = dd.from_delayed(pd_dfs)
print(df1)

chunks = get_chunks(chunk_size=1000, path="data/per_neuron_neuropil_count_post_783.feather")
pd_dfs = [feather_to_pandas(chunk) for chunk in chunks]
df2 = dd.from_delayed(pd_dfs)
print(df2)

chunks = get_chunks(chunk_size=1000, path="data/per_neuron_neuropil_count_pre_783.feather")
pd_dfs = [feather_to_pandas(chunk) for chunk in chunks]
df3 = dd.from_delayed(pd_dfs)
print(df3)