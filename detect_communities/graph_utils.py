""" Graph Utilities / Helper Functions """

from dask import dataframe as ddf

type DDF = ddf.DataFrame


def get_all_node_ids(df: DDF, node_cols=["pre","post"]) -> DDF:
    """ Return a series of every unique node/neuron id in the dataframe """
    node_cols = [df[col].rename("node_id") for col in node_cols]
    all_nodes = ddf.concat(node_cols)
    return all_nodes


def undirect_df():
    raise NotImplementedError


def get_components(df: DDF) -> list[DDF]:
    """ Return a list of component dataframes.
   
    For each component in the graph represented in df, return a dataframe 
    containing data for each edge of that component. This is done by performing
    iteratively performing parallel BFS to identify nodes belonging to different
    components. Only one component can be discovered at a time.

    """    
    raise NotImplementedError


def prune(df: DDF) -> DDF:
    raise NotImplementedError
