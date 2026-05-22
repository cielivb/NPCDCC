""" Graph Utilities / Helper Functions """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame


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
    """ Iteratively remove degree 1 edges from a component dataframe 
    
    DRIVER version of prune
    
    A degree 1 edge is defined here as an edge associated with at least one
    degree 1 node, where a degree 1 node is a node connected by any number of 
    edges to one and only one other node. No nodes in the dataset will have 
    edges to themselves. Synapse count and directionality are not considered.
            
    The naive approach would be to take away one edge at a time. This parallel
    version improves performance by finding all degree 1 edges initially, and
    pruning each 'chain' in parallel. The time to complete is the time it takes
    to process the longest 'chain'.

    """    
    def get_degree_1_nodes(df: DDF) -> DDF:
        node_degrees = df.groupby(df.index)["post"].nunique().to_frame("degree")
        deg1_nodes = node_degrees[
            node_degrees["degree"] == 1].index.to_frame("node")
        return deg1_nodes
    
    pruned, deg1_nodes = df, get_degree_1_nodes(df)
    while deg1_nodes.count().compute() > 0:
        # Get edges where either pre or post is in deg1_nodes
        merged1 = pruned.merge(deg1_nodes, left_index=True, right_on="node", how="inner")
        merged2 = pruned.merge(deg1_nodes, left_on="post", right_on="node", how="inner")
        to_prune = ddf.concat([merged1, merged2])
        
        # Remove edges and recompute degree 1 nodes
        merged = pruned.merge(to_prune, on=["pre","post"], how="left", indicator=True)
        pruned = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).persist()
        deg1_nodes = get_degree_1_nodes(pruned)
        
    return pruned


def prune_pd(df: DDF) -> DDF:
    """ Iteratively remove degree 1 edges from a component dataframe 
    
    WORKER version of prune
    
    A degree 1 edge is defined here as an edge associated with at least one
    degree 1 node, where a degree 1 node is a node connected by any number of 
    edges to one and only one other node. No nodes in the dataset will have 
    edges to themselves. Synapse count and directionality are not considered.
            
    The naive approach would be to take away one edge at a time. This parallel
    version improves performance by finding all degree 1 edges initially, and
    pruning each 'chain' in parallel. The time to complete is the time it takes
    to process the longest 'chain'.

    """    
    pruned = df.compute() # Convert to pandas
    
    def get_degree_1_nodes(df: pd.DataFrame) -> pd.DataFrame:
        node_degrees = df.groupby(df.index)["post"].nunique().to_frame("degree")
        deg1_nodes = node_degrees[
            node_degrees["degree"] == 1].index.to_frame("node")
        return deg1_nodes
    
    deg1_nodes = get_degree_1_nodes(pruned)
    
    while len(deg1_nodes) > 0:
        # Get edges where either pre or post is in deg1_nodes
        merged1 = pruned.merge(deg1_nodes, left_on="pre", right_on="node", how="inner")
        merged2 = pruned.merge(deg1_nodes, left_on="post", right_on="node", how="inner")
        to_prune = pd.concat([merged1, merged2])
        
        # Remove edges and recompute degree 1 nodes
        merged = pruned.merge(to_prune, on=["pre","post"], how="left", indicator=True)
        pruned = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        deg1_nodes = get_degree_1_nodes(pruned)
    
    return ddf.from_pandas(pruned)