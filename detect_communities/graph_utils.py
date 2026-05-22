""" Graph Utilities / Helper Functions """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame
PDF: TypeAlias = pd.DataFrame


def get_all_node_ids(df: DDF|PDF, node_cols=["pre","post"]) -> DDF:
    """ Return a dataframe containing every unique node id in the dataframe """
    node_cols = [df[col].rename("node_id").to_frame() for col in node_cols]
    if isinstance(df, PDF):
        all_nodes = pd.concat(node_cols)
    else:
        all_nodes = ddf.concat(node_cols)
    return all_nodes


def undirect_df(df: DDF|PDF):
    """ Add b->a for every a->b in df, and remove duplicates """
    df_reversed = df.rename(columns={"pre": "post", "post": "pre"})
    if isinstance(df, PDF):
        undirected = pd.concat([df, df_reversed]).drop_duplicates()
        undirected = undirected.set_index("pre", drop = False)
    else:
        undirected = ddf.concat([df, df_reversed]).drop_duplicates()
        undirected = undirected.set_index("pre", drop = False).persist()    
    return undirected


def create_state_df(df: DDF|PDF):
    """ Return either a dask or pandas state dataframe depending on input """
    state = get_all_node_ids(df).drop_duplicates()
    state["state"] = "U"
    state = state.set_index("node_id")
    if isinstance(df, DDF):
        state = state.persist()
    return state


def get_components(df: DDF) -> list[DDF]:
    """ Return a list of component dataframes.
    
    DRIVER version of get_components
   
    For each component in the graph represented in df, return a dataframe 
    containing data for each edge of that component. This is done by performing
    iteratively performing parallel BFS to identify nodes belonging to different
    components. Only one component can be discovered at a time.

    """    
    undirected_df = undirect_df(df)
    state = create_state_df(df)
    components = []
    
    while not (state["state"] == "P").all().compute():
        
        # Get nodes present in next component
        start_node = state[state["state"] == "U"].head(1).index.compute()[0]
        state, pc_df = pbfs(start_node, undirected_df, state, full=False)
        comp_nodes = get_all_nodes(
            pc_df, node_cols=["parent","child"]).to_frame(name="node")
        
        # Build component dataframe
        merged1 = comp_nodes.merge(df, left_on="node", right_index=True, how="inner")
        merged2 = comp_nodes.merge(df, left_on="node", right_on="post", how="inner")
        component_df = ddf.concat([merged1, merged2]).drop_duplicates().persist()
        components.append(component_df)

    return components


def get_components_pd(df: DDF) -> list[DDF]:
    """ Return a list of component dataframes.
    
    WORKER version of get_components
    
    For each component in the graph represented in df, return a dataframe 
    containing data for each edge of that component. This is done by performing
    iteratively performing parallel BFS to identify nodes belonging to different
    components. Only one component can be discovered at a time.

    """
    undirected_df = undirect_df(df)
    state = create_state_df(df)
    components = []
    
    while not (state["state"] == "P").all():
        
        # Get nodes present in next component
        start_node = state[state["state"] == "U"].head(1).index[0]
        state, pc_df = pbfs_pd(start_node, undirected_df, state, full=False)
        comp_nodes = get_all_nodes(
            pc_df, node_cols=["parent", "child"]).to_frame(name="node")

        # Build component dataframe
        merged1 = comp_nodes.merge(df, left_on="node", right_index=True, how="inner")
        merged2 = comp_nodes.merge(df, left_on="node", right_on="post", how="inner")
        component_df = pd.concat([merged1, merged2]).drop_duplicates()
        components.append(component_df)
    
    dask_comps = []
    for comp in components:
        dask_comps.append(ddf.from_pandas(comp))
    return dask_comps

    
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
    
    def get_degree_1_nodes(df: PDF) -> PDF:
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