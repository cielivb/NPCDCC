""" Graph Utilities / Helper Functions """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame
PDF: TypeAlias = pd.DataFrame


def get_all_node_ids(df: DDF|PDF, node_cols=["pre","post"]) -> DDF|PDF:
    """ Return a dataframe containing every unique node id in the dataframe """
    node_cols = [df[col].rename("node_id").to_frame() for col in node_cols]
    if isinstance(df, PDF):
        all_nodes = pd.concat(node_cols)
    else:
        all_nodes = ddf.concat(node_cols)
    return all_nodes


def undirect_df(df: DDF) -> DDF:
    """ Add b->a for every a->b in df """
    df_reversed = df.rename(columns={"pre": "post", "post": "pre"})
    undirected = ddf.concat([df, df_reversed])
    undirected = undirected.set_index("pre", drop = False)
    return undirected


def create_state_df(df: DDF|PDF) -> DDF|PDF:
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

    return components.persist()


def get_components_pd(dask_df: DDF) -> list[DDF]:
    """ Return a list of component dataframes.
    
    WORKER version of get_components
    
    For each component in the graph represented in df, return a dataframe 
    containing data for each edge of that component. This is done by performing
    iteratively performing parallel BFS to identify nodes belonging to different
    components. Only one component can be discovered at a time.

    """
    df = dask_df.compute() # Convert to pandas
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
        dask_comps.append(ddf.from_pandas(comp, npartitions=1).persist())
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
        
    return pruned.persist()


def prune_pd(dask_df: DDF) -> DDF:
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
    pruned = dask_df.compute() # Convert to pandas
    
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
    
    return ddf.from_pandas(pruned, npartitions=1).persist()


def lse(log_values: pd.Series) -> float:
    """ Calculate log10 of the sum of the linear space equivalent of log values.
    Based on https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations.
    """
    m = series.max()
    return m + np.log10((10**(series - m)).sum())


def update_num_sps_df(
    level_nodes: Union[dd.DataFrame, pd.DataFrame],
    depth: int,
    cp_df: Union[dd.DataFrame, pd.DataFrame],
    num_sps_df: Union[dd.DataFrame, pd.DataFrame]
) -> Union[dd.DataFrame, pd.DataFrame]:
    """
    Update num_sps_df with the log10 of the number of shortest paths
    to each node in level_nodes.
    Works with both Pandas and Dask DataFrames.
    """
    if depth == 0:  # Root node already initialized
        return num_sps_df
    
    # Parent num_sps at depth-1
    all_parent_num_sps = num_sps_df[num_sps_df["depth"] == depth - 1]
    
    # Merge child-parent relationships
    cp_rels = level_nodes.merge(cp_df, left_on="node_id", right_index=True, how="inner")
    new_num_sps = cp_rels.merge(all_parent_num_sps, left_on="parent", right_on="node_id", how="inner")
    
    # Compute child contributions in log space
    new_num_sps["child_log10_num_sps"] = (
        new_num_sps["log10_num_sps"] + np.log10(new_num_sps["syn_count"])
    )
    
    # Group by child node and apply log-sum-exp
    grouped = new_num_sps.groupby("node_id")["child_log10_num_sps"].apply(
        lambda s: lse(pd.DataFrame({"log10_num_sps": s})) if isinstance(s, pd.Series)
        else lse(dd.from_pandas(pd.DataFrame({"log10_num_sps": s.compute()}), npartitions=1))
    )
    
    # Reset index and add depth
    result = grouped.reset_index().rename(columns={"child_log10_num_sps": "log10_num_sps"})
    result["depth"] = depth
    
    # Append to num_sps_df
    if isinstance(num_sps_df, dd.DataFrame):
        return dd.concat([num_sps_df, result])
    else:
        return pd.concat([num_sps_df, result])
