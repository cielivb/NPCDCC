""" Graph Utilities / Helper Functions """

import logging
import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

from . import pbfs

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


def get_start_node_for_get_components(state):
    """ Scan partitions for index of first U in that partition. This feels hacky
    but will be faster than original strategy that scanned entire dask dataframe """
    def first_u(df: PDF):
        first_i_w_u = pdf.loc[df["state"] == "U"]
        return first_i_w_u.head(1)
    candidates = state.map_partitions(first_u)
    start_node = candidates.head(1).index.compute()
    return start_node


def get_components(df: DDF) -> list[DDF]:
    """ Return a list of component dataframes.
    
    DRIVER version of get_components
   
    For each component in the graph represented in df, return a dataframe 
    containing data for each edge of that component. This is done by performing
    iteratively performing parallel BFS to identify nodes belonging to different
    components. Only one component can be discovered at a time.

    """    
    LOGGER = logging.getLogger(__name__)
    LOGGER.debug("Creating state dataframe ...")
    state = create_state_df(df)
    components = []
    
    LOGGER.debug("Entering while loop ...")
    while not (state["state"] == "P").all().compute():
        LOGGER.debug(f"Entered while loop")
        # Get nodes present in next component
        LOGGER.debug(f"Getting start node ...")
        start_node = get_start_node_for_get_components(state)
        LOGGER.debug(f"Got start node = {start_node}, type = {type(start_node)}")
        state, pc_df = pbfs.pbfs_hybrid(start_node, df, state, full=False)
        comp_nodes = get_all_nodes(
            pc_df, node_cols=["parent","child"]).to_frame(name="node")
        
        # Build component dataframe
        merged1 = comp_nodes.merge(df, left_on="node", right_index=True, how="inner")
        merged2 = comp_nodes.merge(df, left_on="node", right_on="post", how="inner")
        component_df = ddf.concat([merged1, merged2]).drop_duplicates().persist()
        components.append(component_df)
    
    LOGGER.debug(f"Found {len(component_df)} components!")
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
    state = create_state_df(df)
    components = []
    
    while not (state["state"] == "P").all():
        
        # Get nodes present in next component
        start_node = state[state["state"] == "U"].head(1).index[0]
        state, pc_df = pbfs.pbfs_hybrid(start_node, df, state, full=False)
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


#def union_find_components(df: DDF) -> list[DDF]:
    #""" Return a list of component dataframes. This version of union find
    #makes each node adopt the smallest representative possible among itself
    #and its neighbours with each iteration. This continues until there is no
    #change in labels or until the number of iterations reaches the graph 
    #diameter. """
    #node_ids = get_all_node_ids()
    #node_ids["repr"] = node_ids["node_id"] # Each node starts as its own rep
    #max_iterations = len(node_ids) # Infinite while bad
    #for _ in range(max_iterations):
        ## Join current rep of each node
        #...
    
    
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