""" Parallel Breadth First Search Functions - Dask Implementation """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame
PDF: TypeAlias = pd.DataFrame


def seed_pbfs(df: PDF|DDF, start_node: int, full: bool):
    """ Create initial dataframes for PBFS based on dataframe type and full """
    if isinstance(df, PDF):
        level_nodes = pd.DataFrame({"node_id": pd.Series([start_node], dtype=np.int32)})
        pc_df = pd.DataFrame({"parent": pd.Series(dtype=np.int32), 
                     "child": pd.Series(dtype=np.int32),
                     "syn_count": pd.Series(dtype=np.int16)
                     }).set_index("parent", drop = False)
    else:
        level_nodes = ddf.from_dict({"node_id": [start_node]}, 
                                    dtype = int, npartitions = 1)
        pc_df = ddf.from_dict({"parent": pd.Series(dtype=np.int64), 
                               "child": pd.Series(dtype=np.int64),
                               "syn_count": pd.Series(dtype=np.int64)}, 
                              npartitions=1).set_index("depth", drop = False)        
    if full and isinstance(df, PDF):
        cp_df = pd.DataFrame({"child": pd.Series(dtype=np.int32), 
                              "parent": pd.Series(dtype=np.int32),
                              "syn_count": pd.Series(dtype=np.int16)
                              }).set_index("child", drop = False)
        num_sps_df = pd.DataFrame({"depth": pd.Series([0], dtype=np.int32),
                                   "node_id": pd.Series([start_node], dtype=np.int32),
                                   "num_sps": pd.Series([1], dtype=np.int64)
                                   }).set_index("depth", drop = False)    
    elif full and isinstance(df, DDF):
        cp_df = ddf.from_dict({"child": pd.Series(dtype=np.int64), 
                               "parent": pd.Series(dtype=np.int64),
                               "syn_count": pd.Series(dtype=np.int64)}, 
                              npartitions=1).set_index("child", drop = False)
        num_sps_df = ddf.from_dict({"depth": [0], 
                                    "node_id": [start_node], 
                                    "num_sps": [1]}, 
                                   npartitions=1).set_index("depth", drop = False)
    else:
        cp_df, num_sps_df = None, None
    return level_nodes, pc_df, cp_df, num_sps_df


def pbfs_hybrid(start_node: int, df: DDF|PDF, state: DDF|PDF, full=True):
    """ Run a parallel breadth-first-search on component. 
    
    DRIVER and WORKER version - no need to wrap pandas results in dask because
    not returning directly to main pipeline (returned to edge scoring functions).
    
    Return state dataframe, pc_df (parent-child dataframe), cp_df (child-parent
    dataframe) and num_sps_df (number of shortest paths dataframe).
    
    If full is False, do not create cp_df nor num_sps_df (return None instead).
    Update/create only state dataframe and pc_df dataframe.
    
    The parallel component of this BFS involves processing an entire level/
    frontier at a time, rather than naively iterating over every node for
    every level.

    """
    
    # Set up initial PBFS variables
    depth, checkpoint_interval, is_dask = 0, 50, isinstance(df, DDF)
    level_nodes, pc_df, cp_df, num_sps_df = seed_pbfs(df, start_node, full)
    
    while True:
        # Update child-parent and parent-children relationship dataframes
        state = update_state_df(state, level_nodes, "D")
        level_data = get_level_data(level_nodes, df)
        pc_df, cp_df = update_pc_cp_dfs(
            level_data, pc_df, cp_df, state, full, is_dask)
        
        # Reindex pc_df and cp_df. Do it here because will these are
        # reused in the next frontier, and I want fast lookups in the next
        # frontier as well!
        pc_df = pc_df.set_index("parent", drop=False)
        if full:
            cp_df = cp_df.set_index("child", drop=False)
        
        # Update number of shortest paths dataframe
        if full:
            num_sps_df = update_num_sps_df(level_nodes, depth, cp_df, num_sps_df)
        update_state_df(state, level_nodes, "P")
        
        # Increase depth and change current nodes to child nodes, terminating
        # loop if no children exist in the next level.
        children = get_children(level_nodes, pc_df)        
        level_nodes = children
        if is_dask:
            if df.head(1, npartitions=-1).empty:
                break
        elif df.empty: # pandas equivalent 
            break
        depth += 1
        
        # Persist dask dataframes every checkpoint_interval iterations to 
        # prevent task graph explosions
        if is_dask and depth % checkpoint_interval == 0:
            state, pc_df = state.persist(), pc_df.persist()
            if full:
                cp_df, num_sps = cp_df.persist(), num_sps.persist()


def get_level_data(level_nodes: PDF|DDF, df: PDF|DDF) -> DDF|PDF:
    """ Get edge data for edges involving any node in level_nodes """
    # df is undirected, so only need to look at 'pre' column
    subset_full = level_nodes.merge(component, left_on="node_id", 
                                    right_index=True, how="inner")
    subset = subset_full[["pre", "post", "syn_count"]]
    subset = subset.set_index("pre", drop=False)
    return subset    


def update_state_df(state: PDF|DDF, level_nodes: PDF|DDF, new_status: str) -> DDF|PDF:
    """ Update states of level_nodes in state dataframe to either D or P """    
    merged = state.merge(nodes_to_update, left_index=True, right_on="node_id", 
                         how="left", indicator=True)
    merged["update"] = merged["_merge"] == "both"
    merged = merged.set_index("node_id", drop=False)    
    state["state"] = state["state"].mask(merged["update"], new_status)
    return state    


def update_pc_cp_dfs(edges: DDF|PDF, pc_df: DDF|PDF, cp_df: DDF|PDF|None,
                     state: DDF|PDF, full: bool, is_dask: bool):
    """ Update parent-child and child-parent relationships with this levels data """
    # Get neighbour states
    edges = edges.merge(state, left_on="post", right_index=True, how="inner")
    
    # Add entries to new_pc_df for every parent-child relationship where
    # node_id is the parent (neighbour/child state is U)
    new_pc_df = edges[edges["state"] == "U"][["pre", "post", "syn_count"]]
    new_pc_df = new_pc_df.rename(columns={"pre": "parent", "post": "child"})
    if is_dask:
        pc_df = ddf.concat([pc_df, new_pc_df])
    else:
        pc_df = pd.concat([pc_df, new_pc_df])
    
    if full:
        # Add entries to new_cp_rels for every child-parent relationship where
        # node_id is the child (neighbour/parent state is P)
        new_cp_df = edges[edges["state"] == "P"][["pre", "post", "syn_count"]]
        new_cp_df = new_cp_df.rename(columns={"pre": "child", "post": "parent"})
        if is_dask:
            cp_df = ddf.concat([cp_df, new_cp_df])
        else:
            cp_df = pd.concat([cp_df, new_cp_df])
        cp_df = ddf.concat([cp_df, new_cp_df]) if is_dask else pd.concat([cp_df, new_cp_df])
        
    return (pc_df, cp_df)    


def update_pc_cp_dfs_pd():
    raise NotImplementedError


def update_num_sps_df():
    raise NotImplementedError


def update_num_sps_df_pd():
    raise NotImplementedError


def pbfs_backtrack(pc_df: DDF, cp_df: DDF, num_sps: DDF) -> DDF:
    raise NotImplementedError


def pbfs_backtrack_pd(pc_df: PDF, cp_df: PDF, num_sps: PDF) -> PDF:
    raise NotImplementedError
