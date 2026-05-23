""" Parallel Breadth First Search Functions - Dask Implementation """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame
PDF: TypeAlias = pd.DataFrame




################################### PBFS #######################################


def seed_pbfs(df: PDF|DDF, start_node: int, full: bool):
    """ Create initial dataframes for PBFS based on dataframe type and full """
    if isinstance(df, PDF):
        level_nodes = pd.DataFrame(
            {"node_id": pd.Series([start_node], dtype=np.int32)})
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
                                   "log10_num_sps": pd.Series([1], dtype=np.int64)
                                   }).set_index("depth", drop = False)    
    elif full and isinstance(df, DDF):
        cp_df = ddf.from_dict({"child": pd.Series(dtype=np.int64), 
                               "parent": pd.Series(dtype=np.int64),
                               "syn_count": pd.Series(dtype=np.int64)}, 
                              npartitions=1).set_index("child", drop = False)
        num_sps_df = ddf.from_dict({"depth": [0], 
                                    "node_id": [start_node], 
                                    "log10_num_sps": [1]}, 
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
            num_sps_df = update_num_sps_df(
                level_nodes, depth, cp_df, num_sps_df, is_dask)
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


def update_num_sps_df(level_nodes: DDF|PDF, depth: int, cp_df: PDF|DDF, 
                      num_sps_df: PDF|DDF, is_dask: bool) -> DDF|PDF:
    """ Update num_sps_df with the log10 of the number of shortest paths to each
    node in level_nodes """
    if depth == 0: # Root node - prefilled at start of PBFS
        return num_sps_df
    # Parent num sps will always be at depth one level above this level. Subset
    # to the parent level to speed up scan (avoids scanning all levels).
    all_parent_num_sps = num_sps_df[num_sps_df["depth"] == depth-1]
    
    # Get dataframe with node_id (child), parent, syn_count, and parent log10_num_sps
    cp_rels = level_nodes.merge(
        cp_df, left_on = "node_id", right_index = True, how = "inner")
    new_num_sps = cp_rels.merge(
        all_parent_num_sps, left_on = "parent", right_on = "node_id", how = "inner")
    
    # Compute parent contributions to each child in log space. Multiplication
    # in linear space (num_sps x syn_count) becomes addition in log space.
    new_num_sps["child_log10_num_sps"] = (
        new_num_sps["log10_num_sps"] + np.log10(new_num_sps["syn_count"]))
    
    # Group by child node and apply LSE to each child's log10_num_sps to get the
    # number of shortest paths to that child expressed in log10 space.
    if is_dask:
        grouped = new_num_sps.groupby("node_id")["child_log10_num_sps"].reduction(
            chunk = graph_utils.lse, aggregate = graph_utils.lse, 
            meta = ("log10_num_sps", np.float64))
    else:
        grouped = new_num_sps.groupby("node_id")["child_log10_num_sps"].apply(
            graph_utils.lse)
    
    result = grouped.reset_index().rename(
        columns = {"child_log10_num_sps": "log10_num_sps"})
    result["depth"] = depth
    
    if is_dask:
        return ddf.concat([num_sps_df, result])
    return pd.concat([num_sps_df, result])
    
    
def get_children(level_nodes: PDF|DDF, pc_df: PDF|DDF):
    """ Get the children node ids of all nodes in level_nodes """    
    merged = level_nodes.merge(pc_df, left_on="node_id", right_index=True, how="inner")
    children = merged["child"].drop_duplicates().to_frame(name="node_id")
    return children    





############################## PBFS BACKTRACK ##################################


def seed_edge_score_df(is_dask: bool):
    """ Create initial empty edge score dataframe (pandas or dask) """
    if is_dask:
        edge_score_df = ddf.from_dict(
            {"depth": pd.Series(dtype=np.int32), "parent": pd.Series(dtype=np.int32),
             "child": pd.Series(dtype=np.int32), "score": pd.Series(dtype=np.float64)}, 
            npartitions=1).set_index("depth", drop = False)     
    else:
        edge_score_df = pd.DataFrame(
            {"depth": pd.Series(dtype=np.int32), "parent": pd.Series(dtype=np.int32),
             "child": pd.Series(dtype=np.int32), "score": pd.Series(dtype=np.float64)
             }).set_index("depth", drop = False)
    return edge_score_df


def pbfs_backtrack(pc_df: PDF|DDF, cp_df: PDF|DDF, num_sps: PDF|DDF) -> PDF|DDF:
    """ Iterate from the bottom of the PBFS tree upwards, assigning node credits
    and edge scores along the way. Return edge scores. """    
    # Set up PBFS backtrack
    depth = num_sps["depth"].max().compute() if is_dask else num_sps["depth"].max()
    checkpoint_interval, is_dask = 50, isinstance(num_sps, DDF)    
    edge_score_df = seed_edge_score_df(is_dask)
    
    # Run PBFS backtrack, accumulating edge scores
    while depth >= 0:
        level_num_sps = num_sps[num_sps["depth"] == depth]
        parent_num_sps = num_sps[num_sps["depth"] == depth-1]
        node_credits = assign_node_credits(depth, level_num_sps, edge_score_df)
        edge_scores = assign_edge_scores(
            depth, node_credits, cp_df, parent_num_sps, is_dask)
        if is_dask:
            edge_score_df = ddf.concat([edge_score_df, edge_scores])
        else:
            edge_score_df = pd.concat([edge_score_df, edge_scores])
        edge_score_df = edge_score_df.set_index("depth", drop=False, sort=True)
        depth -= 1
        if depth % checkpoint_interval == 0 and is_dask:
            edge_score_df = edge_score_df.persist()
    
    if is_dask:
        edge_score_df = edge_score_df.persist()
    return edge_score_df


def assign_node_credits(depth: int, level_num_sps: PDF|DDF, 
                        edge_score_df: PDF|DDF) -> PDF|DDF:
    """ Apply Rules 1 & 2 of MMDS Ch10 pp. 365.
    Each node gets a credit equal to 1 plus the sum of the scores of the 
    DAG edges from that node to the level below. Lead nodes thus get credit
    of 1. Return a df with columns node_id, credit."""    
    # Get edge scores relevant only to this level's nodes
    edge_scores = edge_scores_df.loc[depth+1]
    
    # Get parent-child relationships where this level's nodes are the parents
    pc_rels = level_num_sps.merge(edge_scores, left_on="node_id", 
                                  right_on="parent", how="inner")
    # Get child contributions to parent (this level's) node credits
    child_contribs = pc_rels.groupby("node_id")["score"].sum().to_frame() # index=node_id
    
    # Assign node credit
    node_credits = level_num_sps.merge(child_contribs, left_on="node_id", 
                                        right_index=True, how="left")
    node_credits["score"] = node_credits["score"].fillna(0)
    node_credits["credit"] = 1 + node_credits["score"]
    node_credits = node_credits[["node_id", "credit"]]
    return node_credits


def assign_edge_scores(depth: int, node_credits: PDF|DDF, cp_df: PDF|DDF, 
                       parent_num_sps: PDF|DDF, is_dask: bool) -> PDF|DDF:
    """ Apply Rule 3 of MMDS Ch10 pp. 365.
    'A DAG edge e entering node Z from the level above is given a share of the
    credit of Z proportional to the fraction of shortest paths from the root to
    Z that go through e'
    Return a dataframe with columns depth, parent, child, score.
    """
    # Get child-parent relationships where this level's nodes are the children
    cp_rels = node_credits.merge(cp_df, left_on="node_id", 
                                 right_index=True, how="inner")
    
    # Get number of shortest paths for each parent on this level
    sps = cp_rels.merge(parent_num_sps, left_on="parent", 
                        right_on="node_id", how="inner")
    sps = sps[["node_id_x", "credit", "parent", "log10_num_sps", "syn_count"]]
    sps = sps.rename(columns={"node_id_x":"node_id"})
    
    # Assign edge credits. Each node->parent edge is allocated a proportion of 
    # the node's credit such that stronger connections (more synapses) receive
    # less credit. More credit increases the likelihood of a higher edge
    # betweenness score later. Multiplication in linear space (eg syn_count *
    # num_sps) becomes addition in log space. The number of shortest paths is
    # already expressed in log space in sps.
    if is_dask:
        sps["weight"] = 1 / (sps["syn_count"].map_partitions(np.log10) + 
                             sps["log10_num_sps"])
    else:
        sps["weight"] = 1/(np.log10(sps["syn_count"]) + sps["log10_num_sps"])
    total_weights = sps.groupby("node_id")["weight"].sum().to_frame() # Index is node_id
    edge_scores = sps.merge(total_weights, left_on="node_id", 
                            right_index=True, how="inner")
    edge_scores["prop"] = edge_scores["weight_x"] / edge_scores["weight_y"]
    edge_scores["score"] = edge_scores["credit"] * edge_scores["prop"]
    
    # Reformat columns then return
    edge_scores = edge_scores[["parent", "node_id", "score"]]
    edge_scores["depth"] = depth
    edge_scores = edge_scores.rename(columns = {"node_id": "child"})
    edge_scores = edge_scores
    return edge_scores
