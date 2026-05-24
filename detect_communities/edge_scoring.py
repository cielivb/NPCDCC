""" Edge scoring functions """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame
PDF: TypeAlias = pd.DataFrame


def get_start_nodes(df: PDF|DDF):
    """ Get a list of random start nodes for MGN PBFS on dataframe """
    # Get list of all nodes in dataframe
    node_ids = graph_utils.get_all_node_ids(df)
    
    # Choose number of start nodes as a function of number of nodes. I struggled
    # to find much in the literature about an ideal function, so I am using an
    # arbitrary function and hoping for the best.
    num_nodes = len(df)
    match num_nodes:
        case n if n < 500:
            k = 1
        case n if n < 1000:
            k = 0.75
        case n if n < 2000:
            k = 0.5
        case n if n < 10000:
            k = 0.3
        case _:
            k = 0.15
    
    # Get random subset of start nodes. Only need to sample 'pre' because df
    # is undirected.
    if isinstance(df, PDF):
        num_start_nodes = int(num_nodes * k)
        start_nodes = df["pre"].sample(
            n = num_start_nodes).to_numpy(dtype=np.int32)
    else:
        start_nodes = df["pre"].sample(
            frac = k).to_dask_array().astype(np.int32).compute()
    return start_nodes


def get_scores(df: DDF, start_node: int) -> DDF:
    """ Get and return edge scores dataframe starting at start node. 
    DRIVER version. Runs one PBFS and one PBFS backtrack.
    """
    state = graph_utils.create_state_df(df)
    state, pc_df, cp_df, num_sps = pbfs.pbfs_hybrid(start_node, df, state)
    scores = pbfs.pbfs_backtrack(pc_df, cp_df, num_sps)
    return scores.persist()


def get_scores_pd(dask_df: DDF, start_node: int) -> DDF:
    """ Get and return edge scores dataframe starting at start node. 
    WORKER version. Run one PBFS and one PBFS backtrack."""
    df = dask_df.compute() # Convert to pandas
    state = graph_utils.create_state_df(df)
    state, pc_df, cp_df, num_sps = pbfs.pbfs_hybrid(start_node, df, state)
    scores = pbfs.pbfs_backtrack_pd(pc_df, cp_df, num_sps)
    dask_scores = ddf.from_pandas(scores, npartitions=1)
    return dask_scores.persist()


def normalise_edges(df_partition: PDF) -> PDF:
    """ Normalise edges to form min_node (pre) -> max_node (post). 
    Within a partition, it is guaranteed that only one of a->b or b->a 
    exists."""
    parent, child = df_partition["parent"], df_partition["child"]
    new_df = df_partition.drop(columns = ["parent", "child"])
    new_df["pre"] = parent.where(parent <= child, child)
    new_df["post"] = child.where(parent <= child, parent)
    return new_df


def aggregate_scores(df: DDF, scores_list: list[DDF]) -> DDF:
    """ Return original dataframe with aggregated/final edge score column. 
    Each score dataframe in scores_list has columns parent, child, score.
    DRIVER and WORKER version - no computes.
    """
    # Sum scores and prepare for merge
    normalised = [df.map_partitions(normalise_edges) for df in scores_list]
    big_scores_df = ddf.concat(normalised)
    summed_scores_df = big_scores_df.groupby(
        ["pre", "post"])["score"].sum().reset_index()
    undirected = graph_utils.undirect(summed_scores_df)
    scores = undirected.set_index(["pre", "post"], drop = False)
    
    # Merge dataframes and replace NA scores with 0
    merged = df.merge(scores, left_on=["pre", "post"], 
                      right_index = True, how = "left")
    merged["score"] = merged["score"].fillna(0).persist()
    return merged


def get_upper_threshold(edge_scores: DDF|PDF, k: float) -> float:
    """ Calculate Median Absolute Deviation (MAD)-based upper threshold.
    edge_scores is dataframe with columns node1, node2, score. k is a multiplier
    that determines how extreme a score must be to be considered an 'outlier'
    (i.e., a bridge edge). k = 2.5 is commonly used in social sciences for 
    standard outlier detection. A more conservative k is more likely to result
    in larger, fewer, and less symmetric communities.
    """
    if isinstance(edge_scores, PDF):
        scores = edge_scores["score"].to_numpy
        median_score = scores.median()
        MAD = np.absolute(scores - median_score).median()
    else:
        scores = edge_scores["score"].to_dask_array()
        median_score = scores.median().compute()
        MAD = score.map_blocks(
            lambda arr: np.absolute(arr - median_score)).median().compute()
    upper_thresh = median_score + k * MAD    
    return upper_thresh


def chop(df_w_edge_scores: DDF, k: float) -> tuple[DDF]:
    """ Return dataframe with edges that exceeded thresh removed """
    upper_thresh = get_upper_threshold(df_w_edge_scores, k)
    keep = df_w_edge_scores[df_w_edge_scores["score"] < upper_thresh].persist()
    return (keep, df_w_edge_scores)


def chop_pd(dask_df_w_edge_scores: DDF, k: float) -> tuple[DDF]:
    """ Return original and processed dataframe with high edge scores removed """
    df = dask_df_w_edge_scores.compute() # Convert to pandas
    upper_thresh = get_upper_threshold(df, k)
    keep = df_w_edge_scores[df_w_edge_scores["score"] < upper_thresh]
    dask_keep = ddf.from_pandas(keep, npartitions=1).persist()
    return (dask_keep, dask_df_w_edge_scores)