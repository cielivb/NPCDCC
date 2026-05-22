""" Edge scoring functions """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame
PDF: TypeAlias = pd.DataFrame


def get_scores(df: DDF, start_node: int):
    """ Get and return edge scores dataframe starting at start node. 
    DRIVER version. Runs one PBFS and one PBFS backtrack.
    """
    state = graph_utils.create_state_df(df)
    state, pc_df, cp_df, num_sps = pbfs.pbfs(start_node, df, state)
    scores = pbfs.pbfs_backtrack(pc_df, cp_df, num_sps)
    return scores.persist()


def get_scores_pd(dask_df: DDF, start_node: int):
    """ Get and return edge scores dataframe starting at start node. 
    WORKER version. Run one PBFS and one PBFS backtrack."""
    df = dask_df.compute() # Convert to pandas
    state = graph_utils.create_state_df(df)
    state, pc_df, cp_df, num_sps = pbfs.pbfs_pd(start_node, df, state)
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


def aggregate_scores(df: DDF, scores_list: list) -> DDF:
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
    merged["score"] = merged["score"].fillna(0)
    return merged.persist()


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


def chop(df_w_edge_scores: DDF, k: float) -> DDF:
    """ Return dataframe with edges that exceeded thresh removed """
    upper_thresh = get_upper_threshold(df_w_edge_scores, k)
    keep = df_w_edge_scores[df_w_edge_scores["score"] < upper_thresh]
    return keep.persist()


def chop_pd(dask_df_w_edge_scores: DDF, k: float) -> tuple[DDF]:
    """ Return original and processed dataframe with high edge scores removed """
    df = dask_df_w_edge_scores.compute() # Convert to pandas
    upper_thresh = get_upper_threshold(df, k)
    keep = df_w_edge_scores[df_w_edge_scores["score"] < upper_thresh]
    dask_keep = ddf.from_pandas(keep, npartitions=1)
    return dask_keep.persist()