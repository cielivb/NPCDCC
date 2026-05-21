""" Main / Controller Community Detection Script 

run() : orchestrator function; repeatedly submits components to modified
girvan newman to be broken down until no more components can be broken
down, then returns the tagged connectome


"""

from dask import dataframe as ddf
from dask.distributed import Client
from queue import Queue

type DDF = ddf.DataFrame


def run(connectome: DDF) -> DDF:
    """ Submit component jobs then tag connectome with community IDs """
    client = get_client()
    queue, communities = Queue(), []
    initial_components = graph_utils.get_components(connectome)
    for component in initial_components:
        queue.put(component)
    
    # Iterate until all connectome communities are found
    while not queue.empty():
        component = queue.get()
        f = client.submit(modified_girvan_newman, component)
        new_components = f.result() # List of tuples (ddf, bool)
        
        for df, should_continue in new_components:
            if should_continue:
                queue.put(cleaned[0])
            elif df:
                communities.append(df)
                
    tagged = tag_edges(connectome, communities)
    return tagged


def tag_edges(connectome: DDF, communities: list[DDF]) -> DDF:
    """ Return connectome sorted by neuropil and tagged with community IDs. 
    Community IDs are simple integers. Allocate each cluster dataframe in the
    list a community ID, concatenate the cluster dataframes, then left merge
    connectome_df onto cluster_dfs.
    """
    if not communities:
        return None
    community_ids = range(1, len(communities) + 1)
    for community, community_id in zip(communities, community_ids):
        community["community_id"] = community_id
    
    big_community_df = ddf.concat(community_dfs)
    tagged = connectome.merge(big_community_df, 
                              on = ["pre", "post"], 
                              how="left")
    tagged = tagged.sort_values(["neuropil", "community_id"])
    return tagged


def modified_girvan_newman(component: DDF) -> list[tuple[DDF,bool]]:
    """ Remove bridge edges from component dataframe.
    
    Returns a list of tuples containing component dataframes and a boolean for
    whether processing should continue. 
    
    If bool continue is True, the caller should apply another round of modified
    girvan newman on each of the returned component dataframes. continue = False
    occurs when no edges can be removed from the component (community found), or
    when the component is too small, in which case None is returned in place of
    the dataframe.
    
    """
    component = graph_utils.prune(component)
    # Return None if component too small - how to do without compute?
    
    edge_scores = edge_scoring.get_edge_scores(component)
    upper_score_threshold = edge_scoring.get_upper_threshold(edge_scores)
    new_df, num_chopped = edge_scoring.chop(
        component, edge_scores, upper_score_threshold)
    
    if num_chopped == 0: # Community found - don't continue processing
        return [(new_df, False)]
    
    # Further processing required
    new_components = graph_utils.get_components(new_df)
    to_return = []
    for component in new_components:
        to_return.append((component, True))
    return to_return