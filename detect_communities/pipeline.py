""" Main / Controller Community Detection Script 

run() : orchestrator function; repeatedly submits components to modified
girvan newman to be broken down until no more components can be broken
down, then returns the tagged connectome


"""

from dask import dataframe as ddf
from dask.distributed import Client

type DDF = ddf.DataFrame


def run(connectome: DDF, args) -> DDF:
    """ Submit component jobs then tag connectome with community IDs """
    # Set-up controller including active future set
    client = get_client()
    active, communities = set(), []
    init_comps = client.submit(graph_utils.get_components, connectome).result()
    for comp in init_comps:
        f = client.submit(mgn, comp, args.k)
        active.add(f)
    
    while active: # While at least one component is still being/needs processed
        done, active = client.wait(active, return_when='FIRST_COMPLETED')
        for future in done:
            og_comp, chopped = future.result()
            
            # Get number of edges in original and chopped dataframe
            f = [client.submit(count_edges, chopped), 
                 client.submit(count_edges, og_comp)]
            edges_ch, edges_comp = client.gather(f)
            
            # Decide what to do based on number of edges and size
            if edges_ch < edges_comp: # At least one edge was removed
                new_comps = client.submit(graph_utils.get_components, chopped).result()
                new_sizes = [client.submit(count_edges, nc) for nc in new_comps]
                new_sizes = client.gather(new_sizes)
                
                for new_comp, size in zip(new_comps, new_sizes):
                    # Discard if too small, otherwise process further
                    if size > args.minsize: 
                        f = client.submit(mgn, new_comp, args.k)
                        active.add(f)
            
            else: # Same number of edges as before - community found
                communities.append(chopped)        


def count_edges(df):
    return df.map_partitions(len).sum().compute()


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
    
    big_community_df = ddf.concat(communities).persist()
    tagged = connectome.merge(big_community_df, 
                              on = ["pre", "post"], 
                              how="left")
    tagged = tagged.sort_values(["neuropil", "community_id"]).persist()
    return tagged


def mgn(component: DDF, k: float) -> tuple[DDF]:
    """ Remove bridge edges from component dataframe
    
    Modified Girvan Newman. k is used in get_upper_threshold() as a multiplier.
    Returns the original component and the 'chopped' dataframe.
    
    """
    component = graph_utils.prune(component)
    edge_scores = edge_scoring.get_edge_scores(component)
    upper_score_threshold = edge_scoring.get_upper_threshold(edge_scores, k)
    new_df = edge_scoring.chop(component, edge_scores, upper_score_threshold)
    return (component, new_df)