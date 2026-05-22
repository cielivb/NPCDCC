""" Main / Controller Community Detection Script 

run() : orchestrator function; repeatedly submits components to modified
girvan newman to be broken down until no more components can be broken
down, then returns the tagged connectome


"""
import dask
from dask import dataframe as ddf
from dask.distributed import Client

type DDF = ddf.DataFrame


def get_worker_df_size(client) -> float:
    """ Return maximum pandas dataframe size based on worker RAM allowance """
    # Get the minimum worker RAM allocation
    # Calculate safe dataframe size (0.25 x min RAM allocation)
    # Return safe dataframe size
    raise NotImplementedError


def send_to_worker(component, max_worker_df_size):
    """ Return True if worker can handle component size """
    # Get component size
    # True if component_size < max_worker_df_size else False
    raise NotImplementedError


def is_community(og_comp, proc_comp) -> bool:
    """ True if both components have same number of rows """
    num_edges_og = og_comp.map_partitions(len).sum()
    num_edges_proc = proc_comp.map_partitions(len).sum()
    num_edges_og, num_edges_proc = dask.compute(num_edges_og, num_edges_proc)
    return True if num_edges_og == num_edges_proc else False


def run(connectome: DDF, args) -> DDF:
    """ Detect and label communities in connectome. 
    
    The control flow involves the driver serially processing components that are
    too large to fit within worker memory as pandas dataframes, while using 
    workers to process components that fit within worker RAM limits in parallel.
    
    Goal 1: keep workers busy while the driver manhandles giant components.
    
    Goal 2: parallelise as much as possible while respecting the constraints of
    parallel breadth first search (PBFS). PBFS is the core algorithm supporting
    community detection, but its control flow means it requires computes on
    dask dataframes, meaning workers cannot run a pure dask PBFS on their
    assigned components. Workers must run a pandas PBFS on their components 
    instead, but this requires bringing dataframes into memory. It is crucial
    to separate components that can be processed by workers from components that
    are too large for workers.
    
    """
    # Set-up controller variables
    client = get_client()
    max_worker_df_size = get_worker_df_size(client)
    big_comps, small_comps, communities = set(), set(), []
    
    def decide_fates(proc_comp, og_comp=None):
        """ Inspect girvan newman results to decide component fates """
        if og_comp and is_community(og_comp, proc_comp):
            communities.append(og_comp)
        else:
            new_comps = graph_utils.get_components(proc_comp)
            for comp in new_comps:
                if send_to_worker(comp, max_worker_df_size):
                    f = client.submit(worker_gn, comp)
                    small_comps.add(f)
                else:
                    big_comps.add(comp)       
                
    # Get and allocate initial components
    init_components = graph_utils.get_components(connectome)
    decide_fates(init_components)
            
    # Enter control loop - continue breaking connectome components down into
    # communities until no more components remain to be processed
    while len(big_comps) > 0 or len(small_comps) > 0:
        
        # Process one big component
        if len(big_comps) > 0:
            big_comp = big_comps.pop()
            proc_comp = driver_gn(big_comp)
            decide_fates(proc_comp, big_comp)
            
        # Decide what to do with finished small components
        if len(small_comps) > 0:
            done, small_comps = client.wait(small_comps, return_when="FIRST_COMPLETED")
            for og_comp, proc_comp in done:
                decide_fates(proc_comp, big_comp)
                
    # All communities found - can now tag connectome
    return tag(connectome, communities).persist()


def tag(connectome: DDF, communities: list[DDF]) -> DDF:
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


#def mgn(component: DDF, k: float) -> tuple[DDF]:
    #""" Remove bridge edges from component dataframe
    
    #Modified Girvan Newman. k is used in get_upper_threshold() as a multiplier.
    #Returns the original component and the 'chopped' dataframe.
    
    #"""
    #component = graph_utils.prune(component)
    #edge_scores = edge_scoring.get_edge_scores(component)
    #upper_score_threshold = edge_scoring.get_upper_threshold(edge_scores, k)
    #new_df = edge_scoring.chop(component, edge_scores, upper_score_threshold)
    #return (component, new_df)