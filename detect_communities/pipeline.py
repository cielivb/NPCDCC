""" Main / Controller Community Detection Script 

run() : orchestrator function; repeatedly submits components to modified
girvan newman to be broken down until no more components can be broken
down, then returns the tagged connectome


"""
import dask
from dask.distributed import wait
from dask import dataframe as ddf
from dask.distributed import Client
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame


def get_worker_df_size(client) -> float:
    """ Return maximum pandas dataframe size based on worker RAM allowance """  
    # Get the minimum worker RAM allocation
    all_info, min_ram = client.scheduler_info(), None
    for worker, info in all_info["workers"].items():
        gb = info["memory_limit"] / (1024**3)
        min_ram = gb if (not min_ram or gb < min_ram) else min_ram
    
    # Calculate and return safe dataframe size (0.25 x min RAM allocation)
    safe_size = 0.25 * min_ram # in gigabytes
    return safe_size


def should_send_to_worker(component: DDF, max_worker_df_size: float):
    """ Return True if worker can handle dask dataframe component size """
    comp_gb = component.memory_usage(deep = False).sum().compute() / 1024**3
    return comp_gb < max_worker_df_size


def is_community(og_comp, proc_comp) -> bool:
    """ True if both components have same number of rows """
    num_edges_og = og_comp.map_partitions(len).sum()
    num_edges_proc = proc_comp.map_partitions(len).sum()
    num_edges_og, num_edges_proc = dask.compute(num_edges_og, num_edges_proc)
    return True if num_edges_og == num_edges_proc else False


def decide_fates(client, awaiting_fates: set[tuple]):
    """ Inspect girvan newman results to decide component fates. 
    Each tuple in awaiting_fates set is (proc_comp, og_comp)
    """
    communities, needs_further_processing = set(), set()
    for proc_comp, og_comp in awaiting_fates:
        if og_comp and is_community(og_comp, proc_comp):
            communities.add(og_comp)
        else:
            needs_further_processing.add(proc_comp)
    
    future_fates = set()
    for comp in needs_further_processing:
        f = client.submit(graph_utils.get_components, proc_comp)
        future_fates.add(f)
    
    return communities, future_fates


def process_fates(client, future_fates, max_worker_df_size):
    """ Decide whether each component should be processed by a worker.
    Each future in future_fates will be a list of component dataframes """
    big_comp, small_comp = set(), set()
    if future_fates:
        done, future_fates = client.wait(future_fates, return_when="FIRST_COMPLETED")
        for comp_list in done:
            for comp in comp_list:
                if should_send_to_worker(comp, max_worker_df_size):
                    repartitioned = comp.repartition(npartitions=1) # CRITICAL
                    small_comp.add(repartitioned)
                else:
                    big_comp.add(comp)
    return future_fates, big_comp, small_comp


def process_end_of_pipeline(client, chop_futures):
    """ Process small components that have reached the end of the pipeline """
    results = set()
    if chop_futures:
        done, chop_futures = client.wait(chop_futures, return_when="FIRST_COMPLETED")
        for proc_comp, og_comp in done:
            results.add((proc_comp, og_comp))
    return chop_futures, results


def process_agg_futures(client, agg_futures, k):
    """ Process small components that are ready to be 'chopped' (i.e., have 
    final/aggregated edge scores available)"""
    chop_futures = set()
    if agg_futures:
        done, agg_futures = client.wait(agg_futures, return_when="FIRST_COMPLETED")
        for comp, score_df in done:
            f = client.submit(edge_scoring.chop_pd, comp, score_df, k)
            chop_futures.add(f)    
    return agg_futures, chop_futures


def process_esp_dict(client, esp):
    """ Process small components that have all initial edge scores available """
    agg_futures = set()
    for comp_id, comp_tup in esp.items():
        comp, future_list = comp_tup[0], comp_tup[1]
        ready = all(f.done() for f in future_list)
        if ready:
            f = client.submit(edge_scoring.aggregate_scores_pd, comp, future_list)
            agg_futures.add(f)
            del esp[comp_id]
    return esp, agg_futures


def process_pruned_futures(client, pruned_futures, esp, next_id):
    """ Parallelise getting initial edge scores for each small pruned component """
    if pruned_futures:
        done, pruned_futures = client.wait(pruned_futures, return_when="FIRST_COMPLETED")
        for comp in done:
            comp_id = next_id
            next_id += 1
            start_nodes = get_start_nodes(comp).compute()
            for s in start_nodes:
                f = client.submit(edge_scoring.get_scores_pd, comp, s)
                if comp_id in esp:
                    esp[comp_id][1].add(f)
                else:
                    esp[comp_id] = (comp, {f})
    return pruned_futures, esp, next_id


def process_to_prune(client, to_prune):
    """ Prune small components in to_prune """
    pruned_futures = set()
    if to_prune:
        for comp in to_prune:
            f = client.submit(prune_pd, comp)
            pruned_futures.add(f)
    return pruned_futures
            
    
def run(connectome: DDF, args) -> DDF:
    """ Detect and label communities in connectome. 
    
    The control flow involves the driver serially processing components that are
    too large to fit within worker memory as pandas dataframes, while using 
    workers to process components that fit within worker RAM limits in parallel.
    
    Goal 1: keep workers busy while the driver manhandles giant components.
    
    Goal 2: parallelise as much as possible while respecting the constraints of
    parallel breadth first search (PBFS). PBFS is the core algorithm supporting
    community detection, but its control flow means it requires computes on
    dask dataframes, meaning workers cannot usually run a pure dask PBFS on their
    assigned components. Workers must run a pandas PBFS on their components 
    instead, but this requires bringing dataframes into memory. It is crucial
    to separate components that can be processed by workers from components that
    are too large for workers.
    
    """    
    # Set up initial variables
    client, next_id = get_client(), 0
    max_worker_df_size = get_worker_df_size(client)
    agg_futures, chop_futures = set(), set()
    esp = dict() # edge score progress dictionary
    to_prune, awaiting_fates = set(), set()
    
    # Set the pre column of connectome to be the index
    connectome = connectome.set_index("pre", drop=False)
                    
    # Get and allocate initial components
    communities, future_fates = decide_fates(client, set((connectome, None)))
    connectome_future_fate = future_fates.pop()
    wait(connectome_future_fate)
    future_fates, big_comps, small_comps = process_fates(
        client, future_fates, max_worker_df_size)
    
    # Enter control loop - continue breaking connectome components down into
    # communities until no more components remain to be processed
    while len(big_comps) > 0 or len(small_comps) > 0:
        
        # Process one big component
        if len(big_comps) > 0:
            proc_comp, og_comp = driver_mgn(big_comps.pop())
            awaiting_fates.add((proc_comp, og_comp))
        
        # Process small components that have reached the end of the pipeline
        chop_futures, results = process_end_of_pipeline(client, chop_futures)
        for proc_comp, big_comp in results:
            awaiting_fates.add((proc_comp, big_comp))
        
        # Process and assign new components to driver or workers
        future_fates, big, small = process_fates(
            client, future_fates, max_worker_df_size)
        big_comps.update(big), to_prune.update(small)
        
        # Process results of big component processing and end of small pipeline
        new_comms, new_future_fates = decide_fates(client, awaiting_fates)
        communities.update(new_comms)
        awaiting_fates = set()

        # Shuffle small and ready components one step forward through pipeline
        agg_futures, new_chop_futures = process_agg_futures(
            client, agg_futures, args.madk)
        chop_futures.update(new_chop_futures)
        esp, new_agg_futures = process_esp_dict(client, esp)
        agg_futures.update(new_agg_futures)
        pruned_futures, esp, next_id = process_pruned_futures(
            client, pruned_futures, esp, next_id)
        pruned_futures.update(process_to_prune(client, to_prune))
        to_prune = set()
    
    return tag(connectome, communities)


def tag(connectome: DDF, communities: list[DDF]) -> DDF:
    """ Return connectome sorted by neuropil and tagged with community IDs. 
    Community IDs are simple integers. Allocate each cluster dataframe in the
    list a community ID, concatenate the cluster dataframes, then left merge
    connectome_df onto cluster_dfs.
    """
    if not communities:
        return None
    community_ids = range(1, len(communities) + 1)
        
    tagged_communities = []
    for community, community_id in zip(communities, community_ids):
        tagged_communities.append(community.assign(community_id=community_id))
    
    big_community_df = ddf.concat(tagged_communities).persist()
    tagged = connectome.merge(big_community_df, 
                              on = ["pre", "post"], 
                              how="left")
    tagged = tagged.sort_values(["neuropil", "community_id"]).persist()
    return tagged.persist()


def driver_mgn(component: DDF, k: float) -> tuple[DDF]:
    """ Remove bridge edges from component dataframe
    
    Modified Girvan Newman. k is used in get_upper_threshold() as a multiplier.
    Returns the original component and the 'chopped' dataframe.
    
    """
    pruned_component = graph_utils.prune(component)
    edge_scores = edge_scoring.get_edge_scores(pruned_component)
    upper_score_threshold = edge_scoring.get_upper_threshold(edge_scores, k)
    new_df = edge_scoring.chop(pruned_component, edge_scores, upper_score_threshold)
    return (new_df, component)