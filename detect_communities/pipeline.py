""" Main / Controller Community Detection Script 

run() : orchestrator function; repeatedly submits components to modified
girvan newman to be broken down until no more components can be broken
down, then returns the tagged connectome


"""
import dask
from dask.distributed import wait
from dask import dataframe as ddf
from dask.distributed import Client
from queue import Queue
from threading import Lock
from time import sleep
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame

BIG_COMPS = Queue()
SMALL_COMPS = Queue()
COMMUNITIES = Queue()
MAX_WORKER_DF_SIZE = None
ESP = dict() # Edge Scoring Progress
NEXT_ID, LOCK = 1, Lock()
NUM_SMALL_PROCESSED = 0


def get_next_comp_id():
    """ Thread-safe component ID retrieve and update """
    global NEXT_ID, LOCK
    with LOCK:
        to_return = NEXT_ID
        NEXT_ID += 1
    return to_return


def increment_num_small_processed():
    """ Thread-safe increment of global number processed tracker """
    global NUM_SMALL_PROCESSED, LOCK
    with LOCK:
        NUM_SMALL_PROCESSED += 1
    

def get_worker_df_size(client) -> float:
    """ Return maximum pandas dataframe size based on worker RAM allowance """  
    # Get the minimum worker RAM allocation
    all_info, min_ram = client.scheduler_info(), None
    for worker, info in all_info["workers"].items():
        gb = info["memory_limit"] / (1024**3)
        min_ram = gb if (not min_ram or gb < min_ram) else min_ram
    
    # Calculate and return safe in-memory dataframe size (0.2 x min RAM allocation)
    safe_size = 0.2 * min_ram # in gigabytes
    return safe_size


def should_send_to_worker(component: DDF, max_worker_df_size: float):
    """ Return True if worker can handle dask dataframe component size """
    global MAX_WORKER_DF_SIZE
    comp_gb = component.memory_usage(deep = False).sum().compute() / 1024**3
    return comp_gb < MAX_WORKER_DF_SIZE


def is_community(proc_comp, og_comp) -> bool:
    """ True if both components have same number of rows """
    if not og_comp or not proc_comp: return False
    num_edges_og = og_comp.map_partitions(len).sum()
    num_edges_proc = proc_comp.map_partitions(len).sum()
    num_edges_og, num_edges_proc = dask.compute(num_edges_og, num_edges_proc)
    return True if num_edges_og == num_edges_proc else False


def decide_fate(result):
    """ Inspect girvan newman results to decide component fates. 
    Each tuple is of form (proc_comp, og_comp)
    """
    global BIG_COMPS, SMALL_COMPS, COMMUNITIES
    if isinstance(result, dask.distributed.Future):
        proc_comp, og_comp = result.result()
    else: # Special case with original connectome
        proc_comp, og_comp = result
    if is_community(proc_comp, og_comp):
        COMMUNITIES.put(proc_comp)
    else: # get_components involves computes -> driver only
        new_comps = graph_utils.get_components(proc_comp) # Blocking
        for comp in new_comps:
            if should_send_to_worker(comp):
                SMALL_COMPS.put(comp)
            else:
                BIG_COMPS.put(comp)


def process_pruned(pruned_future):
    """ Get edge scores of pruned component """
    global ESP
    pruned = pruned_future.result()
    start_nodes = edge_scoring.get_start_nodes(pruned)
    comp_id = get_next_comp_id()
    ESP[comp_id] = (pruned, set())
    for s in start_nodes:
        f = client.submit(edge_scoring.get_scores_pd, pruned, s)
        ESP[comp_id][1].add(f)


def process_aggregated(aggregated_future):
    """ Chop the dataframe based on edge scores then decide its fate """
    comp_w_scores = aggregated_future.result()
    f = client.submit(edge_scoring.chop_pd, comp_w_scores)
    f.add_done_callback(decide_fate)


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
    global CLIENT, BIG_COMPS, SMALL_COMPS, COMMUNITIES, MAX_WORKER_DF_SIZE
    global NUM_SMALL_PROCESSED
    
    # Set up initial variables
    CLIENT, MAX_WORKER_DF_SIZE = get_client(), get_worker_df_size(CLIENT)
    num_small_submitted = 0    
    
    # Only need 3 columns for community detection
    trimmed = connectome[["pre", "post", "syn_count"]]
    
    # Undirect connectome and set the pre column of connectome to be the index
    undirected_connectome = graph_utils.undirect_df(trimmed)
    undirected_connectome = undirected_connectome.set_index("pre", drop = False)
                    
    # Get and allocate initial components
    decide_fate((undirected_connectome, None))
    
    # Enter control loop - continue breaking connectome components down into
    # communities until no more components remain to be processed
    while not (BIG_COMPS.empty and SMALL_COMPS.empty):
        
        # Submit new small components to start of small components pipeline
        while not SMALL_COMPS.empty:
            to_prune = SMALL_COMPS.get()
            f = CLIENT.submit(graph_utils.prune_pd, to_prune)
            f.add_done_callback(process_pruned)
            num_small_submitted += 1

        # Push small components that have completed the PBFS stage forward
        # through pipeline past the artificial MGN PBFS synchronisation barrier
        for comp_id, comp_tuple in ESP.items():
            comp, future_list = comp_tup[0], comp_tup[1]
            if all(f.done() for f in future_list): # if ready
                scores_list = [f.result() for f in future_list] # list of DDF
                f = client.submit(edge_scoring.aggregate_scores, comp, scores_list)
                f.add_done_callback(process_aggregated)
        
        # Process one big component
        if not BIG_COMPS.empty:
            proc_comp, og_comp = driver_mgn(BIG_COMPS.get()) # Blocks
            decide_fate((proc_comp, og_comp)) # Blocks
        else:
            sleep(5) # Don't waste CPU cycles
    
    # Wait for all components in small pipeline to finish being processed.
    while num_small_submitted < NUM_SMALL_PROCESSED:
        sleep(5)
    
    return tag(connectome, COMMUNITIES)


def tag(connectome: DDF, communities_q: Queue) -> DDF:
    """ Return connectome sorted by neuropil and tagged with community IDs. 
    Community IDs are simple integers. Allocate each cluster dataframe in the
    list a community ID, concatenate the cluster dataframes, then left merge
    connectome_df onto cluster_dfs.
    """
    if communities_q.empty: return None
    
    # Tag communities while draining communities_q
    communities, community_id = [], 0
    while not communities_q.empty():
        community = communities_q.get()
        community["community_id"] = community_id
        community_id += 1
    
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
    pruned = graph_utils.prune(component)
    start_nodes = edge_scoring.get_start_nodes(pruned)
    score_dfs = []
    for s in start_nodes: # This may take a while!
        score_dfs.append(edge_scoring.get_scores(pruned, s))
    fully_scored = edge_scoring.aggregate_scores(pruned, score_dfs)
    proc_comp, og_pruned_comp = edge_scoring.chop(fully_scored, k)
    return (proc_comp, og_pruned_comp)