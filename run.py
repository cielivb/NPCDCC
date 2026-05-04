""" Computing Excitatory-Inhibitory Neurotransmitter Ratios of Discrete 
Subnetworks of the Drosophila Connectome 

Program Author: Ciel Baumann


--- DATA

Dataset: FlyWire Whole-brain Connectome Connectivity Data
Dataset Retrieved From: https://zenodo.org/records/10676866
Dataset Version: 783.0
Dataset Published By: Flywire Consortium

Data Files used:
- proofread_connections_783.feather
- proofread_root_ids_783.npy

Dataset Citation (APA):
FlyWire Consortium. (2024). FlyWire Whole-brain Connectome Connectivity Data 
  (783.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10676866
  
  
--- USAGE

TODO



--- CONTENTS

TODO


"""

import dask
import numpy as np
from dask import bag as db
from dask import dataframe as ddf
from queue import Queue


### CLUSTER IDENTIFICATION - HELPER FUNCTIONS -----------------------------

def edge_df_to_tuple(df: ddf.DataFrame, undirect=True) -> list[tuple[int,list[int]]]:
    """ Used in prune() and bfs_components() """
    edge_bag = df[["pre","post"]].to_bag()
    if undirect:
        # Add (b,a) for every (a,b) in set to the set (makes it undirected)
        edge_bag = edge_bag.map(lambda edge: [edge, (edge[1], edge[0])])
        edge_bag = edge_bag.flatten().distinct()
    
    # Create list of tuples where tup[0] is node id, and tup[1] is list of 
    # neighbour nodes. This makes it easy to track retained edges and/or degree. 
    # Computing here saves many repeated computations during the iterative while 
    # in the prune function, but sacrifices some RAM.
    grouped_edges = edge_bag.foldby(key = lambda edge: edge[0],
                                    binop = lambda accum, edge: accum + [edge[1]],
                                    initial = [],
                                    combine = lambda accum1, accum2: accum1 + accum2,
                                    combine_initial = []).compute()
    return grouped_edges


def grouped_edge_tuples_to_df(tuple_iter, original_df: ddf.DataFrame):
    """ Merge compact tuple edge representation with original dataframe 
    
    Tuple edges are expanded into one tuple per edge format. Then for every
    edge in the original dataframe, if that edge is present in expanded tuple
    format, that edge in the original dataframe is retained in a new dataframe,
    which is then returned.
    """
    # Get iterable of expanded tuples (one tuple per edge)
    def expand(tup):
        """ e.g., edge_data = (0, [1, 5]) -> [(0,1), (0,5)] """
        return list(map(lambda target: (tup[0], target), tup[1]))
    new_edge_bag = db.from_sequence(tuple_iter).map(lambda tup: expand(tup)).flatten()
    
    # Intersect remaining edges with original dataframe
    new_edge_df = new_edge_bag.to_dataframe(meta={"pre": int, "post": int})
    new_df = new_edge_df.merge(original_df, on=["pre","post"], how="inner")
    return new_df


### CLUSTER IDENTIFICATION - PRUNE ----------------------------------------

def prune(df: ddf.DataFrame) -> ddf.DataFrame:
    """ Iteratively remove degree 1 edges from a dask dataframe 
    
    A degree 1 edge is defined here as an edge associated with at least one
    degree 1 node, where a degree 1 node is a node connected by any number of 
    edges to one and only one other node. No nodes in the dataset will have 
    edges to themselves. Synapse count and directionality are not considered.
    
    """
    grouped_edges = edge_df_to_tuple(df)
    deg1_nodes = list(map(lambda tup: tup[0], 
                          filter(lambda tup: len(tup[1]) == 1, grouped_edges)))
    
    # Queue all degree 1 nodes
    # This is a bit hacky (using private vars) but avoids a for loop
    # Idea from https://www.py4u.org/blog/python-putting-list-items-in-a-queue/#3-better-code-practices-for-optimization
    deg1_queue = Queue()        
    with deg1_queue.mutex: # with queue lock
        deg1_queue.queue.extend(deg1_nodes) # Add all nodes at once
        
    # Iteratively prune away degree 1 nodes from edge_bag_dict
    while not deg1_queue.empty():
        deg1_node = deg1_queue.get()
        deg1_node_in_tree = len(list(
            filter(lambda tup: tup[0] == deg1_node, grouped_edges))) == 1
        if not deg1_node_in_tree: continue
        neighbour = list(map(lambda tup: tup[1], 
                             filter(lambda tup: tup[0] == deg1_node, grouped_edges)))[0][0]
        
        # Remove deg1 node from neighbour's connections 
        # (i.e.,remove neighbour -> deg1 edge)
        neighbour_neighbours = list(map(lambda tup: tup[1], 
                                        filter(lambda tup: tup[0] == neighbour, grouped_edges)))[0]
        neighbour_neighbours.remove(deg1_node)
        # If neighbour is now deg1, add neighbour to deg1 queue
        if len(neighbour_neighbours) == 1:
            deg1_queue.put(neighbour)
            
        # Remove deg1 node key from dict (remove deg1 -> neighbour edge)
        deg1_tuple = list(filter(lambda tup: tup[0] == deg1_node, grouped_edges))[0]
        grouped_edges.remove(deg1_tuple)
                
    return grouped_edge_tuples_to_df(grouped_edges, df)


### CLUSTER IDENTIFICATION - BFS COMPONENT SEARCH ------------------------=

def bfs_loop(grouped_edges, nodes, queue, state):
    """ Discover a component """
    while not queue.empty():
        node = queue.get()
        node_index = np.where(nodes == node)[0][0]
        node_neighbours = grouped_edges[node_index][1]
        for neighbour in node_neighbours:
            neighbour_index = np.where(nodes == neighbour)[0][0]
            if state[neighbour_index] == "U":
                state[neighbour_index] == "D"
                queue.put(neighbour)
        state[node_index] = "P"


def bfs_components(df: ddf.DataFrame, min_size=30) -> db.Bag:
    """ Use BFS to return components of df as DataFrames in a Bag """
    grouped_edges = edge_df_to_tuple(df)
    n = len(grouped_edges)
    state = np.full(n, "U", dtype="<U1")
    queue = Queue()
    components = None # Will later be a dask bag of dask dataframe/s
    
    # Iterate through each node in grouped_edges to get components via BFS
    nodes = np.fromiter(map(lambda tup: tup[0], grouped_edges), dtype=int)
    for node_index in range(n):
        
        if state[node_index] == "U":
            prev_state = state.copy()
            state[node_index] == "D"
            queue.put(node_index)
            
            bfs_loop(grouped_edges, nodes, queue, state) # Discover component
            
            # Add new component if large enough
            diff_indices = np.where(state != prev_state)[0]
            if len(diff_indices) < min_size:
                continue
            component_edges = filter(lambda tup: grouped_edges.index(tup) in diff_indices,
                                     grouped_edges)
            component_df = grouped_edge_tuples_to_df(component_edges, df)
            if not components:
                components = db.from_sequence([component_df])
            else:
                components = db.concat([components, db.from_sequence([component_df])])
                
    return components
        
    