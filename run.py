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
from dask import bag as db
from dask import dataframe as ddf
from queue import Queue


### CLUSTER IDENTIFICATION SUPPORT FUNCTIONS ------------------------------

def prune(df: ddf.DataFrame) -> ddf.DataFrame:
    """ Iteratively remove degree 1 edges from a dask dataframe 
    
    A degree 1 edge is defined here as an edge associated with at least one
    degree 1 node, where a degree 1 node is a node connected by any number of 
    edges to one and only one other node. No nodes in the dataset will have 
    edges to themselves. Synapse count and directionality are not considered.
    
    """
    edge_bag = df[["pre","post"]].to_bag()
    # Add (b,a) for every (a,b) in set to the set (makes it undirected)
    edge_bag = edge_bag.map(lambda edge: [edge, (edge[1], edge[0])])
    edge_bag = edge_bag.flatten().distinct()
    
    # Create list of tuples where tup[0] is node id, and tup[1] is list of 
    # neighbour nodes. This makes it easy to track retained edges and degree 
    # simultaneously. Computing here saves many repeated computations during
    # the iterative while loop, but sacrifices some RAM.
    grouped_edges = edge_bag.foldby(key = lambda edge: edge[0],
                                    binop = lambda accum, edge: accum + [edge[1]],
                                    initial = [],
                                    combine = lambda accum1, accum2: accum1 + accum2,
                                    combine_initial = []).compute()
    # Find initial degree 1 nodes
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
                
    # Get expanded list of tuples of all edges post-pruning
    def expand(tup):
        """ e.g., edge_data = (0, [1, 5]) -> [(0,1), (0,5)] """
        return list(map(lambda target: (tup[0], target), tup[1]))
    new_edge_bag = db.from_sequence(grouped_edges).map(lambda tup: expand(tup)).flatten()
    
    # Intersect remaining edges with original dataframe to get pruned df    
    new_edge_df = new_edge_bag.to_dataframe(meta={"pre": int, "post": int})
    pruned_df = new_edge_df.merge(df, on=["pre","post"], how="inner")
    return pruned_df