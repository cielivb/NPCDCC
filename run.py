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
    # Extract the edges as a set (in a dask bag)
    edge_bag = df[["pre","post"]].to_bag()
    
    # Add (b,a) for every (a,b) in set to the set (makes it undirected)
    edge_bag = edge_bag.map(lambda edge: [edge, (edge[1], edge[0])])
    edge_bag = edge_bag.flatten().distinct()
    
    # Create dict where key is node id, and val is list of neighbour nodes.
    # This makes it easy to track retained edges and degree simultaneously.
    edge_bag_tuple_groups = edge_bag.foldby(key = lambda edge: edge[0],
                                            binop = lambda accum, edge: accum + [edge[1]],
                                            initial = [],
                                            combine = lambda accum1, accum2: accum1 + accum2,
                                            combine_initial = [])
    edge_bag_dict = edge_bag_tuple_groups.map(lambda tup: {tup[0]: tup[1]})    
    
    deg1_queue = Queue() # Queue all nodes with only one neighbour
    
    while not deg1_queue.empty:
        # Neighbour is None if deg1_node already pruned away, else [neighbour_id]
        neighbour = edge_bag_dict.get(deg1_queue.pop(), None)
        if neighbour:
            neighbour = neighbour[0] # Neighbour ID is the only item in list
            
            # Remove deg1 node from neighbour's connections 
            # (i.e.,remove neighbour -> deg1 edge)
            edge_bag_dict[neighbour].remove(deg1_node)
            
            # If neighbour is now deg1, add neighbour to deg1 queue
            if len(edge_bag_dict[neighbour]) == 1:
                deg1_queue.put(neighbour)
                
            # Remove deg1 node key from dict (remove deg1 -> neighbour edge)
            del edge_bag_dict[deg1_node]
        
    # Get expanded list of tuples of all edges post-pruning
    def expand(edge_data):
        """ e.g., edge_data = {0: [1, 5]} -> [(0,1), (0,5)] """
        key, target_list = edge_data.popitem()
        return list(map(lambda target: (key, target), target_list))
    new_edge_bag = edge_bag_dict.map(lambda edge_data: expand(edge_data)).flatten()    
    
    # Intersect remaining edges with original dataframe to get pruned df    
    new_edge_df = new_edge_bag.to_dataframe(meta={"pre": int, "post": int})
    pruned_df = new_edge_df.merge(df, on=["pre","post"], how="inner")
    return pruned_df