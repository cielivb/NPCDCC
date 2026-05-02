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

def prune(df: ddf.Dataframe) -> ddf.Dataframe:
    """ Iteratively remove degree 1 edges from a dask dataframe 
    
    A degree 1 edge is defined here as an edge associated with at least one
    degree 1 node, where a degree 1 node is a node connected by any number of 
    edges to one and only one other node. No nodes in the dataset will have 
    edges to themselves. Synapse count and directionality are not considered.
    
    """
    # Extract the edges as a set (in a bag)
    
    # Add (b,a) for every (a,b) in set to the set (makes it undirected)
    
    # Create dict where key is node id, and val is set of neighbour nodes
    
    # Queue all degree 1 nodes (nodes with only one neighbour)
    
    # While queue not empty:
    #    Pop the degree 1 node from queue
    #    Identify neighbour
    #    Remove deg1 node from neighbour's connections
    #    If neighbour is now deg1, add neighbour to queue
    #    Remove deg1 node key from dict
    
    # Intersect remaining edges with original dataframe to get pruned df
    # Return pruned df
    
    pass