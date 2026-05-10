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
import os
import random
from collections import Counter
from collections import defaultdict
from dask import bag as db
from dask import dataframe as ddf
from dask.distributed import Client

CLIENT = None # Assigned properly at bottom of script
MIN_CLUSTER_SIZE = 30
MAD_K = 3.5

ROOT_DIR = os.path.dirname(__file__)



### CLUSTER IDENTIFICATION - HELPER FUNCTIONS -----------------------------

def df_to_adjacency_bag(df, undirect=True):
    """ Convert dataframe edges into adjacency list/bag of edges with weights.
    
    Return an adjacency list for the dataframe as a dask bag of the form
        db.from_sequence([(a, [(b, 3)]), (b: [(a, 3)])])
    if undirected, otherwise
        db.from_sequence([(a, []), (b, [(a, 3)])])
    where the edge from b->a (or b->a and a->b in the undirected version) has
    weight 3.
    
    Weight represents the number of synpases between two neurons a and b.
    
    For the DATA301 project, only the undirected version is required, but I
    have included the option for directed should I choose to extend the 
    project (I have not tested with the directed version).
    
    """
    if undirect: # Add (b,a,w) for every (a,b,w)
        edge_bag = df[["pre", "post", "syn_count"]].to_bag()        
        edge_bag = edge_bag.map(lambda edge: [edge, (edge[1], edge[0], edge[2])])
        edge_bag = edge_bag.flatten().distinct()
        adj_df = edge_bag.to_dataframe(
            meta = {"pre": int, "post": int, "syn_count": int})
    else:
        adj_df = df
        
    # grouped is of the general form
    # pre    post      syn_count
    # 0      [1, 5]    [2, 4]
    # where there is an edge of weight 2 between 0 and 1,
    # an edge of weight 4 between 0 and 5, etc
    grouped = adj_df.groupby("pre").agg({"pre": list, "post": list, "syn_count": list})
    
    # Each entry of form ([pre, pre, ...], [post, post, ...], [syn_count, syn_count, ...])
    # where all the pre values within an entry are equal. Edges are currently
    # represented by indices.
    grouped_as_bag = grouped.to_bag()
    adjacency_bag = grouped_as_bag.map( # Possible memory issue here?
        lambda entry: (entry[0][0], list(zip(entry[1], entry[2]))))
    return adjacency_bag.persist()


def get_all_nodes(adjacency_bag):
    """ Return bag of all nodes in graph represented by bag """
    main_nodes = adjacency_bag.map(lambda node_adj: node_adj[0])
    nnodes = adjacency_bag.map(
        lambda node_adj: node_adj[1]).flatten().map(lambda tup: tup[0])
    nodes = db.concat([main_nodes, nnodes])
    return nodes


def get_num_nodes(adjacency_bag):
    """ Return the number of nodes in the graph represented by bag """
    return get_all_nodes(adjacency_bag).count().compute()


def log_removed_edges(removed_edges):
    """ Log removed edges in a temp file. These edges can be analysed to address
    the research question in a similar manner as the clusters. """
    pass # TODO




### PARALLEL BFS ----------------------------------------------------------

class Layer():
    """ Represents a layer d of nodes at depth d of parallel BFS """
    
    def __init__(self):
        self.nodes = db.from_sequence([])
    
    
    def insert(self, node_bag: db.Bag):
        self.nodes = node_bag.persist() # bag of int nodes in layer e.g., bag([1,2,...])
    
    
    def is_empty(self):
        return self.nodes.count().compute() == 0

    
    def update_leaves(self, adjacencies, leaves, state, node_to_i):
        """ Add leaf nodes in adjacencies to leaves bag. 
        Leaf nodes are those with no children, i.e., none of their neighbours
        are undiscovered.
        """
        # TODO - this func is a slight bottleneck - fix it if time allows        
        no_weights = adjacencies.map(
            lambda node_adjacency: (node_adjacency[0], 
                                    list(map(lambda tup: tup[0], node_adjacency[1]))))
        neighbour_states = no_weights.map(
            lambda node_adjacency: (node_adjacency[0],
                                    list(map(
                                        lambda nnode: state[node_to_i[nnode]],
                                        node_adjacency[1])))).persist()
        del no_weights
        leaf_nodes = neighbour_states.filter(
            lambda tup: Counter(tup[1])["P"] == len(tup[1])).map(
                lambda tup: tup[0]).persist()
        del neighbour_states
        leaves = db.concat([leaves, leaf_nodes]).persist()  
        del leaf_nodes
        return leaves

    
    def get_child_parent_rels(self, adjacency_bag, adjacencies, state, node_to_i, all_children, all_child_parent_rels):
        """ Record parentage for each child 
        The child node's parents are those nodes in the child's adjacencies
        where the states of those nodes are 'P'."""
        # TODO - this func is a slight bottleneck - fix it if time allows
        global CLIENT
        child_ids = all_children.compute()
        child_adjacencies = adjacency_bag.filter(
            lambda node_adjacency: node_adjacency[0] in child_ids).persist()
        del child_ids
        parent_ids = adjacencies.map(
            lambda node_adjacency: node_adjacency[0]).filter(
                lambda node_id: state[node_to_i[node_id]] == "P").compute()
        child_parent_rels = child_adjacencies.map(
            lambda node_adjacency: (
                node_adjacency[0], 
                filter(lambda tup: tup[0] in parent_ids, node_adjacency[1])
            )).persist()
        del parent_ids, child_adjacencies
        all_child_parent_rels = db.concat([all_child_parent_rels, 
                                           child_parent_rels]).persist()
        del child_parent_rels
        return all_child_parent_rels
        
    
    def process(self, adjacency_bag, state, node_to_i, all_child_parent_rels, leaves):
        """ Check all neighbours of self.nodes for those that should be added
        to the next layer out_layer. Updates state, all_child_parent_rels, and
        leaves as required. Returns out_layer. 
        
        adjacency_bag is of the form:
            Bag([(a, [(b, 3)]), (b: [(a, 3), (c, 2)], ...]))
        where the edge from b->a and a->b has weight 3 (i.e., 3 synapses).
        """
        print("Setting up layer processing ...")
        # Set-up layer processing
        out_layer = Layer()
        layer_nodes = self.nodes.compute()
        adjacencies = adjacency_bag.filter( # Get adjacencies for this layer's nodes
            lambda node_adjacency: node_adjacency[0] in layer_nodes).persist()
        del layer_nodes
        print("Set up layer processing")
        
        print("Marking this layer's nodes as processed ...")
        # Mark this layer's nodes as processed. The child-parent rel and leaf
        # computation sections rely on parents being marked as P.
        processed_is = adjacencies.map( # Get this layer's node's indices
            lambda node_adjacency: node_to_i[node_adjacency[0]]).compute()
        state[processed_is] = "P"
        del processed_is
        print("Marked as processed")
        
        print("Getting leaf nodes ...")
        # Get leaf nodes (those with no child nodes)
        leaves = self.update_leaves(adjacencies, leaves, state, node_to_i)
        print("Leaf nodes retrieved")

        print("Getting children nodes ...")
        # Get the children nodes of this layer
        all_children = adjacencies.map(
            lambda tup: tup[1]).flatten().map(
                lambda tup: tup[0]).distinct().filter(
                    lambda node_id: state[node_to_i[node_id]] != "P").persist()
        print("Children nodes retrieved")
        
        # Discover child nodes and compute child-parent relationships
        if all_children.count().compute() > 0:
            print("Inserting children into next layer ...")
            out_layer.insert(all_children)
            print("Children inserted into next layer")
            print("Marking children as discovered ...")
            undiscovered_is = all_children.map(
                lambda child_id: node_to_i[child_id]).compute()
            state[undiscovered_is] = "D"
            del undiscovered_is
            print("Children marked as discovered")
            print("Getting child-parent relationships ...")
            all_child_parent_rels = self.get_child_parent_rels(
                adjacency_bag, adjacencies, state, node_to_i, all_children, all_child_parent_rels)
            print("Relationships retrieved")
        
        print("Returning PBFS data ...")
        return (out_layer, leaves, all_child_parent_rels)
        


def pbfs(start_node: int, adjacency_bag: db.Bag, state=None, nodes=None):
    """ Returns child_parent_rels bag, state array, leaves bag, and 
    num_shortest_paths bag.
    
    The numpy arrays nodes and state will be automatically computed from the
    adjacency bag if not supplied. state is not required to complete the bfs
    search, however, the caller (e.g., get_component_adjacency_bags()) may 
    wish to validate the status of each node after the search is complete.
    
    adjacency_bag has the general form
        db.from_sequence([(c, [(a, w), (b, w)]), (d, [(c, w)])])
    where a and b are parents of c, c is the only parent of d, and w is the
    edge weight or number of synapses along that edge.

    The bag of leaves contains nodes that do not have any children.
    
    The num_shortest_paths bag is of the general form
        db.from_sequence([(node1, num_paths), (node2, num_paths), ...])
    where num_paths is the number of shortest paths from start_node to node1 etc
    
    The original PBFS inspiration comes from the logic behind 'Bags' and 'Pennants'
    described at https://dl.acm.org/doi/epdf/10.1145/1810479.1810534. This is
    a simplified daskified implementation of their PBFS. What they term a 'bag'
    is here a 'Layer', which represents all the nodes at some leve/depth d in the 
    BFS tree.
    
    """
    # Create nodes and state arrays if not supplied
    supplied_nodes = nodes
    if not type(nodes) is np.ndarray:
        nodes = np.array(adjacency_bag.map(
            lambda node_adjacency: node_adjacency[0]).compute())
    if not type(state) is np.ndarray:
        state = np.full(len(nodes), "U", dtype="<U1")
    
    # Create look-up dict node_to_i and initialise state array.
    # node_to_i has key = node, value = corresponding index in state array.
    node_to_i = {node: i for i, node in enumerate(nodes)} 
    start_node_index = node_to_i[start_node]
    state[start_node_index] = "D"
    
    # The nodes array is no longer required for this function. If nodes was not
    # originally supplied, delete the nodes array. Don't delete the nodes 
    # array if it was supplied to avoid interfering with other functions.
    if not type(supplied_nodes) is np.ndarray:
        del nodes, supplied_nodes
    
    # Create empty leaves and all_child_parent_rels bags
    leaves = db.from_sequence([])
    all_child_parent_rels = db.from_sequence([(start_node, [])])
    
    # Initialise current layer (depth/level = 0)
    current_layer = Layer()
    current_layer.insert(db.from_sequence([start_node]))

    
    # Process each layer until max tree depth reached
    while not current_layer.is_empty():
        next_layer, leaves, all_child_parent_rels = current_layer.process(
            adjacency_bag, state, node_to_i, all_child_parent_rels, leaves)
        current_layer = next_layer

    return (all_child_parent_rels, state, leaves)


def get_component_adjacency_bags(df: ddf.DataFrame, undirected=True):
    """ Return a dask bag of adjacency lists/bags for each component in df.
   
    For each component in the graph represented in df, return the 
    adjacency list of the component containing each edge and their 
    weights. This is done by performing iteratively performing parallel BFS to 
    identify nodes belonging to different components. Only one component can
    be discovered at a time.
    
    Parent-child relationships require a start node, which is outside the
    scope of this function. See the bfs_search function.

    """
    big_adjacency_bag = df_to_adjacency_bag(df, undirected)
    nodes = np.array(big_adjacency_bag.map(
        lambda node_adjacency: node_adjacency[0]).compute())
    state = np.full(len(nodes), "U", "<U1") # nodes indices map to state indices
    components = [] # Will later be a dask bag of adjacency bags (one per component)
    
    # Iterate until all nodes are assigned to a component. Must find one
    # component at a time.
    for node_index in range(len(nodes)):
        if state[node_index] == "U":
            prev_state = state.copy()
            start_node = nodes[node_index]            
            
            all_child_parent_rels, state, leaves = pbfs(start_node, 
                                                        big_adjacency_bag,
                                                        state, nodes)
            del all_child_parent_rels, leaves
            
            # Add new component
            diff_indices = np.where(state != prev_state)[0]
            component_nodes = nodes[diff_indices]
            component_adj = big_adjacency_bag.filter(
                lambda node_adjacency: node_adjacency[0] in component_nodes)
            components = components + [component_adj.persist()]
    
    return db.from_sequence(components)




### CLUSTER IDENTIFICATION - PRUNE ----------------------------------------

def cut_deg1_edge(node_adj, should_cut: bool):
    """ Change node adjacency list to empty list if should cut. 
    This effectively makes the node a degree 0 node. """
    if should_cut:
        node_adj = (node_adj[0], [])
    return node_adj


def remove_deg_1_nodes(node_adj, deg1_nodes):
    """ Remove edges from node to nodes in degree 1 nodes """
    neighbours = list(filter(
        lambda tup: tup[0] not in deg1_nodes, node_adj[1]))
    return (node_adj[0], neighbours)


def prune(adjacency_bag: db.Bag) -> db.Bag:
    """ Iteratively remove degree 1 edges from a dask dataframe 
    
    A degree 1 edge is defined here as an edge associated with at least one
    degree 1 node, where a degree 1 node is a node connected by any number of 
    edges to one and only one other node. No nodes in the dataset will have 
    edges to themselves. Synapse count and directionality are not considered.
    
    Adjacency bag format:
            db.from_sequence([(a, [(b, 3)]), (b: [(a, 3)])])
            
    The naive approach would be to take away one edge at a time. This parallel
    version improves performance by finding all degree 1 edges initially, and
    pruning each 'chain' in parallel. The time to complete is the time it takes
    to process the longest 'chain'.

    """
    deg1_node_adjs = adjacency_bag.filter( # Get degree 1 nodes (deg 0 not present)
        lambda node_adj: len(node_adj[1]) == 1)
    
    while deg1_node_adjs.count().compute() > 0:
        deg1_nodes = deg1_node_adjs.map(
            lambda node_adj: node_adj[1][0][0]).compute()
        
        # Remove edge from deg1 node to neighbour
        adjacency_bag = adjacency_bag.map(
            lambda node_adj: cut_deg1_edge(node_adj, node_adj[0] in deg1_nodes))
        
        # Remove edge from neighbours to deg1 nodes
        neighbours = deg1_nodes_adj.map(
            lambda node_adjacency: node_adjacency[1][0][0])        
        adjacency_bag = adjacency_bag.map(
            lambda node_adj: remove_deg_1_nodes(node_adj, deg1_nodes))
        
        # Get fresh degree 1 nodes
        deg1_node_adjs = adjacency_bag.filter( # Get degree 1 nodes (deg 0 not present)
            lambda node_adj: len(node_adj[1] == 1))
        deg1_nodes = deg1_node_adjs.map(
            lambda node_adj: node_adj[1][0][0]).compute()
        
    return adjacency_bag





### CLUSTER IDENTIFICATION - GIRVAN NEWMAN --------------------------------


def calculate_edge_scores(start_node, child_parent_rels, num_shortest_paths, 
                          leaves, component):
    """ Return Bag of tuples Bag([((pre, post), edge_score), ...])
    Edge scoring rules are detailed on page 365 of MMDS Chapter 10 
    
    The num_shortest_paths bag is of the general form
        db.from_sequence([(node1, num_paths), (node2, num_paths), ...])
    where num_paths is the number of shortest paths from start_node to node1 etc
    
    """
    to_score = leaves
    edge_scores = db.from_sequence([])
    
    def credit(node):
        """ Rule 1: leaves get credit = 1. Rule 2: other nodes get credit = 1 + 
        sum of credits of the DAG edges from that node to its children """
        credit = 1 + edge_scores.filter(
            lambda entry: entry[0][0] == node).map(
                lambda entry: entry[1]).sum()
        return (node, credit)
    
    def process_nodes(node_w_credit):
        """ Rule 3: A DAG edge e entering node Z from the level above is given a
        share of the credit of Z proportional to the fraction of shortest
        paths from the root to Z that go through e.
        """
        node, credit = node_w_credit
        parents = db.from_sequence(all_child_parent_rels[node])
        total_num_shortest_paths_to_parents = parents.map(
            lambda parent: num_shortest_paths[parent])
        node_edge_scores = parents.map(
            lambda parent: ((parent, node), 
                            credit * num_shortest_paths[parent] / 
                            total_num_shortest_paths_to_parents))
        return node_edge_scores
        
    while to_score.count().compute() > 0:
        nodes_w_credits = to_score.map(credit)
        new_edge_scores = process_nodes.map(nodes_w_credits).flatten()
        edge_scores = db.concat([edge_scores, new_edge_scores])
        to_score = nodes_w_credits.map( # to_score = parents
            lambda node_w_c: all_child_parent_rels[node_w_c[0]]).flatten()
        
    return edge_scores


def get_edge_scores(start_node, component):
    """ Return dict of Girvan Newman edge scores starting at start_node.
    df should only contain pre, post, and syn_count cols """
    child_parent_rels, state, leaves, num_shortest_paths = pbfs(start_node, component)
    edge_scores = calculate_edge_scores(start_node, child_parent_rels, 
                                        num_shortest_paths, leaves, component)
    return edge_scores


def girvan_newman(component):
    """ Set up and do the edge-score calculation phase of Girvan-Newman """
    # Map random subset of nodes to get_edge_scores.
    # For now, using sample size = quarter the number of nodes in the df.
    component_nodes = get_all_nodes(component).compute()
    random_nodes = db.from_sequence(
        random.sample(component_nodes, len(component_nodes)/4))
    
    # Bag([((pre, post), edge_score), ...])    
    all_edge_scores = random_nodes.map(get_edge_scores, component).flatten()
    
    # Sum edge scores and divide by factor
    factor = 0.5 # Used sample size = quarter # nodes in df -> factor = 0.5
    scores = all_edge_scores.foldby(
        key = lambda edge_score: edge_score[0],
        binop = lambda accum, edge_score: accum + edge_score[1],
        initial = 0,
        combine = lambda accum1, accum2: accum1 + accum2,
        combine_initial = 0
    )
    standardised_scores = scores.map(
        lambda edge_score: (edge_score[0], edge_score[1]/factor))
    return standardised_scores




### CLUSTER IDENTIFICATION - DECOMPOSITION --------------------------------

def get_upper_threshold(edge_scores):
    """ Calculate MAD-based upper threshold.
    edge_scores of form Bag([((pre, post), edge_score), ...])
    """
    global MAD_K
    pass # TODO


def chop(component, edge_scores, upper_threshold):
    """ Remove outlier edges from component. 
    Return updated adjacency bag and bag of edges removed.
    """
    pass # TODO




### CLUSTER IDENTIFICATION - IDENTIFY CLUSTERS ----------------------------

def process_component(component):
    """ Returns a bag of components and whether processing should continue.
    Bag([(component1_bag, _continue), (component2_bag, _continue), ...])
    where component1, component2, ... are components derived from component.
        
    If _continue is true, then the caller function should apply another round
    of girvan newman on the components. continue = False occurs when no more
    edges are removed from the input component (i.e., the component is a 
    cluster).
    
    Takes a bag called component representing the edges of a component.
    """
    global MIN_CLUSTER_SIZE
    edge_scores = girvan_newman(component)
    upper_score_threshold = get_upper_threshold(edge_scores)
    new_adj_bag, removed_edges = chop(component, edge_scores, upper_score_threshold)
    log_removed_edges(removed_edges)
    if get_num_nodes(new_adj_bag).compute() < MIN_CLUSTER_SIZE:
        return (None, False) # Cluster/component too small
    if removed_edges.count().compute() == 0:
        return (new_bag, False) # Cluster found! Don't continue processing.    
    

def recurse(component):
    component_bag, _continue = component
    if not _continue:
        return (component_bag, False)
    return identify_clusters(adjacency_bags = component_bag)


def identify_clusters(df=None, adjacency_bags=None):
    global MIN_CLUSTER_SIZE, CLIENT
    
    # Clean and filter
    if df: adjacency_bags = get_component_adjacency_bags(df)
    adjacency_bags = adjacency_bags.map(prune)
    adjacency_bags = adjacency_bags.filter(
        lambda adj_bag: adj_bag.count >= MIN_CLUSTER_SIZE)
    
    # Bag([(component1_bag, continue), (component2_bag, continue), ...])
    components = adjacency_bags.map(process_component).flatten()
    components = components.map(lambda component: recurse(component))
    
    # Filter and map components to include only bags
    clusters = components.filter(
        lambda component: component[0] is not None).map(
            lambda component: component[0])
    return clusters



### -----------------------------------------------------------------------

if __name__ == "__main__":
    CLIENT = Client()
