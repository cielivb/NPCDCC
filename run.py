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
from queue import Queue
from threading import Lock

CLIENT = None # Assigned properly at bottom of script
MIN_CLUSTER_SIZE = 30
MAD_K = 3.5

ROOT_DIR = os.path.dirname(__file__)
TEMP_CPR_CSV = os.path.join(ROOT_DIR, "temp", "child_parent_rel.csv")



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
        
    
    def process(self, adjacency_bag, nodes, state, leaves, all_child_parent_rels, node_to_i):
        """ Check all neighbours of vertices for those that should be added
        to the next layer out_layer. Updates parents_dict, leaves, and state
        as required. Returns out_layer. 
        
        adjacency_bag is of the form:
            Bag([(a, Bag([(b, 3)])), (b: Bag([(a, 3), (c, 2)]), ...]))
        where the edge from b->a and a->b has weight 3 (i.e., 3 synapses). 
        
        """
        global CLIENT
        # Set-up layer processing
        out_layer = Layer()
        layer_nodes = self.nodes.compute()
        adjacencies = adjacency_bag.filter( # Get adjacencies for this layer's nodes
            lambda node_adjacency: node_adjacency[0] in layer_nodes).persist()
        del layer_nodes
                
        # Mark this layer's nodes as processed. The child-parent rel and leaf
        # computation sections rely on parents being marked as P.
        processed_is = adjacencies.map( # Get this layer's node's indices
            lambda node_adjacency: node_to_i[node_adjacency[0]]).compute()
        state[processed_is] = "P"
        del processed_is
        
        # Get leaf nodes (those with no child nodes)
        leaves = self.update_leaves(adjacencies, leaves, state, node_to_i)

        # Get the children nodes of this layer
        all_children = adjacencies.map(
            lambda tup: tup[1]).flatten().map(
                lambda tup: tup[0]).distinct().filter(
                    lambda node_id: state[node_to_i[node_id]] != "P").persist()
        
        # Discover child nodes and compute child-parent relationships
        if all_children.count().compute() > 0:
            out_layer.insert(all_children)
            undiscovered_is = all_children.map(
                lambda child_id: node_to_i[child_id]).compute()
            state[undiscovered_is] = "D"
            del undiscovered_is
            all_child_parent_rels = self.get_child_parent_rels(
                adjacency_bag, adjacencies, state, node_to_i, all_children, all_child_parent_rels)
        
        # Free memory
        return (out_layer, leaves, all_child_parent_rels)
        


def pbfs(start_node: int, adjacency_bag: db.Bag, state=None, nodes=None):
    """ Return parents dask bag and a set of leaves. 
    
    The numpy arrays nodes and state will be automatically computed from the
    adjacency bag if not supplied. state is not required to complete the bfs
    search, however, the caller (e.g., get_component_adjacency_bags()) may 
    wish to validate the status of each node after the search is complete.
    
    Indices in parents_bag correspond to indices in nodes (or more generally, 
    to indices in adjacency_bag, although that cannot be indexed). 
    
    parents_bag has the general form
        db.from_sequence([(c, [(a, w), (b, w)]), (d, [(c, w)])])
    where a and b are parents of c, c is the only parent of d, and w is the
    edge weight or number of synapses along that edge.

    The set of leaves contains nodes that do not have any children.
    
    The original PBFS inspiration comes from the logic behind 'Bags' and 'Pennants'
    described at https://dl.acm.org/doi/epdf/10.1145/1810479.1810534. This is
    a simplified daskified implementation of their PBFS. What they term a 'bag'
    is here a 'Layer', which represents all the nodes at some leve/depth d in the 
    BFS tree.
    
    Returns parents_bag, state, and leaves

    """
    global TEMP_CPR_CSV
    if not type(nodes) is np.array:
        nodes = np.array(adjacency_bag.map( # TODO: is there a better way?
            lambda node_adjacency: node_adjacency[0]).compute())
    leaves, n = db.from_sequence([]), len(nodes)
    if not type(state) is np.array:
        state = np.full(n, "U", dtype="<U1")
    node_to_i = {node: i for i, node in enumerate(nodes)}        
    start_node_index = node_to_i[start_node]
    state[start_node_index] = "D"
    all_child_parent_rels = db.from_sequence([(start_node, [])])
    layer_0 = Layer()
    start_node_as_bag = db.from_sequence([start_node])
    layer_0.insert(start_node_as_bag)
    current_layer = layer_0

    while not current_layer.is_empty():
        next_layer, leaves, all_child_parent_rels = current_layer.process(
            adjacency_bag, nodes, state, leaves, all_child_parent_rels, node_to_i)
        current_layer = next_layer

    return (all_child_parent_rels, state, leaves)




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


### CLUSTER IDENTIFICATION - BFS TREE -------------------------------------

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
    big_adjacency_bag = df_to_adjacency_bag(df)
    nodes = big_adjacency_bag.map(
        lambda node_adjacency: node_adjacency[0]).compute() # numpy array
    state = np.full(len(nodes), "U", "<U1") # nodes indices map to state indices
    components = None # Will be a dask bag of adjacency bags (one per component)
    
    # Iterate until all nodes are assigned to a component. Must find one
    # component at a time.
    for node_index in range(len(nodes)):
        if state[node_index] == "U":
            prev_state = state.copy()
            node = nodes[node_index]            
            state[node_index] = "D"
            
            pbfs(node, big_adjacency_bag, nodes, state)
            
            # Add new component
            diff_indices = np.where(state != prev_state)[0]
            component_nodes = nodes[diff_indices]
            component_adj = big_adjacency_bag.filter(
                lambda node_adjacency: node_adjacency[0] in component_nodes)
            if not components:
                components = [component_adj]
            else:
                components = components.append(component_adj)
    
    return db.from_sequence(components)




### CLUSTER IDENTIFICATION - GIRVAN NEWMAN --------------------------------


def get_num_shortest_paths(start_node: int, parents: dict, 
                           df: ddf.DataFrame, grouped_edges) -> dict:
    """ Label each node with number of shortest paths to it from start_node. 
    Accounts for number of synapses.
    """
    state = np.full(len(parents), "U", dtype="<U1")
    queue = Queue()
    leaves = []
    num_shortest_paths = {start_node: 1}
    
    nodes = np.fromiter(map(lambda tup: tup[0], grouped_edges), dtype=int)    
    start_node_index = np.where(nodes == start_node)[0][0]
    first_children = grouped_edges[start_node_index][1]
    for child in first_children:
        queue.put(child)
    
    while not queue.empty():
        node = queue.get()
        
        # Get number of shortest paths to node
        num_paths_to_node = 0
        for parent in parents[node]:
            num_synapses_to_parent = ...
            num_paths += num_shortest_paths[parent] * num_synapses_to_parent
        num_shortest_paths[node] = num_paths_to_node
        
        # Add node's children to queue
        node_index = np.where(nodes == node)[0][0]
        children = grouped_edges[node_index][1]
        if len(children) == 0:
            leaves.append(node)
        for child in children:
            queue.put(child)
    
    return (num_shortest_paths, leaves)


def calculate_edge_scores(start_node: int, parents: list, 
                          num_shortest_paths: dict, grouped_edges: list, leaves: list):
    """ Edge scoring rules are detailed on page 365 of MMDS Chapter 10 """
    nodes = np.fromiter(map(lambda tup: tup[0], grouped_edges), dtype=int) 
    nodes_to_score, edge_scores = Queue(), dict()
    with nodes_to_score.mutex: # with queue lock
        nodes_to_score.queue.extend(leaves) # Add all leaves at once
        
    while not nodes_to_score.empty():
        node = nodes_to_score.get()
        node_index = np.where(nodes == node)[0][0]
        
        if node in leaves:
            node_credit = 1 # Rule 1
        else:
            # Rule 2
            children = grouped_edges[node_index][1]
            child_edge_credits = 0
            for child in children:
                child_edge_credits += edge_scores[(node, child)]
            node_credit = 1 + child_edge_credits
            
        # Rule 3
        node_parents = parents[node_index]
        total_num_shortest_paths_to_parents = 0
        for parent in node_parents:
            total_num_shortest_paths_to_parents += num_shortest_paths[parent]
        for parent in node_parents:
            edge_credit = node_credit * num_shortest_paths[parent] / total_num_shortest_paths_to_parents
            edge_scores[(parent, node)] = edge_credit
            nodes_to_score.put(parent)
    
    return edge_scores


def get_edge_scores(start_node, grouped_edges, df):
    """ Return dict of Girvan Newman edge scores starting at start_node.
    df should only contain pre, post, and syn_count cols """
    parents = bfs_search(start_node, grouped_edges)
    num_shortest_paths, leaves = get_num_shortest_paths(start_node, parents, df, grouped_edges)
    edge_scores = calculate_edge_scores(start_node, parents, num_shortest_paths, grouped_edges, leaves)
    return edge_scores


def girvan_newman(grouped_edges, df):
    # Choose random subset of nodes. 
    # For now, using sample size = half the number of nodes in the df.
    nodes = db.concat([df["pre"].unique().to_bag(), 
                       df["post"].unique().to_bag]).unique().compute()
    random_nodes = db.from_sequence(random.sample(nodes, len(nodes)/2))
    
    # Map random subset of nodes to get_edge_scores()
    score_dicts = random_nodes.map(get_edge_scores, grouped_edges, df)
    # Sum edge scores and divide by a factor (e.g., 2 if using all nodes)
    def score_dict_to_tuples(score_dict):
        """ Score dict contains edge scores for each edge """
        score_tuples = []
        for edge, score in score_dict.items():
            score_tuples.append((edge, score))
        return db.from_sequence(score_tuples)
    score_tuples = score_dicts.map(score_dict_to_tuples).flatten()
    scores = score_tuples.foldby(
        key = lambda edge_score: edge_score[0],
        binop = lambda accum, edge_score: accum + edge_score[1],
        initial = 0,
        combine = lambda accum1, accum2: accum1 + accum2,
        combine_initial = 0
    )
    # Used sample size = half # nodes in df -> factor = 1
    factor = 1
    standardised_scores = scores.map(lambda edge_score: (edge_score[0], edge_score[1]/factor))
    return standardised_scores


### CLUSTER IDENTIFICATION - IDENTIFY CLUSTERS ----------------------------

def identify_clusters(df=None, adjacency_bags=None):
    global MIN_CLUSTER_SIZE, CLIENT
    # Clean and filter
    if df: adjacency_bags = get_component_adjacency_bags(df)
    adjacency_bags = adjacency_bags.map(prune)
    adjacency_bags = adjacency_bags.filter(
        lambda adj_bag: adj_bag.count >= MIN_CLUSTER_SIZE)
    if adjacency_bags.count == 0:
        return # Base case - no large enough components, don't process further
    
    # Partition adjacency bags into further components
    edge_scores = girvan_newman(adjacency_bags)
    upper_score_threshold = get_upper_threshold(gn_scores)
    new_df, num_edges_removed = chop_df(df, edge_scores, upper_score_threshold)
    if num_edges_removed == 0:
        if new_df.shape[0].compute() >= MIN_CLUSTER_SIZE:
            return df # Cluster found!
        return # No edges removed, but the cluster is too small to report.
    
    new_components = component_adjacency_bags(new_df)
    clusters = CLIENT.map(identify_clusters, adjacency_bags=new_components)
    return clusters # Caller will have to wait for result then flatten.



### -----------------------------------------------------------------------

if __name__ == "__main__":
    CLIENT = Client()
