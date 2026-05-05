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
import random
from collections import defaultdict
from dask import bag as db
from dask import dataframe as ddf
from queue import Queue


GRAIN_SIZE = 128
MIN_CLUSTER_SIZE = 30
MAD_K = 3.5


### CLUSTER IDENTIFICATION - HELPER FUNCTIONS -----------------------------

def df_to_adjacency_bag(df, undirected=True):
    """ Convert dataframe edges into adjacency list/bag of edges with weights.
    
    Return an adjacency list for the dataframe as a dask bag of the form
        db.from_sequence([(a, [(b, 3)]), (b: [(a, 3)])])
    if undirected, otherwise
        db.from_sequence([(a, []), (b, [(a, 3)])])
    where the edge from b->a (or b->a and a->b in the undirected version) has
    weight 3.
    
    Weight represents the number of synpases between two neurons a and b.
    
    """
    edge_bag = df[["pre", "post", "syn_count"]].to_bag()
    if undirected: # Add (b,a,w) for every (a,b,w)
        edge_bag = edge_bag.map(lambda edge: [edge, (edge[1], edge[0], edge[2])])
        edge_bag = edge_bag.flatten().distinct()
    
    adjacency_bag = edge_bag.foldby(
        key = lambda edge: edge[0],
        binop = lambda accum, edge: accum + [(edge[1], edge[2])],
        initial = [],
        combine = lambda accum1, accum2: accum1 + accum2,
        combine_initial = []
    )
    return adjacency_bag


class Pennant():
    """"""    
    def __init__(self, element: int):
        """ Initialise a pennant holding a single element """
        global GRAIN_SIZE # grain size is the # of elements this pennant can hold
        self.left, self.right = None, None # Assign null pointers to children
        elements = np.full(GRAIN_SIZE, None, dtype="object")
        elements[0] = element
    
    def pennant_union(self, other):
        """ Combine self with other pennant.
        "Two pennants x and y of size 2^k can be combined to form a pennant of 
        size 2^(k+1) in O(1) time" """
        other.right = self.left
        self.left = other
        return self
    
    def pennant_split(self):
        """ Splits self into two pennants (i.e., inverse of pennant_union()).
        Requires self to contain at least 2 elements. Pennants self and new will 
        each contain half the elements in original self. """
        new = self.left
        self.left = new.right
        new.right = None
        return new


class PBag():
    """ A collection of pennants, no two of which have the same size. """
    
    def __init__(self, num_nodes_to_store: int):
        """ PBFS represents a bag S using a ﬁxed-size array S[0 . . r], called 
        the backbone, where 2^(r+1) exceeds the maximum number ofelements ever 
        stored in a bag. Each entry S[k] in the backbone con-tains either a 
        null pointer or a pointer to a pennant """
        global GRAIN_SIZE
        
        r = log(num_nodes_to_store)/log(2) - 1 # Simple rearrangement of equality
        backbone_size = int(r) + 2 # Ensure the formula exceeds num_nodes_to_store
        self.backbone = np.full(backbone_size, None, dtype="object")
        
        # A PBag also maintains an additional pennant node called the hopper,
        # which it fills gradually.
        hopper = np.full(GRAIN_SIZE, None, dtype="object")
        
    def insert(self, element):
        pass

def pbfs_search(start_node: int, adjacency_bag: db.Bag, nodes=None, state=None):
    """ Return parents dask bag, num_shortest_paths array, and set of leaves. 
    
    The numpy arrays nodes and state will be automatically computed from the
    adjacency bag if not supplied. state is not required to complete the bfs
    search, however, the caller (e.g., get_component_adjacency_bags()) may 
    wish to validate the status of each node after the search is complete.
    state is not returned; it is modified in place.
    
    Indices in parents_bag and num_shortest_paths array correspond to indices
    in nodes (or more generally, to indices in adjacency_bag, although that
    cannot be indexed). 
    
    parents_bag has the general form
        db.from_sequence([(c, {a, b}), (d, {c})])
    where a and b are parents of c, and c is the only parent of d.
    
    The values in num_shortest_paths are the number of shortest paths from the
    start node to that respective node. Given an edge a->b with weight = 3,
    there are 3 synaptic connections between a and b, and thus 3 paths from
    a to b. That is, the integer weight represents the number of paths from
    a to b.
    
    The set of leaves contains nodes that do not have any children.
    
    Parallel algorithm inspired by:
    https://dl.acm.org/doi/epdf/10.1145/1810479.1810534
    
    """
    # Set up
    if not nodes:
        nodes = adjacency_bag.map(
            lambda node_adjacency: node_adjacency[0]).compute()
    state = np.full(len(nodes), "U", dtype)
    
    layer_0 = bag_create(len(nodes)) # Allocate space for a fixed-size backbone of null pointers
    
    


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
            
            pbfs_search(node, big_adjacency_bag, nodes, state)
            
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


### CLUSTER IDENTIFICATION - BFS COMPONENT SEARCH -------------------------

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
            state[node_index] = "D"
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


### CLUSTER IDENTIFICATION - GIRVAN NEWMAN --------------------------------

def bfs_search(start_node, grouped_edges) -> list:
    """ Create parent array via BFS starting at start node """
    n = len(grouped_edges)
    parents = defaultdict(set) # key is node, vals are parents
    state = np.full(n, "U", dtype="<U1")
    queue = Queue()
    nodes = np.fromiter(map(lambda tup: tup[0], grouped_edges), dtype=int)
    start_node_index = np.where(nodes == start_node)[0][0]
    state[start_node_index] = "D"
    queue.put(start_node)
    
    while not queue.empty():
        node = queue.get()
        node_index = np.where(nodes == node)[0][0]
        node_neighbours = grouped_edges[node_index][1]
        for neighbour in node_neighbours:
            neighbour_index = np.where(nodes == neighbour)[0][0]
            if state[neighbour_index] in ["U", "D"]:
                state[neighbour_index] = "D"
                parents[neighbour].add(node)
                queue.put(neighbour)
        state[node_index] = "P"
    
    return parents


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

def identify_clusters(df: ddf.DataFrame, is_pruned=True):
    if not is_pruned:
        df = prune(df)
    grouped_edges = edge_df_to_tuple(df)
    gn_scores = girvan_newman(grouped_edges, df)
    upper_score_threshold = get_upper_threshold(gn_scores)
    new_df = chop_df(df, edge_scores, upper_score_threshold)
    new_components = bfs_components(new_df)
    new_clusters = db.Bag()
    for component in new_components():
        cluster = prune(component)
        new_clusters = db.concat([new_clusters, cluster])
    return new_clusters
