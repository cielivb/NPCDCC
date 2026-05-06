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
from threading import Lock
from threading import Thread


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




### PARALLEL BFS ----------------------------------------------------------

class Pennant():
    """"""    
    
    def __init__(self, element: int|None):
        """ Initialise a pennant holding a single element """
        global GRAIN_SIZE # grain size is the # of elements this pennant can hold
        self.left, self.right = None, None # Assign null pointers to children
        self.elements = np.full(GRAIN_SIZE, None, dtype="object")
        self.elements[0] = element
    
    
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


    def get_tree_size(self):
        """ Get the total # of elements stored in self and all descendants """
        pass


class PBag():
    """ Represents a layer d at depth d of parallel BFS search """
    
    def __init__(self, num_nodes_to_store: int):
        """ PBFS represents a bag S using a ﬁxed-size array S[0 . . r], called 
        the backbone, where 2^(r+1) exceeds the maximum number ofelements ever 
        stored in a bag. Each entry S[k] in the backbone con-tains either a 
        null pointer or a pointer to a pennant """        
        self.max_node_capacity = num_nodes_to_store
        r = log(num_nodes_to_store)/log(2) - 1 # Simple rearrangement of equality
        backbone_size = int(r) + 2 # Ensure the formula exceeds num_nodes_to_store
        self.backbone = np.full(backbone_size, None, dtype="object")
        
        # A PBag also maintains an additional pennant node called the hopper,
        # which it fills gradually.
        self.hopper = Pennant(None)
        self.hopper_capacity = self.hopper.elements.size
        
        self.bag_lock = Lock()
        
        
    def insert(self, element: int):
        """ Insert element into bag """
        new_pennant = Pennant(element)
        hopper_none_indices = np.where(self.hopper.elements == None)[0]
        with self.bag_lock:
            # Insert element into hopper if hopper is not full (most cases)
            if hopper_none_indices.size > 0:
                self.hopper.elements[hopper_none_indices[0]] = new_pennant
            else:
                # If hopper full, insert hopper into backbone then put element in new
                # hopper (occurs once for every GRAIN_SIZE insertions)            
                backbone_none_indices = np.where(self.backbone == None)[0]
                self.backbone[backbone_none_indices[0]] = self.hopper
                self.hopper = new_pennant
    
    
    def _full_adder(x, y, z):
        """ Union 3 pennants into 2 pennants. 
        Logic source: https://dl.acm.org/doi/epdf/10.1145/1810479.1810534 pg 5
        """
        x_empty = np.all(x.elements == None)
        y_empty = np.all(y.elements == None)
        z_empty = np.all(z.elements == None)
        
        match (x_empty, y_empty, z_empty):
            case (True, True, True): return (None, None)
            case (False, True, True): return (x, None)
            case (True, False, True): return (y, None)
            case (True, True, False): return (z, None)
            case (False, False, True): return (None, x.pennant_union(y))
            case (False, True, False): return (None, x.pennant_union(z))
            case (True, False, False): return (None, y.pennant_union(z))
            case (False, False, False): return (x, y.pennant_union(z))
        
        
    def union(self, other_pbag):
        """ Move all elements from other_pbag to self, and destroy other_pbag.
        Uses an algorithm similar to ripple-carry addition of two binary 
        counters """
        # Determine which bag has the less full hopper
        num_spaces_avail_self = np.sum(self.hopper.elements == None)
        num_spaces_avail_other = np.sum(other_pbag.hopper.elements == None)
        spaces_per_bag = [(self, num_spaces_avail_self), 
                          (other_pbag, num_spaces_avail_other)]
        emptier_bag = max(spaces_per_bag, key = lambda x: x[1])
        fuller_bag = min(spaces_per_bag, key = lambda x: x[1])
        emptier_hopper, fuller_hopper = emptier_bag.hopper, fuller_bag.hopper
        
        # Move as many elements of the less full hopper into the more full 
        # hopper as possible
        num_elements_in_emptier = np.sum(emptier_hopper.elements != None)
        num_spaces_avail_in_fuller = np.sum(fuller_hopper.elements == None)
        if num_elements_in_emptier > num_spaces_avail_in_fuller:
            # Cut the last items in the emptier hopper to be moved
            elements_to_move = emptier_hopper.elements[-num_spaces_avail_in_fuller:]
            emptier_hopper.elements[-num_spaces_avail_in_fuller] = None
        else:
            elements_to_move = emptier_hopper.elements
            emptier_hopper.elements = None
        max_fill = len(elements_to_move)        
        fuller_none_indices = np.where(fuller_hopper.elements == None)[0]
        fuller_hopper[fuller_none_indices[:max_fill]] = elements_to_move
        
        # Finally, the actual union step. Should at least one element remain in 
        # the emptier hopper, set y = the more full hopper, else None.
        num_elements_in_emptier_hopper = np.sum(emptier_hopper.elements != None)
        y = fuller_hopper if num_elements_in_emptier_hopper > 0 else None
        for k in range(self.hopper_capacity + 1):
            emptier_bag.backbone[k], y = self._full_adder(emptier_bag.backbone[k],
                                                          fuller_bag.backbone[k],
                                                          y)
        del fuller_bag
    
    
    def split(self):
        """ Remove half (to within some constant amount GRAIN_SIZE) of the 
        elements from self, and put them in a new bag new_bag. "operates like
        an arithmetic right shift" """
        bag2 = PBag(self.max_node_capacity)
        bag2.hopper = self.backbone[0]
        self.backbone[0] = None
        for k in range(1, self.hopper_capacity + 1):
            if self.backbone[k]:
                bag2.backbone[k-1] = self.backbone[k].pennant_split()
                self.backbone[k-1] = self.backbone[k]
                self.backbone[k] = None
        return bag2
    
    
    def is_empty(self):
        """ Return True if no elements are stored in the bag 
        TODO : Validate """
        if np.all(self.hopper == None) and np.all(self.backbone == None):
            return True
        return False
    
    
    def size(self):
        """ Return number of elements stored in the bag 
        TODO: validate """
        # Get the indices of the backbone where there are pennants
        pennant_indices = np.where(self.backbone != None)[0][0]
        # Sum the elements in the pennants.
        
        # Add on the number of elements in the hopper.
        pass # TODO


def pbfs_search(start_node: int, adjacency_bag: db.Bag, nodes=None, state=None):
    """ Return parents dask bag and a set of leaves. 
    
    The numpy arrays nodes and state will be automatically computed from the
    adjacency bag if not supplied. state is not required to complete the bfs
    search, however, the caller (e.g., get_component_adjacency_bags()) may 
    wish to validate the status of each node after the search is complete.
    state is not returned; it is modified in place.
    
    Indices in parents_bag correspond to indices in nodes (or more generally, 
    to indices in adjacency_bag, although that cannot be indexed). 
    
    parents_bag has the general form
        db.from_sequence([(c, [(a, w), (b, w)]), (d, [(c, w)])])
    where a and b are parents of c, c is the only parent of d, and w is the
    edge weight or number of synapses along that edge.

    The set of leaves contains nodes that do not have any children.
    
    Parallel algorithm inspired by:
    https://dl.acm.org/doi/epdf/10.1145/1810479.1810534
    
    """
    if not nodes:
        nodes = adjacency_bag.map(
            lambda node_adjacency: node_adjacency[0]).compute()
    parents_dict = defaultdict(set)
    leaves = set()
    n = len(nodes)
    state = np.full(n, "U", dtype)
    start_node_index = np.where(nodes == start_node)[0][0]
    state[start_node_index] = "D"
    
    state_lock, parent_lock = Lock(), Lock()
    
    layer_0 = PBag(n)
    layer_0.insert(start_node)
    current_layer = layer_0
    
    def process_node(child_node, parent_node, out_bag, out_bag_lock):
        child_node_index = np.where(nodes == child_node)[0][0]
        # Discover child node
        if state[child_node_index] == "U":
            with state_lock:
                state[child_node_index] = "D"
            out_bag.insert(child_node)
            
        # Adjacency bag has general format:
        # [(node_id, [(neighbour1_id, num_synapses), (neighbour2_id, num_synapses)]), (...), (...)]
        # Get the number of synapses between parent and child node
        num_synapses = adjacency_bag.filter(
            lambda node_adjacency: node_adjacency[0] == child_node).map(
                lambda node_adjacency: node_adjacency[1]).filter(
                    lambda tup: tup[0] == parent_node).map(lambda tup: tup[1])
        # Add parent and edge weight to child entry in parents dictionary
        with parent_lock:
            parents[child_node].add((parent_node, num_synapses))
            
    def process_layer(in_bag, out_bag):
        """ Each iteration processes the layer in_bag by checking all the 
        neighbors of vertices in in_bag for those that should be added to 
        the next layer out_bag """
        global GRAIN_SIZE
        if in_bag.size() < GRAIN_SIZE:
            for parent_node in in_bag:
                children_and_weights = adjacency_bag.filter(
                    lambda node_adjacency: node_adjacency[0] == node).map(
                        lambda node_adjacency: node_adjacency[1])
                if children_and_weights.count() == 0:
                    leaves.add(parent_node)
                    continue
                children = children_and_weights.map(lambda tup: tup[0])
                children = children.map(
                    lambda child: process_node(child, parent_node, out_bag))
            return
        new_bag = in_bag.split()
        thread = Thread(target=process_layer, args=(new_bag, out_bag))
        thread.start()
        process_layer(in_bag, out_bag)
        thread.join()

    while not current_layer.is_empty():
        next_layer = PBag(n)
        process_layer(current_layer, next_layer)
        current_layer = next_layer
    
    # Convert parents_dict values to lists, then convert parents_dict to dask bag
    parents_dict = {child: list(parents) for child, parents in parents_dict.items()}
    parents = db.from_sequence(parents_dict.items())
    return (parents, leaves)




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
