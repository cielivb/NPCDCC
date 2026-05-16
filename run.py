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

import argparse
import dask
import os
from dask import bag as db
from dask import dataframe as ddf
from dask.distributed import Client
from time import sleep

CLIENT = None # Assigned properly at bottom of script
MIN_CLUSTER_SIZE = 30
MAD_K = 3.5

ROOT_DIR = os.path.dirname(__file__)

dask.config.set({"dataframe.shuffle.method": "tasks"})




######################## GENERIC HELPER FUNCTIONS ##############################

def df_to_adjacency_bag(df, undirect=True):
    """ Convert dataframe into adjacency list/bag of edges with synapse counts.
    
    Return an adjacency list for the dataframe as a dask bag of the form
        db.from_sequence([(a, [(b, 3)]), (b: [(a, 3)])])
    if undirected, otherwise
        db.from_sequence([(a, []), (b, [(a, 3)])])
    where the edge/neuronal connection from b->a (or b->a and a->b in the 
    undirected version) involves 3 synapses.
    
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


def adj_bag_to_adj_df(adj_bag: db.Bag) -> ddf.DataFrame:
    """ Convert an adjacency bag to an adjacency dataframe. 
    
    An adjacency dataframe differs from the original dataframe in that it has
    two columns - one for node ID, and the other for neighbours of that node.
    
    """
    adj_df = adj_bag.to_dataframe(
        meta = {"node_id": int, "neighbours": object}).persist()
    return adj_df


def adj_bag_to_df(adj_bag: db.Bag) -> ddf.DataFrame:
    """ Convert an adjacency bag to an edge dataframe. 
    
    The returned dataframe will have three columns: pre, post, syn_count.
    
    adj_bag has the form [(pre, [(post1, syn_count), (post2, syn_count)], ...)]
    
    """
    # Expand adj_bag to bag of tuples (pre, post, syn_count)
    def expand(adj_tup):
        pre, post_list = adj_tup[0], adj_tup[1]
        expanded = list(map(lambda tup: (pre, tup[0], tup[1]), post_list))
        return expanded
    tuple_bag = adj_bag.map(expand).flatten()
    
    # Convert tuple_bag to dataframe
    df = tuple_bag.to_dataframe(
        meta = {"pre": int, "post": int, "syn_count": int}).persist()
    return df


def adj_df_to_adj_bag(adj_df: ddf.DataFrame) -> db.Bag:
    """ Convert an adjacency dataframe to an adjacency bag.
    
    An adjacency dataframe differs from the original dataframe in that it has
    two columns - one for node ID, and the other for neighbours of that node.

    """
    adj_bag = adj_df.to_bag().map(lambda tup: (tup[0], tup[1])).persist()
    return adj_bag


def get_all_nodes(df: ddf.DataFrame) -> db.Bag:
    """ Return a bag of every unique node in the dataframe """
    unique_pre = df["pre"].drop_duplicates().to_bag()
    unique_post = df["post"].drop_duplicates().to_bag()
    unique_nodes = db.concat([unique_pre, unique_post])
    return unique_nodes
    
    
def get_num_nodes(df: ddf.DataFrame) -> int:
    """ Compute the number of unique nodes present in a dataframe """
    return get_all_nodes.count().compute()


def create_state_df(component: ddf.DataFrame) -> ddf.DataFrame:
    """ The state dataframe uses node ids as indexes to track state in PBFS """
    nodes = get_all_nodes(component)
    state = nodes.to_dataframe(meta = {"node_id": int})
    state["state"] = "U"
    state = state.set_index("node_id", sort=True)
    print("Updated state")
    return state





################################## LOAD ########################################

def parse_args():
    """ Validate and store script arguments in global variables """
    raise NotImplementedError # TODO : implement


def load_connectome() -> ddf.DataFrame:
    """ Parse connectome feather file into dask dataframe """
    raise NotImplementedError # TODO: implement







############################ IDENTIFY CLUSTERS #################################


### Pre-PBFS component preparation functions ------------------------------

def get_component_adjacency_bags(df: ddf.DataFrame, undirected=True) -> db.Bag:
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
                                                        state)
            del all_child_parent_rels, leaves
            
            # Add new component
            diff_indices = np.where(state != prev_state)[0]
            component_nodes = nodes[diff_indices]
            component_adj = big_adjacency_bag.filter(
                lambda node_adjacency: node_adjacency[0] in component_nodes)
            components = components + [component_adj.persist()]
    
    return db.from_sequence(components)


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
    deg1_nodes = adjacency_bag.filter(
        lambda node_adj: len(node_adj[1]) == 1).map(
            lambda node_adj: node_adj[0]).compute()
    
    while True:
        if len(deg1_nodes) == 0:
            break
        
        # Remove edge from deg1 node to neighbour
        adjacency_bag = adjacency_bag.map(
            lambda node_adj: cut_deg1_edge(node_adj, node_adj[0] in deg1_nodes)).filter(
                lambda node_adj: len(node_adj[1]) > 0).persist()
        
        # Remove edge from neighbours to deg1 nodes
        adjacency_bag = adjacency_bag.map(
            lambda node_adj: remove_deg_1_nodes(node_adj, deg1_nodes)).persist()
    
    return adjacency_bag




### Calculating edge scores functions (includes PBFS & backtracking) ------
    
def update_state_array(state: ddf.DataFrame, nodes_to_update: ddf.DataFrame, 
                       new_status: str) -> ddf.DataFrame:
    """ Update states of nodes_to_update in state dataframe to either D or P """
    indexes_to_update = state.merge(nodes_to_update, left_index=True,
                                    right_on="node_id", how="inner")["node_id"]
    state["state"] = state["state"].mask(indexes_to_update, new_status)
    state = state.persist()
    return state


def get_component_subset(level_nodes: ddf.DataFrame, 
                         component: ddf.DataFrame) -> ddf.DataFrame:
    """ Get all edges from component involving any node in level_nodes """
    subset1 = level_nodes.merge(component, left_on="node_id", 
                                right_on="pre", how="inner")
    subset2 = level_nodes.merge(component, left_on="node_id", 
                                right_on="post", how="inner")
    subset = ddf.concat([subset1, subset2]).drop_duplicates()
    subset = subset.set_index("node_id", drop=False, sort=True)
    return subset


def update_pc_cp_dfs(level_nodes: ddf.DataFrame, comp_subset: ddf.DataFrame, 
                     pc_df: ddf.DataFrame, 
                     cp_df: ddf.DataFrame) -> tuple[ddf.DataFrame]:
    """ Update parent-child and child-parent relationships with this levels data """
    
    def process_node(node_id: int) -> tuple[ddf.DataFrame]:
        """ Create dataframes of new pc and cp relationships """      
        # Do a left join of neighbours to level_nodes then filter by nodes 
        # present only on the left side (i.e., present in neighbours but not in
        # level_nodes) to get all prospective parent and child neighbours.
        all_neighbours = comp_subset.loc[node_id]["post"].to_frame("post").persist()
        print(f"all neighbours = {all_neighbours}")
        print(f" level nodes = {level_nodes}")
        merged = all_neighbours.merge(level_nodes, left_on="post", right_on="node_id", 
                                      how="left", indicator=True)
        neighbours = merged[merged["_merge"] == "left_only"]["node_id"].persist()
        
        # Add entries in new_cp_rels for every child-parent relationship, where
        # node_id is the child
        p_neighbours = ddf.merge(left = neighbours, right = pc_df, 
                                 left_on="node_id", right_on="parent", how="inner")
        new_cp_rels = p_neighbours[p_neighbours["child"] == node_id]
        new_cp_rels = new_cp_rels[["child", "parent", "syn_count"]].persist()
        
        # Add entries in new_pc_rels for every parent-child relationship, where
        # node_id is the parent. Child neighbours are those neighbours in 
        # neighbours that are not parent neighbours.
        merged2 = ddf.merge(left = neighbours, right = new_cp_rels,
                                 left_on = "node_id", right_on = "parent",
                                 how = "left", indicator = True)
        c_neighbour_nodes = merged2[merged2["_merge"] == "left_only"]["node_id"]
        c_neighbours = ddf.merge(left = c_neighbour_nodes, right = comp_subset,
                                 left_on = "node_id", right_on = "child", how = "inner")
        new_pc_rels = c_neighbours[c_neighbours["parent"] == node_id].persist()
        return (new_pc_rels, new_cp_rels)
        
    new_dfs = level_nodes["node_id"].to_bag().map(process_node).persist()
    print(1)
    new_pc_dfs = new_dfs.map(lambda tup: tup[0]).compute() # List of new pc dask dfs
    print(2)
    new_pc_df = ddf.concat(new_pc_dfs)
    print(3)
    new_cp_dfs = new_dfs.map(lambda tup: tup[1]).compute()
    print(4)
    new_cp_df = ddf.concat(new_cp_dfs)
    print(5)
    pc_df = ddf.concat([pc_df, new_pc_df]).persist()
    print(6)
    cp_df = ddf.concat([cp_df, new_cp_df]).persist()
    print(7)
    return (pc_df, cp_df)


def update_num_sps_df(level_nodes: ddf.DataFrame, depth: int, cp_df: ddf.DataFrame, 
                      num_sps_df: ddf.DataFrame) -> ddf.DataFrame:
    """ Update num_sps_df with the number of shortest paths to each node in level_nodes. 
    num_sps_df has the columns : depth, node_id, num_sps
    """
    if depth == 0: # Root node - prefilled at start of PBFS
        return num_sps_df
    # Parent num sps will always be at depth one level above this level. Subset
    # to the parent level to speed up scan (avoids scanning all levels).
    all_parent_num_sps = num_sps_df[num_sps_df["depth"] == depth-1].persist()
    
    def get_num_sps(node_id: int) -> int:
        """ Sum the total number of shortest paths from node_id to parents. 
        If an edge a->b has three synapses, then there are three shortest paths
        from a->b. 
        """
        parent_data = cp_df[cp_df["child"] == node_id]
        parent_data = ddf.merge(left=parent_data, right=all_parent_num_sps,
                                left_on="parent", right_on="node_id", how="inner")
        parent_data["parent_num_sps"] = parent_data["num_sps"]
        parent_data["num_sps"] = parent_data["parent_num_sps"] * parent_data["syn_count"]
        num_sps = parent_data["num_sps"].shape[0].persist()
        return num_sps
    
    # Get this level's number of shortest paths
    new_num_sps = level_nodes
    new_num_sps["depth"] = depth
    new_num_sps["num_sps"] = level_nodes["node_id"].map(get_num_sps).persist()
    
    # Update and return full num_sps_df dataframe
    num_sps_df = ddf.concat([num_sps_df, new_num_sps]).persist()
    return num_sps_df


def get_children(level_nodes: ddf.DataFrame, pc_df: ddf.DataFrame) -> ddf.DataFrame:
    """ Get the children node ids of all nodes in level_nodes """
    merged = level_nodes.merge(pc_df, left_on="node_id", right_on="parent", how="inner")
    children = merged["child"].drop_duplicates().to_frame(name="node_id").persist()
    return children


def pbfs(start_node: int, component: ddf.DataFrame, state: ddf.DataFrame):
    """ Run a parallel breadth-first-search on component. 
    
    Return state dataframe, pc_df (parent-child dataframe), cp_df (child-parent
    dataframe) and num_sps_df (number of shortest paths dataframe).
    
    The parallel component of this BFS involves processing an entire level/
    frontier at a time, rather than naively iterating over every node for
    every level.

    """
    # Set-up PBFS
    depth = 0
    level_nodes = ddf.from_dict({"node_id": [start_node]}, npartitions=1).persist()
    pc_df = ddf.from_dict(
        {"parent": [], "child": [], "syn_count": []}, npartitions=1).persist()
    cp_df = ddf.from_dict(
        {"child": [], "parent": [], "syn_count": []}, npartitions=1).persist()
    num_sps_df = ddf.from_dict(
        {"depth": [0], "node_id": [start_node], "num_sps": [1]}, npartitions=1).persist()
    num_sps_df = num_sps_df.set_index("depth", drop=False).persist()

    # Run PBFS
    while True:
        # Update child-parent and parent-children relationship dataframes
        state = update_state_array(state, level_nodes, "D")
        comp_subset = get_component_subset(level_nodes, component)
        pc_df, cp_df = update_pc_cp_dfs(level_nodes, comp_subset, pc_df, cp_df)
        
        # Repartition and reindex pc_df and cp_df. Do it here because will be
        # reused in the next frontier, and I want fast lookups in the next
        # frontier as well!
        pc_df = pc_df.set_index("parent", drop=False).persist()
        cp_df = cp_df.set_index("child", drop=False).persist()
        
        # Update number of shortest paths dataframe
        num_sps_df = update_num_sps_df(level_nodes, depth, cp_df, num_sps_df)
        update_state_array(state, level_nodes, "P")
        
        # Increase depth and change current nodes to child nodes
        children = get_children(level_nodes, pc_df)        
        level_nodes = children
        if level_nodes.shape[0].compute() == 0:
            break
        depth += 1
    
    return (state, pc_df, cp_df, num_sps_df)


def get_initial_edge_scores(start_node: int, component: ddf.DataFrame, 
                            num_nodes: int) -> db.Bag:
    """ Run one PBFS then one PBFS backtrack then collate edge scores.
    Return Bag of Girvan Newman edge scores starting at start_node, of general 
    form Bag of tuples Bag([((pre, post), edge_score), ...]) """   
    state = create_state_df(component)
    state, pc_df, cp_df, num_sps_df = pbfs(start_node, component, state)
    
    # TODO - Implement PBFS backtrack to get initial edge scores
    raise NotImplementedError


def get_edge_scores(component: ddf.DataFrame) -> db.Bag:
    """ Set up and do the edge-score calculation phase of Girvan-Newman. 
    Output edge score bag should be of general form:
    Bag([((pre, post), edge_score), ...])
    """
    # Get random subset of nodes. For now, using sample size = ~1/4 the number
    # of nodes in component.
    component_nodes = get_all_nodes(component)
    num_nodes = component_nodes.count().compute()
    random_nodes = db.random.sample(component_nodes, int(num_nodes/4))
    
    # Get list of bags containing initial edge scores from each start node
    all_edge_score_bags_list = random_nodes.map(
        lambda start_node: get_initial_edge_scores(start_node, component, num_nodes)).compute()
    
    # Concatenate all initial edge scores into one bag. There should be 
    # duplicate entries for each edge. 
    all_edge_scores = db.concat(all_edge_score_bags_list)
    
    # TODO - make the below preserve edge identity (pre, post), and ensure
    # (a, b) and (b, a) scores are summed together as well
    # Sum edge scores and divide by factor
    #factor = 0.5 # Used sample size = quarter # nodes in df -> factor = 0.5
    #scores = all_edge_scores.foldby(
        #key = lambda edge_score: edge_score[0],
        #binop = lambda accum, edge_score: accum + edge_score[1],
        #initial = 0,
        #combine = lambda accum1, accum2: accum1 + accum2,
        #combine_initial = 0
    #)
    #standardised_scores = scores.map(
        #lambda edge_score: (edge_score[0], edge_score[1]/factor))
    #return standardised_scores
    raise NotImplementedError



### Chopping functions ----------------------------------------------------

def get_upper_threshold(edge_scores, k):
    """ Calculate MAD-based upper threshold.
    edge_scores of form Bag([((pre, post), edge_score), ...])
    """
    global MAD_K
    if not k:
        k = MAD_K
    scores = np.array(edge_scores.map(lambda tup: tup[1]).compute())
    median_score = np.median(scores)
    mad = np.median(np.absolute(scores - median_score))
    upper_threshold = median_score + k * mad
    return upper_threshold


def chop(component, edge_scores, upper_score_threshold):
    raise NotImplementedError # TODO




### Top-level cluster tagging/identification ------------------------------

def modified_girvan_newman(component: ddf.DataFrame) -> tuple[ddf.DataFrame|None, bool]:
    """ Remove linker edges from the component dataframe.
    
    Returns a bag of components and whether processing should continue.
    Bag([(component1_df, _continue), (component2_df, _continue), ...])
    where component1, component2, ... are components derived from component.
        
    If _continue is true, then the caller function should apply another round
    of girvan newman on the subcomponents. continue = False occurs when no more
    edges are removed from the input component (i.e., the component is a 
    cluster), in which case the associated component dataframe is a cluster, or
    when the associated component dataframe is too small, in which case None is
    returned in place of the too-small component.
    
    """
    global MIN_CLUSTER_SIZE
    edge_scores = get_edge_scores(component)
    upper_score_threshold = get_upper_threshold(edge_scores)
    new_adj_df, removed_edges = chop(component, edge_scores, upper_score_threshold)
    if get_num_nodes(new_adj_df).compute() < MIN_CLUSTER_SIZE:
        return (None, False) # Cluster/component too small
    if removed_edges.count().compute() == 0:
        return (new_adj_df, False) # Cluster found! Don't continue processing.    
    return (new_adj_df, True) # Further processing required
    

def process_raw_df(raw_df: ddf.DataFrame):
    """ Split up raw_df and create futures mapping to modified GN """
    global CLIENT, MIN_CLUSTER_SIZE
    adj_bags = get_component_adjacency_bags(raw_df)
    adj_bags = adj_bags.map(prune)
    adj_bags = adj_bags.filter(
        lambda adj_bag: adj_bag.count().compute() >= MIN_CLUSTER_SIZE)
    component_dfs = adj_bags.map(adj_bag_to_df).compute() # List of dfs
    new_futures = set()
    for df in component_dfs:
        new_futures.add(CLIENT.submit(modified_girvan_newman, df))
    return new_futures


def identify_clusters(connectome_df: ddf.DataFrame) -> list[ddf.DataFrame]:
    """ Run modified Girvan Newman repeatedly to identify clusters """
    future_set, cluster_dfs = set(), []
    process_raw_df(connectome_df) # Start processing from the top
    
    # Check for new results in future set and create new futures as required
    while len(future_list) > 0:
        to_delete = set()
        
        # Process finished futures        
        for future in future_set:
            if future.done():
                fresh_df, should_continue = future.result()
                if should_continue:
                    future_set.add(process_raw_df(fresh_df))
                else:
                    if fresh_df is not None:
                        cluster_dfs.append(fresh_df)
                to_delete.add(future)
                
        # Update future set then snooze
        future_set -= to_delete
        sleep(30)
        
    return cluster_dfs


def tag_edges(cluster_dfs, connectome_df):
    """ Return connectome_df sorted by and tagged with cluster IDs """
    raise NotImplementedError # TODO - implement







################################# ANALYSIS #####################################

def do_stats(clusters):
    """ Run a statistical analysis on the clusters and report the results """
    pass # TODO - implement


def make_graphs(clusters):
    """ Generate supporting graphs """
    pass # TODO - implement







################################### MAIN #######################################

def main():
    """ Run the full statistical analysis pipeline from loading to reporting """
    parse_args()
    connectome_df = load_connectome()
    clusters = identify_clusters(connectome_df)
    do_stats(clusters)    
    make_graphs(clusters)


if __name__ == "__main__":
    CLIENT = Client()
    main()