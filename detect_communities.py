""" MAD-mediated Girvan-Newman Community Detection for Connectomics 
Author: Ciel Baumann
"""

import dask
from dask import bag as db
from dask import dataframe as ddf
from dask import delayed
from dask.distributed import Client
from time import sleep

CLIENT, MINSIZE, MADK = None, None, None



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
    df = df.set_index("pre", drop=False, sort=True)
    return df


def adj_df_to_adj_bag(adj_df: ddf.DataFrame) -> db.Bag:
    """ Convert an adjacency dataframe to an adjacency bag.
    
    An adjacency dataframe differs from the original dataframe in that it has
    two columns - one for node ID, and the other for neighbours of that node.

    """
    adj_bag = adj_df.to_bag().map(lambda tup: (tup[0], tup[1])).persist()
    return adj_bag


def get_all_nodes(df: ddf.DataFrame, node_cols=["pre","post"]) -> db.Bag:
    """ Return a bag of every unique node in the dataframe """
    if len(node_cols) != 2:
        raise Exception("Must use two node columns")
    unique_pre = df[node_cols[0]].drop_duplicates().to_bag()
    unique_post = df[node_cols[1]].drop_duplicates().to_bag()
    unique_nodes = db.concat([unique_pre, unique_post])
    return unique_nodes
    
    
def get_num_nodes(df: ddf.DataFrame) -> int:
    """ Compute the number of unique nodes present in a dataframe """
    return get_all_nodes.count().compute()


def create_state_df(component: ddf.DataFrame) -> ddf.DataFrame:
    """ The state dataframe uses node ids as indexes to track state in PBFS """
    nodes = get_all_nodes(component)
    state = nodes.to_dataframe(meta = {"node_id": int}).drop_duplicates()
    state["state"] = "U"
    state = state.set_index("node_id", sort=True)
    return state


def undirect_df(df: ddf.DataFrame) -> ddf.DataFrame:
    """ Add b->a for every a->b in df, and remove duplicates """
    df_reversed = df.rename(columns={"pre":"post", "post":"pre"}).persist()
    undirected = ddf.concat([df, df_reversed]).drop_duplicates().persist()
    undirected = undirected.set_index("pre", drop=False, sort=True)
    return undirected










############################ IDENTIFY CLUSTERS #################################


### Pre-PBFS component preparation functions ------------------------------

def get_components(df: ddf.DataFrame) -> list[ddf.DataFrame]:
    """ Return a list of component dataframes.
   
    For each component in the graph represented in df, return a dataframe 
    containing data for each edge of that component. This is done by performing
    iteratively performing parallel BFS to identify nodes belonging to different
    components. Only one component can be discovered at a time.

    """
    undirected_df = undirect_df(df)
    state = create_state_df(df)
    components = []
    
    while not (state["state"] == "P").all().compute():
        
        # Get nodes present in next component
        start_node = state[state["state"] == "U"]["state"].index.min().compute()
        results = pbfs(start_node, undirected_df, state, full=False)
        state, pc_df = results[0], results[1]
        component_nodes = get_all_nodes(
            pc_df, node_cols=["parent","child"]).to_dataframe(meta={"node":int})
        
        # Create and append dataframe from component nodes
        merged1 = component_nodes.merge(df, left_on="node", right_index=True, 
                                        how="inner").persist()
        merged2 = component_nodes.merge(df, left_on="node", right_on="post", 
                                        how="inner").persist()
        component_df = ddf.concat([merged1, merged2]).drop_duplicates().persist()
        components.append(component_df)

    return components


@delayed
def prune(df: ddf.DataFrame) -> ddf.DataFrame:
    """ Iteratively remove degree 1 edges from a component dataframe 
    
    A degree 1 edge is defined here as an edge associated with at least one
    degree 1 node, where a degree 1 node is a node connected by any number of 
    edges to one and only one other node. No nodes in the dataset will have 
    edges to themselves. Synapse count and directionality are not considered.
            
    The naive approach would be to take away one edge at a time. This parallel
    version improves performance by finding all degree 1 edges initially, and
    pruning each 'chain' in parallel. The time to complete is the time it takes
    to process the longest 'chain'.

    """
    def get_degree_1_nodes(df):
        node_degrees = df.groupby(df.index)["post"].nunique().to_frame("degree")
        deg1_nodes = node_degrees[
            node_degrees["degree"] == 1].index.to_frame("node")
        return deg1_nodes
    
    pruned, deg1_nodes = df, get_degree_1_nodes(df)
    while deg1_nodes.count().compute() > 0:
        # Get edges where any involved node is in deg1_nodes
        merged1 = pruned.merge(deg1_nodes, left_index=True, 
                               right_on="node", how="inner").persist()
        merged2 = pruned.merge(deg1_nodes, left_on="post", 
                               right_on="node", how="inner").persist()
        to_prune = ddf.concat([merged1, merged2]).persist()
        
        # Remove edges and recompute degree 1 nodes
        merged = pruned.merge(to_prune, on=["pre","post"], how="left", indicator=True)
        pruned = merged[merged["_merge"] == "left_only"].persist()
        deg1_nodes = get_degree_1_nodes(pruned).persist()
        
    return pruned





### Calculating edge scores functions (includes PBFS & backtracking) ------
    
def update_state_df(state: ddf.DataFrame, nodes_to_update: ddf.DataFrame, 
                       new_status: str) -> ddf.DataFrame:
    """ Update states of nodes_to_update in state dataframe to either D or P """
    merged = state.merge(nodes_to_update, left_index=True, right_on="node_id", 
                         how="left", indicator=True)
    merged["update"] = merged["_merge"] == "both"
    merged = merged.set_index("node_id", drop=False)    
    state["state"] = state["state"].mask(merged["update"], new_status)
    state = state.persist()
    return state


def get_component_subset(level_nodes: ddf.DataFrame, 
                         component: ddf.DataFrame) -> ddf.DataFrame:
    """ Get all edges from component involving any node in level_nodes """
    # Select columns in component pre that match level_nodes node_id. 
    subset_full = level_nodes.merge(component, left_on="node_id", 
                                    right_index=True, how="inner")
    subset = subset_full[["pre", "post", "syn_count"]]
    subset = subset.set_index("pre", drop=False)
    subset = subset.persist()
    return subset


def update_pc_cp_dfs(level_nodes: ddf.DataFrame, comp_subset: ddf.DataFrame, 
                     pc_df: ddf.DataFrame, cp_df: ddf.DataFrame,
                     state: ddf.DataFrame, full: bool) -> tuple[ddf.DataFrame]:
    """ Update parent-child and child-parent relationships with this levels data """
    edges = level_nodes.merge(comp_subset, left_on="node_id", 
                              right_index=True, how="inner")
    
    # Get neighbour states
    edges = edges.merge(state, left_on="post", right_index=True, how="inner")
    
    # Add entries to new_pc_df for every parent-child relationship where
    # node_id is the parent (neighbour/child state is U)
    new_pc_df = edges[edges["state"] == "U"][["pre", "post", "syn_count"]]
    new_pc_df = new_pc_df.rename(columns={"pre": "parent", "post": "child"}).persist()
    pc_df = ddf.concat([pc_df, new_pc_df]).persist()
    
    if full:
        # Add entries to new_cp_rels for every child-parent relationship where
        # node_id is the child (neighbour/parent state is P)
        new_cp_df = edges[edges["state"] == "P"][["pre", "post", "syn_count"]]
        new_cp_df = new_cp_df.rename(columns={"pre": "child", "post": "parent"}).persist()
        cp_df = ddf.concat([cp_df, new_cp_df]).persist()
        
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
    
    # Get number of shortest paths to each node on this level
    cp_rels = level_nodes.merge(cp_df, left_on="node_id", 
                                right_index=True, how="inner")
    new_num_sps = cp_rels.merge(all_parent_num_sps, left_on="parent", 
                                right_on="node_id", how="inner")
    new_num_sps["child_num_sps"] = new_num_sps["num_sps"] * new_num_sps["syn_count"]    
    new_num_sps = new_num_sps[["node_id_x", "child_num_sps"]]
    new_num_sps = new_num_sps.rename(
        columns={"node_id_x": "node_id", "child_num_sps": "num_sps"})
    new_num_sps = new_num_sps.groupby("node_id")["num_sps"].sum().reset_index()
    new_num_sps["depth"] = depth
    new_num_sps = new_num_sps.persist()
    
    # Update and return original number of shortest paths dataframe
    num_sps_df = ddf.concat([num_sps_df, new_num_sps]).persist()
    return num_sps_df


def get_children(level_nodes: ddf.DataFrame, pc_df: ddf.DataFrame) -> ddf.DataFrame:
    """ Get the children node ids of all nodes in level_nodes """
    merged = level_nodes.merge(pc_df, left_on="node_id", right_index=True, how="inner")
    children = merged["child"].drop_duplicates().to_frame(name="node_id").persist()
    return children


def pbfs(start_node: int, component: ddf.DataFrame, state: ddf.DataFrame, full=True):
    """ Run a parallel breadth-first-search on component. 
    
    Return state dataframe, pc_df (parent-child dataframe), cp_df (child-parent
    dataframe) and num_sps_df (number of shortest paths dataframe).
    
    The parallel component of this BFS involves processing an entire level/
    frontier at a time, rather than naively iterating over every node for
    every level.

    """
    # Set-up PBFS
    depth = 0
    level_nodes = ddf.from_dict({"node_id": [start_node]}, 
                                dtype=int, npartitions=1).persist()
    pc_df = ddf.from_dict(
        {"parent": pd.Series(dtype=np.int64), "child": pd.Series(dtype=np.int64),
         "syn_count": pd.Series(dtype=np.int64)}, npartitions=1).persist()
    pc_df = pc_df.set_index("parent", drop=False).persist()
    if full:
        cp_df = ddf.from_dict(
            {"child": pd.Series(dtype=np.int64), "parent": pd.Series(dtype=np.int64),
             "syn_count": pd.Series(dtype=np.int64)}, npartitions=1).persist()    
        num_sps_df = ddf.from_dict(
            {"depth": [0], "node_id": [start_node], "num_sps": [1]}, 
            npartitions=1).persist()
        cp_df = cp_df.set_index("child", drop=False).persist()
        num_sps_df = num_sps_df.set_index("depth", drop=False).persist()
    else:
        cp_df, num_sps_df = None, None

    # Run PBFS
    while True:
        # Update child-parent and parent-children relationship dataframes
        state = update_state_df(state, level_nodes, "D")
        comp_subset = get_component_subset(level_nodes, component)
        pc_df, cp_df = update_pc_cp_dfs(level_nodes, comp_subset, pc_df, cp_df, state, full)
        
        # Repartition and reindex pc_df and cp_df. Do it here because will be
        # reused in the next frontier, and I want fast lookups in the next
        # frontier as well!
        pc_df = pc_df.set_index("parent", drop=False).persist()
        if full:
            cp_df = cp_df.set_index("child", drop=False).persist()
        
        # Update number of shortest paths dataframe
        if full:
            num_sps_df = update_num_sps_df(level_nodes, depth, cp_df, num_sps_df)
        update_state_df(state, level_nodes, "P")
        
        # Increase depth and change current nodes to child nodes
        children = get_children(level_nodes, pc_df)        
        level_nodes = children
        if level_nodes.shape[0].compute() == 0:
            break
        depth += 1

    return (state, pc_df, cp_df, num_sps_df)


def assign_node_credits(depth: int, level_num_sps: ddf.DataFrame, 
                        edge_scores_df: ddf.DataFrame) -> ddf.DataFrame:
    """ Apply Rules 1 & 2 of MMDS Ch10 pp. 365.
    Each node gets a credit equal to 1 plus the sum of the scores of the 
    DAG edges from that node to the level below. Lead nodes thus get credit
    of 1. Return a df with columns node_id, credit."""    
    # Get edge scores relevant only to this level's nodes
    edge_scores = edge_scores_df.loc[depth+1]
    
    # Get parent-child relationships where this level's nodes are the parents
    pc_rels = level_num_sps.merge(edge_scores, left_on="node_id", 
                                  right_on="parent", how="inner")
    # Get child contributions to parent (this level's) node credits
    child_contribs = pc_rels.groupby("node_id")["score"].sum().to_frame() # index=node_id
    
    # Assign node credit
    node_credits = level_num_sps.merge(child_contribs, left_on="node_id", 
                                        right_index=True, how="left")
    node_credits["score"] = node_credits["score"].fillna(0)
    node_credits["credit"] = 1 + node_credits["score"]
    node_credits = node_credits[["node_id", "credit"]]
    return node_credits


def assign_edge_scores(depth: int, node_credits: ddf.DataFrame, 
                       cp_df: ddf.DataFrame, 
                       parent_num_sps: ddf.DataFrame) -> ddf.DataFrame:
    """ Apply Rule 3 of MMDS Ch10 pp. 365.
    'A DAG edge e entering node Z from the level above is given a share of the
    credit of Z proportional to the fraction of shortest paths from the root to
    Z that go through e'
    Return a dataframe with columns depth, parent, child, score.
    """
    # Get child-parent relationships where this level's nodes are the children
    cp_rels = node_credits.merge(cp_df, left_on="node_id", 
                                 right_index=True, how="inner")
    
    # Get number of shortest paths for each parent on this level
    sps = cp_rels.merge(parent_num_sps, left_on="parent", 
                        right_on="node_id", how="inner")
    sps = sps[["node_id_x", "credit", "parent", "num_sps", "syn_count"]]
    sps = sps.rename(columns={"node_id_x":"node_id"})
    
    # Assign edge credits. Each node->parent edge is allocated a proportion of 
    # the node's credit such that stronger connections (more synapses) receive
    # less credit. More credit increases the likelihood of a higher edge
    # betweenness score later.
    sps["weight"] = 1/(sps["syn_count"] * sps["num_sps"])    
    total_weights = sps.groupby("node_id")["weight"].sum().to_frame() # Index is node_id
    edge_scores = sps.merge(total_weights, left_on="node_id", 
                            right_index=True, how="inner")
    edge_scores["prop"] = edge_scores["weight_x"] / edge_scores["weight_y"]
    edge_scores["score"] = edge_scores["credit"] * edge_scores["prop"]
    
    # Reformat columns then return
    edge_scores = edge_scores[["parent", "node_id", "score"]]
    edge_scores["depth"] = depth
    edge_scores = edge_scores.rename(columns = {"node_id": "child"})
    edge_scores = edge_scores.persist()
    return edge_scores


def pbfs_backtrack(pc_df: ddf.DataFrame, cp_df: ddf.DataFrame, 
                   num_sps: ddf.DataFrame) -> ddf.DataFrame:
    """ Iterate from the bottom of the PBFS tree upwards, assigning node credits
    and edge scores along the way. Return edge scores. """
    # Set-up PBFS backtrack
    depth = num_sps["depth"].max().compute()
    edge_score_df = ddf.from_dict(
        {"depth": pd.Series(dtype=np.int64), "parent": pd.Series(dtype=np.int64),
         "child": pd.Series(dtype=np.int64), "score": pd.Series(dtype=np.float64)}, 
        npartitions=1).persist()
    edge_score_df = edge_score_df.set_index("depth", drop=False, sort=True)    
    
    # Run PBFS backtrack, accumulating edge scores
    while depth >= 0:
        level_num_sps = num_sps[num_sps["depth"] == depth].persist()
        parent_num_sps = num_sps[num_sps["depth"] == depth-1]
        node_credits = assign_node_credits(depth, level_num_sps, edge_score_df)
        edge_scores = assign_edge_scores(depth, node_credits, cp_df, parent_num_sps)
        edge_score_df = ddf.concat([edge_score_df, edge_scores])
        edge_score_df = edge_score_df.set_index("depth", drop=False, sort=True)
        depth -= 1
    
    # Reformat edge_score_df such that the smaller node number is always node1
    # and the larger node number is node2. This makes grouping easier later.
    def reformat(df):
        df["node1"] = df[["parent", "child"]].min(axis=1)
        df["node2"] = df[["parent", "child"]].max(axis=1)
        return df[["node1", "node2", "score"]]    
    edge_score_df = edge_score_df.map_partitions(reformat).persist()
    return edge_score_df


def get_initial_edge_scores(start_node: int, component: ddf.DataFrame, 
                            num_nodes: int) -> db.Bag:
    """ Run one PBFS then one PBFS backtrack then collate edge scores.
    Return Bag of Girvan Newman edge scores starting at start_node, of general 
    form Bag of tuples Bag([((pre, post), edge_score), ...]) """   
    state = create_state_df(component)
    state, pc_df, cp_df, num_sps = pbfs(start_node, component, state)
    init_edge_scores = pbfs_backtrack(pc_df, cp_df, num_sps)
    return init_edge_scores


def get_edge_scores(component: ddf.DataFrame) -> ddf.DataFrame:
    """ Set up and do the edge-score calculation phase of Girvan-Newman. 
    Output edge score df columns node1, node2, score
    """
    # Get random subset of nodes. For now, using sample size = ~1/4 the number
    # of nodes in component.
    mult_factor = 0.25
    component_nodes = get_all_nodes(component)
    num_nodes = component_nodes.count().compute()
    random_nodes = db.random.sample(component_nodes, int(mult_factor*num_nodes))
    
    # Get delayed dataframe partitions of edge scores (can mush them up because 
    # will all be concatenated anyways)
    delayed_df_partitions = random_nodes.map(
        lambda start_node: get_initial_edge_scores(start_node).to_delayed())
    delayed_df_partitions = delayed_df_partitions.flatten().to_delayed()
    
    # Concatenate sub-dataframes into one big dataframe of edge scores:
    # Group by (node1, node2) and sum edge scores then divide by appropriate
    # factor (for now, factor = 0.5 because sample size = 0.25*num nodes in 
    # component). Edge direction is already normalised so don't need to worry
    # about accounting for node2->node1 edges.    
    div_factor = 0.5    
    sub_dfs = [ddf.from_delayed(_) for _ in delayed_df_partitions]
    ungrouped = ddf.concat(sub_dfs)
    edge_scores = ungrouped.groupby(["node1","node2"])["score"].sum()
    edge_scores["score"] = edge_scores["score"] / div_factor
    edge_scores = edge_scores.persist()    
    
    # Account for edges not included in any shortest paths:
    # Collapse component dataframe, then left merge on edge_scores. Subset
    # merged result by those edges that did not merge and assign score = 0 to
    # them, then concatenate zero-score edges to edge_scores.
    def reformat(df):
        df["node1"] = df[["parent", "child"]].min(axis=1)
        df["node2"] = df[["parent", "child"]].max(axis=1)    
        return df[["node1","node2"]]
    collapsed = component.map_partitions(reformat)
    merged = collapsed.merge(edge_scores, on=["node1","node2"], 
                             how="left", indicator=True)
    no_scores = merged[merged["_merged"] == "left_only"][["node1","node2"]]
    no_scores["score"] = 0
    no_scores = no_scores.persist()
    edge_scores = ddf.concat([edge_scores, no_scores]).persist()

    return edge_scores



### Chopping functions ----------------------------------------------------

def get_upper_threshold(edge_scores: ddf.DataFrame) -> float:
    """ Calculate MAD-based upper threshold.
    edge_scores is dataframe with columns node1, node2, score
    """
    global ARGS
    scores = edge_scores["scores"].to_dask_array()
    median_score = scores.median().compute()
    
    def get_abs_dev(arr: np.ndarray):
        return np.absolute(arr - median_score)
    MAD = scores.map_blocks(get_abs_dev).median().compute()
    upper_threshold = median_score + ARGS.k * MAD
    return upper_threshold


def chop(component: ddf.DataFrame, edge_scores: ddf.DataFrame, 
         upper_score_threshold: float) -> tuple:
    """ Remove edges from component dataframe where score exceeds threshold """
    # Get edges to chop: expand to_chop to include (b->a) for each (a->b), and
    # rename columns for easy merging.    
    to_chop = edge_scores[
        edge_scores["score"] >= upper_score_threshold].persist()
    to_chop_reversed = to_chop.rename(
        columns={"node1":"node2", "node2":"node1"}).persist()
    to_chop = ddf.concat(
        [to_chop, to_chop_reversed])[["node1", "node2"]].persist()
    to_chop = to_chop.rename(columns={"node1":"pre", "node2":"post"})
    
    # Make copy of component with edges to_chop removed.
    size_before = component.shape[0].compute()
    merged = component.merge(to_chop, on=["pre","post"], 
                             how="left", indicator=True)
    new_df = merged[merged["_merge"] == "left_only"].persist()
    size_after = new_df.shape[0].compute()
    num_chopped = size_before - size_after
    return (new_df, num_chopped)



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
    global ARGS
    if get_num_nodes(new_df).compute() < ARGS.minsize:
        return (None, False) # Cluster/component too small - no point processing
    
    # Top-level modified Girvan-Newman
    edge_scores = get_edge_scores(component)
    upper_score_threshold = get_upper_threshold(edge_scores)
    new_df, num_chopped = chop(component, edge_scores, upper_score_threshold)
    
    if num_chopped == 0:
        return (new_df, False) # Cluster found! Don't continue processing.
    return (new_df, True) # Further processing required
    

def process_raw_df(raw_df: ddf.DataFrame):
    """ Split up raw_df and create futures mapping to modified GN """
    global CLIENT
    component_dfs_list = get_components(raw_df)
    pruned = [prune(df) for df in component_dfs_list]
    new_futures = set()
    for df in pruned:
        new_futures.add(CLIENT.submit(modified_girvan_newman, df))
    return new_futures


def run(connectome_df: ddf.DataFrame, minsize=30, k=2.5) -> ddf.DataFrame:
    """ Run modified Girvan Newman repeatedly to identify clusters """
    global CLIENT, MINSIZE, MADK
    CLIENT = get_client()
    MINSIZE, MADK = minsize, k
    future_set, cluster_dfs = set(), []
    
    process_raw_df(connectome_df) # Start processing from the top
    
    # Check for new results in future set and create new futures as required
    while len(future_set) > 0:
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
        sleep(10)
    
    tagged_connectome = tag_edges(cluster_dfs, connectome_df)
    return tagged_connectome


def tag_edges(cluster_dfs: list[ddf.DataFrame], 
              connectome_df: ddf.DataFrame) -> ddf.DataFrame:
    """ Return connectome_df sorted by and tagged with community IDs. 
    Community IDs are simple integers. Allocate each cluster dataframe in the
    list a community ID, concatenate the cluster dataframes, then left merge
    connectome_df onto cluster_dfs.
    """
    if len(cluster_dfs) == 0:
        return None
    cluster_ids = range(1, len(cluster_dfs)+1)
    for cluster_df, cluster_id in zip(cluster_dfs, cluster_ids):
        cluster_df["community_id"] = cluster_id
        cluster_df = cluster_df.persist()
    
    big_cluster_df = ddf.concat(cluster_dfs).persist()
    
    tagged = connectome_df.merge(big_cluster_df, on=["pre","post"], 
                                 how="left").persist()
    return tagged