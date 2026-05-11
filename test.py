""" Cluster Detection Test Suite """

import os
import numpy as np
import pandas as pd
import unittest

from dask import bag as db
from dask import dataframe as ddf
from dask.distributed import Client
from datetime import datetime
from memory_profiler import memory_usage
from pandas.testing import assert_frame_equal
from time import time

import run


TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
NUM_CPUS_AVAIL = os.cpu_count()


def get_six_node_cycle_dask_df():
    d = {"pre": [0,0,1,5,3,3],
         "post": [1,5,2,4,2,4],
         "syn_count": [2,4,6,7,3,8],
         "misc": [1,2,3,4,5,6]}
    df = ddf.from_pandas(pd.DataFrame(data=d))
    return df


def get_nine_node_line_dask_df():
    d = {"pre": [11,12,13,14,15,16,17,18],
         "post": [12,13,14,15,16,17,18,19],
         "syn_count": [2,3,1,3,2,1,1,2],
         "misc": [7,8,9,10,11,12,13,14]}
    df = ddf.from_pandas(pd.DataFrame(data=d))
    return df


def get_twelve_node_dask_df():
    d = {"pre": [20,21,22,23,24,25,26,27,28,29,29,30,31,31,31,31,31,31,31],
         "post": [21,22,23,24,25,26,27,28,29,20,30,31,23,29,28,27,26,25,24],
         "syn_count": [7,5,2,1,1,2,3,4,3,2,1,2,1,2,2,3,2,2,2],
         "misc": [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]}
    df = ddf.from_pandas(pd.DataFrame(data=d))
    return df


def process_computed_adjacency_bag(result):
    """ Sort by node and sort node neighbours by their nodes too """
    very_sorted_result = []
    for adjacency in sorted(result, key=lambda x: x[0]):
        # Sort the lists in the tuples
        node, neighbours = adjacency[0], adjacency[1]
        very_sorted_result.append((node, sorted(neighbours)))
    return very_sorted_result


def report_test_result(outfile, test, time1, max_mem):
    global TIMESTAMP, NUM_CPUS_AVAIL
    line = f"{TIMESTAMP} - {NUM_CPUS_AVAIL} - {test} - {time1} - {max_mem}\n"
    with open(outfile, 'a') as file:
        file.write(line)




### UNIT TESTS ------------------------------------------------------------

@unittest.skip("Passing as of 8/5/26")
class TestDfToAdjacencyBag(unittest.TestCase):
    
    OUTFILE = os.path.join(TEST_OUTPUT_DIR, "test-df-to-adjacency-bag.txt")
        
    def test_case_1_undirect(self):
        df = get_six_node_cycle_dask_df()
        expected = [(0, [(1, 2), (5, 4)]),
                    (1, [(0, 2), (2, 6)]),
                    (2, [(1, 6), (3, 3)]),
                    (3, [(2, 3), (4, 8)]),
                    (4, [(3, 8), (5, 7)]),
                    (5, [(0, 4), (4, 7)])]
        result = process_computed_adjacency_bag(run.df_to_adjacency_bag(df).compute())
        self.assertEqual(result, expected)
    
    def test_case_2_undirect(self):
        df = get_nine_node_line_dask_df()
        expected = [(11, [(12, 2)]),
                    (12, [(11, 2), (13, 3)]),
                    (13, [(12, 3), (14, 1)]),
                    (14, [(13, 1), (15, 3)]),
                    (15, [(14, 3), (16, 2)]),
                    (16, [(15, 2), (17, 1)]),
                    (17, [(16, 1), (18, 1)]),
                    (18, [(17, 1), (19, 2)]),
                    (19, [(18, 2)])]
        result = process_computed_adjacency_bag(run.df_to_adjacency_bag(df).compute())
        self.assertEqual(result, expected)        
        
    def test_case_3_undirect_2_components(self):
        """ Combines test case 2 and 1 """
        df = ddf.concat([get_six_node_cycle_dask_df(), 
                        get_nine_node_line_dask_df()], axis=0)
        expected = [(0, [(1, 2), (5, 4)]),
                    (1, [(0, 2), (2, 6)]),
                    (2, [(1, 6), (3, 3)]),
                    (3, [(2, 3), (4, 8)]),
                    (4, [(3, 8), (5, 7)]),
                    (5, [(0, 4), (4, 7)]),
                    (11, [(12, 2)]),
                    (12, [(11, 2), (13, 3)]),
                    (13, [(12, 3), (14, 1)]),
                    (14, [(13, 1), (15, 3)]),
                    (15, [(14, 3), (16, 2)]),
                    (16, [(15, 2), (17, 1)]),
                    (17, [(16, 1), (18, 1)]),
                    (18, [(17, 1), (19, 2)]),
                    (19, [(18, 2)])]
        
        # Time test while it runs
        start_time = time()
        result = process_computed_adjacency_bag(run.df_to_adjacency_bag(df).compute())
        time1 = time() - start_time
        self.assertEqual(result, expected) # Check test passes
        
        # Get max memory usage
        max_mem = max(memory_usage((run.df_to_adjacency_bag, (df,))))
        report_test_result(TestDfToAdjacencyBag.OUTFILE, 
                           "test_case_3_undirect_2_components",
                           time1, max_mem)




@unittest.skip("Passing as of 11/5/26")
class TestPrune(unittest.TestCase):
    
    OUTFILE = os.path.join(TEST_OUTPUT_DIR, "test-prune.txt")
    
    def test_base_case(self):
        """ A graph with no degree 1 edges returns itself """
        before = run.df_to_adjacency_bag(get_six_node_cycle_dask_df())
        expected = run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute()
        
        # Get time to run
        start_time = time()
        result = run.prune(before).compute()
        time1 = time() - start_time
        
        # Check if test passed
        self.assertEqual(result, expected)
        
        # Get max memory usage
        max_mem = max(memory_usage((run.prune, (before,))))
        report_test_result(TestPrune.OUTFILE, "test_base_case", 
                           time1, max_mem)
        
    def test_single_incoming_deg1_edge(self):
        d_b = {"pre": [0,1,3,3,5,0,6], "post": [1,2,2,4,4,5,0],
               "syn_count": [2,6,3,8,7,4,21], "misc": [1,2,3,4,5,6,7]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
        
    def test_single_outgoing_deg1_edge(self):
        d_b = {"pre": [0,1,3,3,5,0,0], "post": [1,2,2,4,4,5,6],
               "syn_count": [2,6,3,8,7,4,21], "misc": [1,2,3,4,5,6,7]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
        
    def test_two_deg1_edges_both_incoming(self):
        d_b = {"pre": [0,1,3,3,5,0,6,7], "post": [1,2,2,4,4,5,0,4],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
    
    def test_two_deg1_edges_both_outgoing(self):
        d_b = {"pre": [0,1,3,3,5,0,0,4], "post": [1,2,2,4,4,5,6,7],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
    
    def test_two_deg1_edges_one_in_one_out(self):
        d_b = {"pre": [0,1,3,3,5,0,6,4], "post": [1,2,2,4,4,5,0,7],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
    
    def test_two_deg1_edges_one_out_one_in(self):
        d_b = {"pre": [0,1,3,3,5,0,0,7], "post": [1,2,2,4,4,5,6,4],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
        
    def test_two_deg1_edges_both_incoming_2(self):
        d_b = {"pre": [0,1,3,3,5,0,6,7], "post": [1,2,2,4,4,5,0,0],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
    
    def test_two_deg1_edges_both_outgoing_2(self):
        d_b = {"pre": [0,1,3,3,5,0,0,0], "post": [1,2,2,4,4,5,6,7],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
    
    def test_two_deg1_edges_one_in_one_out_2(self):
        d_b = {"pre": [0,1,3,3,5,0,0,7], "post": [1,2,2,4,4,5,6,0],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
    
    def test_two_deg1_edges_one_out_one_in_2(self):
        d_b = {"pre": [0,1,3,3,5,0,6,0], "post": [1,2,2,4,4,5,0,7],
               "syn_count": [2,6,3,8,7,4,21,21], "misc": [1,2,3,4,5,6,7,8]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())
        result = process_computed_adjacency_bag(run.prune(before).compute())
        self.assertEqual(result, expected)
    
    def test_combo_case(self):
        d_b = {"pre": [0,1,3,3,5,0,0,6,7,9,9,10,1,13,14,
                       15,15,17,18,18,20,4,21,22,24,25], 
               "post": [1,2,2,4,4,5,6,7,8,8,10,11,12,1,2,
                        3,16,16,4,19,19,21,22,23,23,24],
               "syn_count": [2,6,3,8,7,4,21,38,1,69,78,254,3,1,18,
                             2,5,3,23,21,1,2,21532,2,35,10], 
               "misc": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        16,17,18,19,20,21,22,23,24,25,26]}
        before = run.df_to_adjacency_bag(ddf.from_pandas(pd.DataFrame(data=d_b)))
        expected = process_computed_adjacency_bag(
            run.df_to_adjacency_bag(get_six_node_cycle_dask_df()).compute())  

        # Get time to run
        start_time = time()
        result = process_computed_adjacency_bag(run.prune(before).compute())
        time1 = time() - start_time
        
        # Check test passed
        self.assertEqual(result, expected)
        
        # Get max memory usage
        max_mem = max(memory_usage((run.prune, (before,))))
        report_test_result(TestPrune.OUTFILE, "test_combo_case", 
                           time1, max_mem)        




#@unittest.skip("Passing as of 10/5/26")
class TestPBFS(unittest.TestCase):
    
    OUTFILE = os.path.join(TEST_OUTPUT_DIR, "test_pbfs.txt")
    
    def setUp(self):
        run.CLIENT = Client()
        
    def tearDown(self):
        run.CLIENT.close()
    
    def test_case_1(self):
        df = get_six_node_cycle_dask_df()
        adjacency_bag = run.df_to_adjacency_bag(df).persist()
        start_node = 0
        exp_leaves = {3}
        exp_state = np.full(6, "P", "<U1")
        exp_parent_adj = [(3, [(2, 3), (4, 8)]), (2, [(1, 6)]), (4, [(5, 7)]), 
                          (1, [(0, 2)]), (5, [(0, 4)]), (0, [])]
        exp_parent_adj = process_computed_adjacency_bag(exp_parent_adj)
        
        parents_bag, state, leaves = run.pbfs(start_node, adjacency_bag)
        leaves = leaves.compute()
        
        parent_adj = process_computed_adjacency_bag(parents_bag.compute())
        self.assertEqual(parent_adj, exp_parent_adj)
        assert (state == exp_state).all()
        self.assertTrue(len(leaves) == 1)
        self.assertTrue(exp_leaves == set(leaves))
        del adjacency_bag
        
    def test_case_2(self):
        df = get_nine_node_line_dask_df()
        adjacency_bag = run.df_to_adjacency_bag(df).persist()
        start_node = 12
        exp_leaves = {11, 19}
        exp_state = np.full(9, "P", "<U1")
        exp_parent_adj = [(11, [(12, 2)]), (19, [(18, 2)]), (12, []),
                          (18, [(17, 1)]), (17, [(16, 1)]), (16, [(15, 2)]),
                          (15, [(14, 3)]), (14, [(13, 1)]), (13, [(12, 3)])]
        exp_parent_adj = process_computed_adjacency_bag(exp_parent_adj)
        
        parents_bag, state, leaves = run.pbfs(start_node, adjacency_bag)
        leaves = leaves.compute()
        
        parent_adj = process_computed_adjacency_bag(parents_bag.compute())
        self.assertEqual(parent_adj, exp_parent_adj)
        assert (state == exp_state).all()
        self.assertTrue(len(leaves) == 2)
        self.assertTrue(exp_leaves == set(leaves))
        del adjacency_bag
        
    def test_case_3(self):
        df = get_twelve_node_dask_df()
        adjacency_bag = run.df_to_adjacency_bag(df).persist()
        start_node = 29
        exp_leaves = {22, 24, 25, 26, 27, 30}
        exp_state = np.full(12, "P", "<U1")
        exp_parent_adj = [(22, [(21, 5), (23, 2)]), (24, [(31, 2)]),
                          (25, [(31, 2)]), (26, [(31, 2)]), 
                          (27, [(31, 3), (28, 4)]), (30, [(29, 1)]),
                          (23, [(31, 1)]), (21, [(20, 7)]), (20, [(29, 2)]),
                          (31, [(29, 2)]), (28, [(29, 3)]), (29, [])]
        exp_parent_adj = process_computed_adjacency_bag(exp_parent_adj)
        
        parents_bag, state, leaves = run.pbfs(start_node, adjacency_bag)
        leaves = leaves.compute()
        
        parent_adj = process_computed_adjacency_bag(parents_bag.compute())
        self.assertEqual(parent_adj, exp_parent_adj)
        assert (state == exp_state).all()
        self.assertTrue(len(leaves) == 6)
        self.assertTrue(exp_leaves == set(leaves))
        del adjacency_bag
        
    def test_case_4(self):
        """ Combines case 3 and case 1 """
        df = ddf.concat([get_six_node_cycle_dask_df(),
                         get_twelve_node_dask_df()], axis=0)
        adjacency_bag = run.df_to_adjacency_bag(df).persist()
        start_node = 29
        exp_leaves = {22, 24, 25, 26, 27, 30}
        exp_parent_adj = [(22, [(21, 5), (23, 2)]), (24, [(31, 2)]),
                          (25, [(31, 2)]), (26, [(31, 2)]), 
                          (27, [(31, 3), (28, 4)]), (30, [(29, 1)]),
                          (23, [(31, 1)]), (21, [(20, 7)]), (20, [(29, 2)]),
                          (31, [(29, 2)]), (28, [(29, 3)]), (29, [])]  
        exp_parent_adj = process_computed_adjacency_bag(exp_parent_adj)
        
        # Time it
        start_time = time()
        parents_bag, state, leaves, num_sps = run.pbfs(start_node, adjacency_bag)
        time1 = time() - start_time
        leaves = leaves.compute()
        num_sps = num_sps.compute()
        
        # Do assertions
        parent_adj = process_computed_adjacency_bag(parents_bag.compute())
        print(parent_adj)
        print(exp_parent_adj)
        self.assertEqual(parent_adj, exp_parent_adj)
        self.assertEqual(18, len(state))
        self.assertEqual(12, np.sum(state == "P"))
        self.assertEqual(6, np.sum(state == "U"))
        self.assertTrue(len(leaves) == 6)
        self.assertTrue(exp_leaves == set(leaves))
        self.assertTrue((22, 74) in num_sps)
        self.assertTrue((21, 14) in num_sps)
        
        # Get memory usage and report results
        #max_mem = max(memory_usage((run.pbfs, (start_node, adjacency_bag))))
        #report_test_result(TestPBFS.OUTFILE, "test_case_4",
        #                   time1, max_mem)
        del adjacency_bag
        
        


class TestGetUpperThreshold(unittest.TestCase):
    
    def test_get_upper_threshold(self):
        data = [((None,None), 1),
                ((None,None), 1.5),
                ((None,None), 1.5),
                ((None,None), 4),
                ((None,None), 4.5),
                ((None,None), 4.5),
                ((None,None), 5),
                ((None,None), 5),
                ((None,None), 12)]
        edge_scores = db.from_sequence(data)
        result = run.get_upper_threshold(edge_scores, k=5)
        self.assertEqual(result, 7)


        


### SUB-INTEGRATION TESTS -------------------------------------------------

class TestGetComponentAdjacencyBags(unittest.TestCase):
    """ Function depends on PBFS and df_to_adjacency_bag """
    
    OUTFILE = os.path.join(TEST_OUTPUT_DIR, "test-get-component-adjacency-bags.txt")    
    
    def test_case_2_components(self):
        df = ddf.concat([get_six_node_cycle_dask_df(), 
                        get_nine_node_line_dask_df()],
                       axis=0)     
        expected1 = [(0, [(1, 2), (5, 4)]),
                    (1, [(0, 2), (2, 6)]),
                    (2, [(1, 6), (3, 3)]),
                    (3, [(2, 3), (4, 8)]),
                    (4, [(3, 8), (5, 7)]),
                    (5, [(0, 4), (4, 7)])]
        expected2 = [(11, [(12, 2)]),
                    (12, [(11, 2), (13, 3)]),
                    (13, [(12, 3), (14, 1)]),
                    (14, [(13, 1), (15, 3)]),
                    (15, [(14, 3), (16, 2)]),
                    (16, [(15, 2), (17, 1)]),
                    (17, [(16, 1), (18, 1)]),
                    (18, [(17, 1), (19, 2)]),
                    (19, [(18, 2)])]
        
        start_time = time()
        result = run.get_component_adjacency_bags(df)
        time1 = time() - start_time
        
        # Run assertions
        self.assertIsInstance(result, db.Bag)
        components = result.compute()
        self.assertTrue(len(components) == 2)
        self.assertIsInstance(components[0], db.Bag)
        c1, c2 = components[0].compute(), components[1].compute()
        c1 = process_computed_adjacency_bag(c1)
        c2 = process_computed_adjacency_bag(c2)
        self.assertNotEqual(c1, c2)
        if len(c1) == 6:
            self.assertEqual(c1, expected1)
            self.assertEqual(c2, expected2)
        elif len(c2) == 6:
            self.assertEqual(c1, expected2)
            self.assertEqual(c2, expected1)
        else:
            raise(f"Expected component lengths 6 and 9: got {len(c1)}, {len(c2)}")
        
        # Get memory usage and report results
        max_mem = max(memory_usage((run.get_component_adjacency_bags, (df,))))
        report_test_result(TestGetComponentAdjacencyBags.OUTFILE, 
                           "test_case_2_components", time1, max_mem)




class TestGirvanNewman(unittest.TestCase):
    """ girvan_newman uses random.sample to get a random subset of nodes.
    This means exact tests cannot be used. The tests instead ensure the 
    relative edge score distributions are as expected, and check the relative
    values of known bridge edges compared to other edges. Edge scores are a 
    measure of edge betweenness. 
    
    girvan_newman returns a result in the general form:
        Bag([((pre, post), edge_score), ...])
    where there is an edge from pre to post with score = edge_score.
    
    This integration test relies on run.get_upper_threshold which is tested
    elsewhere.
    
    """
    
    def test_case_1(self):
        """ 6-node cycle with no bridges - similar edge scores for each edge """
        before = run.df_to_adjacency_bag(get_six_node_cycle_dask_df())
        edge_scores = run.girvan_newman(before).compute()
        upper_thresh = run.get_upper_threshold(edge_scores, 2.5)
        scores = map(lambda tup: tup[1], edge_scores)
        exceeds_thresh = list(map(lambda score: score > upper_thresh, scores))
        self.assertFalse(any(exceeds_thresh))
      
        





### MAIN INTEGRATION TESTS ------------------------------------------------


class TestIdentifyClusters(unittest.TestCase):
    pass




class TestGetClusterData(unittest.TestCase):
    pass





### GLOBAL TESTS FOR FINAL REPORT -----------------------------------------

# Need to test with different problem sizes and different numbers of processors.