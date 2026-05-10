""" Cluster Detection Test Suite """

import os
import numpy as np
import pandas as pd
import unittest

from dask import dataframe as ddf
from dask.distributed import Client
from datetime import datetime
from memory_profiler import memory_usage
from pandas.testing import assert_frame_equal
from time import time

import run


TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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

def report_test_result(outfile, test, time1, max_mem, comp_time, comp_max_mem):
    global TIMESTAMP
    line = f"{TIMESTAMP} - {test} - {time1} - {max_mem} - {comp_time} - {comp_max_mem}\n"
    with open(outfile, 'a') as file:
        file.write(line)


### UNIT TESTS ------------------------------------------------------------

#@unittest.skip("Passing as of 8/5/26")
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
                           time1, max_mem, None, None)


@unittest.skip("OBSOLETE - bfs_components no longer takes dataframes")
class TestPrune(unittest.TestCase):
    """ OBSOLETE - bfs_components no longer takes dataframes """    
    
    OUTFILE = os.path.join(TEST_OUTPUT_DIR, "test-prune.txt")
    
    def test_base_case(self):
        """ A graph with no degree 1 edges returns itself """
        d = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d))
        df_after = pd.DataFrame(data=d).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)
        
        # Get time to run
        start_time = time()
        result_before_compute = run.prune(df_before)
        time1 = time() - start_time
        comp_start_time = time()
        result = result_before_compute.compute()
        time2 = time() - comp_start_time
        
        # Check if test passed
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)
        assert_frame_equal(result, df_after)
        
        # Get max memory usage
        max_mem = max(memory_usage((run.prune, (df_before,))))
        comp_max_mem = max(memory_usage((lambda: result_before_compute.compute(),)))
        report_test_result(TestPrune.OUTFILE, "test_base_case", 
                           time1, max_mem, time2, comp_max_mem)
        
    def test_single_incoming_deg1_edge(self):
        d_b = {"pre": [0,1,3,3,5,0,6], "post": [1,2,2,4,4,5,0]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}        
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
        
    def test_single_outgoing_deg1_edge(self):
        d_b = {"pre": [0,1,3,3,5,0,0], "post": [1,2,2,4,4,5,6]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}        
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
        
    def test_two_deg1_edges_both_incoming(self):
        d_b = {"pre": [0,1,3,3,5,0,6,7], "post": [1,2,2,4,4,5,0,4]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
    
    def test_two_deg1_edges_both_outgoing(self):
        d_b = {"pre": [0,1,3,3,5,0,0,4], "post": [1,2,2,4,4,5,6,7]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
    
    def test_two_deg1_edges_one_in_one_out(self):
        d_b = {"pre": [0,1,3,3,5,0,6,4], "post": [1,2,2,4,4,5,0,7]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
    
    def test_two_deg1_edges_one_out_one_in(self):
        d_b = {"pre": [0,1,3,3,5,0,0,7], "post": [1,2,2,4,4,5,6,4]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
        
    def test_two_deg1_edges_both_incoming_2(self):
        d_b = {"pre": [0,1,3,3,5,0,6,7], "post": [1,2,2,4,4,5,0,0]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]} 
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
    
    def test_two_deg1_edges_both_outgoing_2(self):
        d_b = {"pre": [0,1,3,3,5,0,0,0], "post": [1,2,2,4,4,5,6,7]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
    
    def test_two_deg1_edges_one_in_one_out_2(self):
        d_b = {"pre": [0,1,3,3,5,0,0,7], "post": [1,2,2,4,4,5,6,0]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
    
    def test_two_deg1_edges_one_out_one_in_2(self):
        d_b = {"pre": [0,1,3,3,5,0,6,0], "post": [1,2,2,4,4,5,0,7]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)        
        assert_frame_equal(result, df_after)
    
    def test_combo_case(self):
        d_b = {"pre": [0,1,3,3,5,0,0,6,7,9,9,10,1,13,14,
                       15,15,17,18,18,20,4,21,22,24,25], 
               "post": [1,2,2,4,4,5,6,7,8,8,10,11,12,1,2,
                        3,16,16,4,19,19,21,22,23,23,24]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = pd.DataFrame(data=d_a).sort_values(by=["pre","post"])
        
        # Get time to run
        start_time = time()
        result_before_compute = run.prune(df_before)
        time1 = time() - start_time
        comp_start_time = time()
        result = result_before_compute.compute()
        time2 = time() - comp_start_time
        
        # Check test passed
        df_after = df_after.reset_index(drop=True)        
        result = run.prune(df_before).compute()
        result = result.sort_values(by=["pre","post"]).reset_index(drop=True)
        assert_frame_equal(result, df_after)
        
        # Get max memory usage
        max_mem = max(memory_usage((run.prune, (df_before,))))
        comp_max_mem = max(memory_usage((lambda: result_before_compute.compute(),)))
        report_test_result(TestPrune.OUTFILE, "test_combo_case", 
                           time1, max_mem, time2, comp_max_mem)        


class TestGetUpperThreshold(unittest.TestCase):
    pass

@unittest.skip("OBSOLETE - bfs_components no longer takes dataframes")
class TestBfsComponents(unittest.TestCase):
    """ OBSOLETE - bfs_components no longer takes dataframes """
    
    OUTFILE = os.path.join(TEST_OUTPUT_DIR, "test-bfs-components.txt")
    
    def test_one_component_small(self):
        d = {"pre": [0,1,2], "post": [1,2,0]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d))
        df_after = pd.DataFrame(data=d).sort_values(by=["pre","post"])
        df_after = df_after.reset_index(drop=True)
        
        # Get time to run
        start_time = time()
        result_before_compute = run.bfs_components(df_before, min_size=0)
        time1 = time() - start_time
        comp_start_time = time()
        result = result_before_compute.compute()[0].compute()
        time2 = time() - comp_start_time
        
        # Check if test passed
        result = result.sort_values(by=["pre", "post"]).reset_index(drop=True)
        assert_frame_equal(result, df_after)
        
        # Get max memory usage
        max_mem = max(memory_usage((run.bfs_components, (df_before, 0))))
        comp_max_mem = max(memory_usage((lambda: result_before_compute.compute(),)))
        report_test_result(TestBfsComponents.OUTFILE, "test_one_component_small", 
                           time1, max_mem, time2, comp_max_mem)
    
    def test_multiple_components(self):
        """ There are 5 components of varying size in the initial dataframe """
        d_b = {"pre": [0,2,3,4,5,6,8,8,10,5,11,12,13,14,15,16,17,18,
                       20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,29],
               "post": [1,3,4,2,6,7,7,9,9,10,12,13,14,15,16,17,18,19,
                        21,22,23,24,25,26,27,28,29,30,31,29,28,27,26,25,24,23,20]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        
        d_c1 = {"pre": [0], "post": [1]}
        d_c2 = {"pre": [2,3,4], "post": [3,4,2]}
        d_c3 = {"pre": [5,6,8,8,10,5], "post": [6,7,7,9,9,10]}
        d_c4 = {"pre": [11,12,13,14,15,16,17,18], 
                "post": [12,13,14,15,16,17,18,19]}
        d_c5 = {"pre": [20,21,22,23,24,25,26,27,28,29,30,31,31,31,31,31,31,31,29],
                "post": [21,22,23,24,25,26,27,28,29,30,31,29,28,27,26,25,24,23,20]}
        component_dfs = []
        for component_data in [d_c1, d_c2, d_c3, d_c4, d_c5]:
            df = pd.DataFrame(data=d_c1).sort_values(by=["pre","post"])
            df = df.reset_index(drop=True)
            component_dfs.append(df)
        
        # Get time to run
        start_time = time()
        result_before_compute = run.bfs_components(df_before, min_size=0)
        time1 = time() - start_time
        comp_start_time = time()
        result = result_before_compute.compute()
        time2 = time() - comp_start_time
        
        # Check test passed
        num_matches_by_size = {len(comp_df): 0 for comp_df in component_dfs}
        for df in result:
            df = df.sort_values(by=["pre","post"]).reset_index(drop=True)
            for component_df in component_dfs:
                if len(component_df) == len(df):
                    assert_frame_equal(df.compute(), component_df)
                    num_matches_by_size[len(df)] += 1
                    break
        for size, num in num_matches_by_size.items():
            if num != 1:
                raise Exception(f"Size {size} dataframe matched {num} times")
        
        # Get max memory usage
        max_mem = max(memory_usage((run.bfs_components, (df_before, 0))))
        comp_max_mem = max(memory_usage((lambda: result_before_compute.compute(),)))
        report_test_result(TestBfsComponents.OUTFILE, "test_combo_case", 
                           time1, max_mem, time2, comp_max_mem)               


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
        parents_bag, state, leaves = run.pbfs(start_node, adjacency_bag)
        time1 = time() - start_time
        leaves = leaves.compute()
        
        # Do assertions
        parent_adj = process_computed_adjacency_bag(parents_bag.compute())
        self.assertEqual(parent_adj, exp_parent_adj)
        self.assertEqual(18, len(state))
        self.assertEqual(12, np.sum(state == "P"))
        self.assertEqual(6, np.sum(state == "U"))
        self.assertTrue(len(leaves) == 6)
        self.assertTrue(exp_leaves == set(leaves))       
        
        # Get memory usage and report results
        max_mem = max(memory_usage((run.pbfs, (start_node, adjacency_bag))))
        report_test_result(TestPBFS.OUTFILE, "test_case_4",
                           time1, max_mem, None, None)
        del adjacency_bag
        
        
        
        


### INTEGRATION TESTS -----------------------------------------------------

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
        self.assertInstance(result, db.Bag)
        components = result.compute()
        self.assertTrue(len(components) == 2)
        self.assertInstance(components[0], db.Bag)
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
                           "test_case_2_components", time1, max_mem, None, None)


class TestGirvanNewman(unittest.TestCase):
    pass


class TestIdentifyClusters(unittest.TestCase):
    pass


class TestGetClusterData(unittest.TestCase):
    pass
