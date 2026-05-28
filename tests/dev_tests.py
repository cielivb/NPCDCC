""" Cluster Detection Test Suite """

import dask
import numpy as np
import pandas as pd
import unittest
from dask import bag as db
from dask import dataframe as ddf
from dask.distributed import Client
from math import isclose
from pandas.testing import assert_frame_equal

import run


### Functions to create test dataframes -----------------------------------

def get_six_node_cycle_dask_df():
    d = {"pre": [0,0,1,5,3,3],
         "post": [1,5,2,4,2,4],
         "syn_count": [2,4,6,7,3,8],
         "misc": [1,2,3,4,5,6]}
    df = ddf.from_pandas(pd.DataFrame(data=d)).set_index("pre",drop=False,sort=True).persist()
    return df


def get_twelve_node_dask_df():
    d = {"pre": [20,21,22,23,24,25,26,27,28,29,29,30,31,31,31,31,31,31,31],
         "post": [21,22,23,24,25,26,27,28,29,20,30,31,23,29,28,27,26,25,24],
         "syn_count": [7,5,2,1,1,2,3,4,3,2,1,2,1,2,2,3,2,2,2],
         "misc": [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]}
    df = ddf.from_pandas(pd.DataFrame(data=d)).persist()
    df = df.set_index("pre", drop=False, sort=True).persist()
    return df



### Cluster identification tests ------------------------------------------

class TestPBFS(unittest.TestCase):
    """ Test that parallel breadth-first search outputs state, parent-child,
    child-parent, and num-shortest-paths dataframes correctly """
    
    def setUp(self):
        run.CLIENT = Client()
        
    def tearDown(self):
        run.CLIENT.close()
    
    def get_expected_pc_df(self):
        d = {"parent": [29,29,29,29,20,28,31,31,31,31,31,21,23],
             "child": [20,30,28,31,21,27,27,26,25,24,23,22,22],
             "syn_count": [2,1,3,2,7,4,3,2,2,2,1,5,2]
            }
        df = pd.DataFrame(data=d).sort_values(by=["parent","child"]).reset_index(drop=True)
        return df
    
    def get_expected_cp_df(self):
        d = {"child": [20,30,28,31,21,27,27,26,25,24,23,22,22],
             "parent": [29,29,29,29,20,28,31,31,31,31,31,21,23],
             "syn_count": [2,1,3,2,7,4,3,2,2,2,1,5,2]
            }        
        df = pd.DataFrame(data=d).sort_values(by=["child","parent"]).reset_index(drop=True)
        return df        
    
    def get_expected_num_sps_df(self):
        d = {"depth": [0,1,1,1,1,2,2,2,2,2,2,3],
             "node_id": [29,20,30,31,28,21,27,26,25,24,23,22],
             "num_sps": [1,2,1,2,3,14,18,4,4,4,2,74]
             }
        df = pd.DataFrame(data=d).sort_values(by=["node_id"]).reset_index(drop=True)
        return df        
    
    def test_case_1(self):
        """ 1-component test with 4 levels """
        df = get_twelve_node_dask_df()
        df = run.adj_bag_to_df(run.df_to_adjacency_bag(df)).persist() # Undirect df
        state = run.create_state_df(df).persist()
        start_node = 29
        
        # Get results
        state, pc_df, cp_df, num_sps = run.pbfs(start_node, df, state)      
        state = state.compute()
        pc_df = pc_df.compute().reset_index(drop=True).sort_values(by=["parent","child"]).reset_index(drop=True)
        cp_df = cp_df.compute().reset_index(drop=True).sort_values(by=["child","parent"]).reset_index(drop=True)
        num_sps = num_sps.compute().reset_index(drop=True).sort_values(by=["node_id"]).reset_index(drop=True)
        
        self.assertEqual(12, np.sum(state == "P")) # All nodes should be processed
        self.assertEqual(0, np.sum(state != "P")) # No nodes should be unprocessed
        assert_frame_equal(pc_df, self.get_expected_pc_df())
        assert_frame_equal(cp_df, self.get_expected_cp_df())
        assert_frame_equal(num_sps, self.get_expected_num_sps_df())
        
    def test_case_2(self):
        """ 2-component test - only 1 component should be processed """
        df = ddf.concat([get_six_node_cycle_dask_df(),
                         get_twelve_node_dask_df()], axis=0)
        df = run.adj_bag_to_df(run.df_to_adjacency_bag(df)).persist() # Undirect df        
        state = run.create_state_df(df)
        start_node = 29
        
        state, pc_df, cp_df, num_sps = run.pbfs(start_node, df, state)
        state = state.compute()
        pc_df = pc_df.compute().reset_index(drop=True).sort_values(by=["parent","child"]).reset_index(drop=True)
        cp_df = cp_df.compute().reset_index(drop=True).sort_values(by=["child","parent"]).reset_index(drop=True)
        num_sps = num_sps.compute().reset_index(drop=True).sort_values(by=["node_id"]).reset_index(drop=True)
        
        self.assertEqual(12, np.sum(state == "P")) # 12 nodes should be processed
        self.assertEqual(6, np.sum(state == "U")) # 6 nodes should be undiscovered
        assert_frame_equal(pc_df, self.get_expected_pc_df())
        assert_frame_equal(cp_df, self.get_expected_cp_df())
        assert_frame_equal(num_sps, self.get_expected_num_sps_df())
        
    def test_not_full_1(self):
        """ Test that cp_df and num_sps are None when full is False """
        df = get_twelve_node_dask_df()
        df = run.adj_bag_to_df(run.df_to_adjacency_bag(df)).persist() # Undirect df
        state = run.create_state_df(df).persist()
        start_node = 29
        
        # Get results
        state, pc_df, cp_df, num_sps = run.pbfs(start_node, df, state, full=False)
        state = state.compute()
        pc_df = pc_df.compute().reset_index(drop=True).sort_values(by=["parent","child"]).reset_index(drop=True)
        
        # Run assertions
        self.assertIsNone(cp_df)
        self.assertIsNone(num_sps)
        self.assertEqual(12, np.sum(state == "P")) # 12 nodes should be processed
        assert_frame_equal(pc_df, self.get_expected_pc_df())        
        
    def test_not_full_2(self):
        """ 2-component test - 1 component should be processed, and cp_df and 
        num_sps should be be None """
        df = ddf.concat([get_six_node_cycle_dask_df(),
                         get_twelve_node_dask_df()], axis=0)
        df = run.adj_bag_to_df(run.df_to_adjacency_bag(df)).persist() # Undirect df        
        state = run.create_state_df(df)
        start_node = 29
        
        state, pc_df, cp_df, num_sps = run.pbfs(start_node, df, state, full=False)
        state = state.compute()
        pc_df = pc_df.compute().reset_index(drop=True).sort_values(by=["parent","child"]).reset_index(drop=True)
        
        self.assertIsNone(cp_df)
        self.assertIsNone(num_sps)
        self.assertEqual(12, np.sum(state == "P")) # 12 nodes should be processed
        self.assertEqual(6, np.sum(state == "U")) # 6 nodes should be undiscovered
        assert_frame_equal(pc_df, self.get_expected_pc_df())
        

class TestPBFSBackTrack(unittest.TestCase):
    """ Test that backtracking assigns correct edge scores """
    
    def setUp(self):
        run.CLIENT = Client()
    
    def tearDown(self):
        run.CLIENT.close()
        
    def get_expected_scores(self):
        d = {"node1": [20, 29, 29, 28, 27, 20, 27, 26, 25, 24, 23, 21, 22],
             "node2": [29, 30, 31, 29, 28, 21, 31, 31, 31, 31, 31, 22, 23],
             "score": [2.054055, 1, 6.612612, 1.333333, 0.333333, 1.054055, 
                       0.666667, 1, 1, 1, 1.945945, 0.054055, 0.945945]}
        df = pd.DataFrame(data=d).sort_values(
            by=["node1","node2"]).reset_index(drop=True)
        return df
    
    def test_pbfs_backtrack(self):
        """ Test that assigned edge scores closely match expected edge scores """
        # Set-up test
        df = get_twelve_node_dask_df()
        df = run.adj_bag_to_df(run.df_to_adjacency_bag(df)).persist() # Undirect df
        state = run.create_state_df(df).persist()
        start_node = 29
        state, pc_df, cp_df, num_sps = run.pbfs(start_node, df, state)
        
        # Get result
        edge_scores_dask_df = run.pbfs_backtrack(pc_df, cp_df, num_sps)
        edge_scores = edge_scores_dask_df.compute().sort_values(
            by=["node1","node2"]).reset_index(drop=True)
        
        # Get expected result
        expected = self.get_expected_scores()
        print(f"result = \n{edge_scores}")
        print(f"expected = \n{expected}")
        # Compare each row 
        for i, row in edge_scores.iterrows():
            with self.subTest(i=i):
                self.assertTrue(isclose(row["score"], 
                                        expected.loc[i]["score"],
                                        abs_tol=1e-05))
                
                

class TestIdentifyClusters(unittest.TestCase):
    
    def setUp(self):
        run.CLIENT = Client()
    
    def tearDown(self):
        run.CLIENT.close()
        
    def test_case_tiny(self):
        """ Test that no errors are thrown with tiny fake connectome """
        df = get_twelve_node_dask_df()
        tagged = run.identify_clusters(df).compute()
        print(tagged)
        self.assertTrue(len(tagged["cluster"]) == 12)
        


################################################################################
        
class TestWriteSummaryStats(unittest.TestCase):
    
    def setUp(self):
        run.CLIENT = Client()
        
    def tearDown(self):
        run.CLIENT.close()
        
    def test_aggregate_community_data(self):
        nt_list = [0.01, 0.02, 0.01, 0.02, 0.03, 0.7, 0.9, 0.98, 0.7, 0.8, 0.8, 0.8, 0.8, 0.1, 0.01]
        fake_df = pd.from_dict({"pre": [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                                "post": [2,3,2,3,2,3,1,3,1,3,1,2,1,2,1],
                                "syn_count": [1,2,3,4,5,6,7,8,9,8,7,6,5,4,3],
                                "gaba_avg": nt_list, "ach_avg": nt_list,
                                "glut_avg": nt_list, "oct_avg": nt_list,
                                "ser_avg": nt_list, "da_avg": nt_list,
                                "neuropil": ["N" for _ in range(15)],
                                "community_id": [1 for _ in range(15)]})
        
        result = run.write_community_data.aggregate(fake_df)