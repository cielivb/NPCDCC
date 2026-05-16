""" Cluster Detection Test Suite """

import numpy as np
import pandas as pd
import unittest
from dask import bag as db
from dask import dataframe as ddf
from dask.distributed import Client
from pandas.testing import assert_frame_equal

import run

def get_six_node_cycle_dask_df():
    d = {"pre": [0,0,1,5,3,3],
         "post": [1,5,2,4,2,4],
         "syn_count": [2,4,6,7,3,8],
         "misc": [1,2,3,4,5,6]}
    df = ddf.from_pandas(pd.DataFrame(data=d)).persist()
    return df


def get_twelve_node_dask_df():
    d = {"pre": [20,21,22,23,24,25,26,27,28,29,29,30,31,31,31,31,31,31,31],
         "post": [21,22,23,24,25,26,27,28,29,20,30,31,23,29,28,27,26,25,24],
         "syn_count": [7,5,2,1,1,2,3,4,3,2,1,2,1,2,2,3,2,2,2],
         "misc": [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]}
    df = ddf.from_pandas(pd.DataFrame(data=d)).persist()
    return df


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
        df = ddf.from_pandas(pd.DataFrame(data=d))
        return df
    
    def get_expected_cp_df(self):
        d = {"child": [20,30,28,31,21,27,27,26,25,24,23,22,22],
             "parent": [29,29,29,29,20,28,31,31,31,31,31,21,23],
             "syn_count": [2,1,3,2,7,4,3,2,2,2,1,5,2]
            }        
        df = ddf.from_pandas(pd.DataFrame(data=d))
        return df        
    
    def get_expected_num_sps_df(self):
        d = {"depth": [0,1,1,1,1,2,2,2,2,2,2,3],
             "node_id": [29,20,30,31,28,21,27,26,25,24,23,22],
             "num_sps": [1,2,1,2,3,14,18,4,4,4,2,74]
             }
        df = ddf.from_pandas(pd.DataFrame(data=d))
        return df        
    
    
    def test_case_1(self):
        """ 1-component test with 4 levels """
        df = get_twelve_node_dask_df()
        df = run.adj_bag_to_df(run.df_to_adjacency_bag(df)).persist() # Undirect df
        state = run.create_state_df(df).persist()
        start_node = 29
        
        state, pc_df, cp_df, num_sps = run.pbfs(start_node, df, state)      
        state = state.compute()
        pc_df = pc_df.compute()
        cp_df = pc_df.compute()
        num_sps = num_sps.compute()
        
        self.assertEqual(12, np.sum(state == "P")) # All nodes should be processed
        self.assertEqual(0, np.sum(state != "P")) # No nodes should be unprocessed
        assert_frame_equal(pc_df, get_expected_pc_df())
        assert_frame_equal(cp_df, get_expected_cp_df())
        assert_frame_equal(num_sps, get_expected_num_sps_df())
        
        
    def test_case_2(self):
        """ 2-component test - only 1 component should be processed """
        df = ddf.concat([get_six_node_cycle_dask_df(),
                         get_twelve_node_dask_df()], axis=0)
        state = run.create_state_df(df)
        start_node = 29
        
        state, pc_df, cp_df, num_sps = run.pbfs(start_node, df, state)      
        state = state.compute()
        pc_df = pc_df.compute()
        cp_df = pc_df.compute()
        num_sps = num_sps.compute()
        
        self.assertEqual(12, np.sum(state == "P")) # 12 nodes should be processed
        self.assertEqual(6, np.sum(state == "U")) # 6 nodes should be undiscovered
        assert_frame_equal(pc_df, get_expected_pc_df())
        assert_frame_equal(cp_df, get_expected_cp_df())
        assert_frame_equal(num_sps, get_expected_num_sps_df())        