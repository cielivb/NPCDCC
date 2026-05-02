""" Cluster Detection Test Suite """

import unittest
from dask import dataframe as ddf

import main


### UNIT TESTS ------------------------------------------------------------

class TestPrune(unittest.TestCase):
    
    def base_case(self):
        """ A graph with no degree 1 edges returns itself """
        d = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d))
        df_after = ddf.from_pandas(pd.DataFrame(data=d))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)
        
    def single_incoming_deg1_edge(self):
        d_b = {"pre": [0,1,3,3,5,0,6], "post": [1,2,2,4,4,5,0]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}        
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)
        
    def single_outgoing_deg1_edge(self):
        d_b = {"pre": [0,1,3,3,5,0,0], "post": [1,2,2,4,4,5,6]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}        
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)
        
    def two_deg1_edges_both_incoming(self):
        d_b = {"pre": [0,1,3,3,5,0,6,7], "post": [1,2,2,4,4,5,0,4]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)        
    
    def two_deg1_edges_both_outgoing(self):
        d_b = {"pre": [0,1,3,3,5,0,0,4], "post": [1,2,2,4,4,5,6,7]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)        
    
    def two_deg1_edges_one_in_one_out(self):
        d_b = {"pre": [0,1,3,3,5,0,6,4], "post": [1,2,2,4,4,5,0,7]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)        
    
    def two_deg1_edges_one_out_one_in(self):
        d_b = {"pre": [0,1,3,3,5,0,0,7], "post": [1,2,2,4,4,5,6,4]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)
        
    def two_deg1_edges_both_incoming_2(self):
        d_b = {"pre": [0,1,3,3,5,0,6,7], "post": [1,2,2,4,4,5,0,0]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]} 
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)        
    
    def two_deg1_edges_both_outgoing_2(self):
        d_b = {"pre": [0,1,3,3,5,0,0,0], "post": [1,2,2,4,4,5,6,7]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)        
    
    def two_deg1_edges_one_in_one_out_2(self):
        d_b = {"pre": [0,1,3,3,5,0,0,7], "post": [1,2,2,4,4,5,6,0]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)        
    
    def two_deg1_edges_one_out_one_in_2(self):
        d_b = {"pre": [0,1,3,3,5,0,6,0], "post": [1,2,2,4,4,5,0,7]}
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)        
    
    def combo_case(self):
        d_b = {"pre": [0,1,3,3,5,0,0,6,7,9,9,10,1,13,14,
                       15,15,17,18,18,20,4,21,22,24,25], 
               "post": [1,2,2,4,4,5,6,7,8,8,10,11,12,1,2,
                        3,16,16,4,19,19,21,22,23,23,24]}        
        d_a = {"pre": [0,1,3,3,5,0], "post": [1,2,2,4,4,5]}
        df_before = ddf.from_pandas(pd.DataFrame(data=d_b))
        df_after = ddf.from_pandas(pd.DataFrame(data=d_a))
        result = main.prune(df_before).compute()
        self.assertEqual(df_after, result)


class TestGetUpperThreshold(unittest.TestCase):
    pass


class TestBfsComponents(unittest.TestCase):
    pass


class TestDijkstra(unittest.TestCase):
    pass


class TestGirvanNewman(unittest.TestCase):
    """ Technically an integration test (depends on dijkstra). """
    pass



### INTEGRATION TESTS -----------------------------------------------------

class TestIdentifyClusters(unittest.TestCase):
    pass


class TestGetClusterData(unittest.TestCase):
    pass