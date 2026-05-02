""" Cluster Detection Test Suite """

import unittest


### UNIT TESTS ------------------------------------------------------------

class TestPrune(unittest.TestCase):
    pass


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