""" Edge scoring functions """

from dask import dataframe as ddf

type DDF = ddf.DataFrame


def get_initial_edge_scores():
    raise NotImplementedError


def get_edge_scores(df: DDF) -> DDF:
    raise NotImplementedError


def get_upper_threshold(edge_scores: DDF) -> ...:
    raise NotImplementedError


def chop(df: DDF, edge_scores: DDF, upper_thresh: ...) -> DDF:
    """ Return dataframe with edges that exceeded thresh removed """
    raise NotImplementedError
    