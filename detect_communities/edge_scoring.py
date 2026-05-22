""" Edge scoring functions """

import pandas as pd
from dask import dataframe as ddf
from typing import TypeAlias

DDF: TypeAlias = ddf.DataFrame
PDF: TypeAlias = pd.DataFrame


def get_scores(df: DDF, start_node: int):
    """ Get and return edge scores dataframe starting at start node """
    raise NotImplementedError


def get_scores_pd(df: DDF, start_node: int):
    """ Get and return edge scores dataframe starting at start node """
    raise NotImplementedError


def aggregate_scores(df: DDF, scores_list: list) -> tuple[DDF]:
    """ Return component and aggregated/final edge scores """
    raise NotImplementedError


def aggregate_scores_pd(df: DDF, scores_list: list) -> tuple[DDF]:
    """ Return component and aggregated/final edge scores """
    raise NotImplementedError


def get_upper_threshold(edge_scores: DDF) -> ...:
    raise NotImplementedError


def chop(df: DDF, edge_scores: DDF, upper_thresh: ...) -> DDF:
    """ Return dataframe with edges that exceeded thresh removed """
    raise NotImplementedError


def chop_pd(df: DDF, edge_scores: ..., k: float) -> tuple[DDF]:
    """ Return original and processed dataframe with high edge scores removed """
    raise NotImplementedError