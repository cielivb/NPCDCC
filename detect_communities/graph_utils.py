""" Graph Utilities / Helper Functions """

from dask import dataframe as ddf

type DDF = ddf.DataFrame


def undirect_df():
    raise NotImplementedError


def get_components(df: DDF) -> list[DDF]:
    raise NotImplementedError


def prune(df: DDF) -> DDF:
    raise NotImplementedError
