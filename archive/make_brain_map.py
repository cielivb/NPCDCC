""" PyVista Brain Map Creation """
import os
import numpy as np
import pandas as pd
import pyvista as pv
from dask import dataframe as ddf


def attach_colour_groups(tagged: ddf.DataFrame):
    """ Attach column with colour group assignment to tagged dataframe.
    Edges are assigned to colour groups based on whether an edge is tagged, and 
    what the dominant neurotransmitter type is at that edge."""
    # Filter out edges not assigned to a community
    filtered = tagged[~tagged["community_id"].isna()]
    
    def assign_colour_group(tagged: pd.DataFrame) -> pd.Series:
        """ Return series containing colours 'g', 'B', 'Y', 'P' 
        'g' = grey, 'B' = blue, 'Y' = yellow, 'P' = pink """
        # Concatenate NT columns then get the indexes of column-wise max values
        gaba, ach, other = tagged["gaba"], tagged["ach"], tagged["other"]
        probs = pd.concat([gaba, ach, other], axis=1)
        all_na = probs.isna().all(axis=1)
        max_prob = probs.fillna(-1).idxmax(axis=1) # Won't crash on all NaN (picks arbitrary)
        
        # Assign colours
        colour = pd.Series("g", index=tagged.index) # Default to grey
        is_na = tagged["community_id"].isna()
        colour[(~is_na) & (max_prob == "gaba")] = "B"
        colour[(~is_na) & (max_prob == "ach")] = "Y"
        colour[(~is_na) & (max_prob == "other")] = "P"
        colour[all_na] = "g"  # force grey if all probs NaN        
        return colour
    
    colour_series = filtered.map_partitions(
        assign_colour_group, meta=("colour", "object"))
    tagged = tagged.assign(colour=colour_series)
    return tagged.persist()


def get_point_cloud(p: pd.DataFrame):
    """ Create a point cloud with appropriate colours based on coords in p """
    # Extract numpy coords (pyvista requires numpy)
    coords = p[["x", "y", "z"]].to_numpy()
    cloud = pv.PolyData(coords)
    # Assign colours based on colour group assignment
    colour = np.empty((len(p), 4)) # Empty RGBA array, as long as partition
    colour[p["colour"] == "g"] = [0.5, 0.5, 0.5, 0.2] # Grey semi-transparent
    colour[p["colour"] == "B"] = [0, 0, 1, 1] # Blue opaque
    colour[p["colour"] == "Y"] = [1, 1, 0, 1] # Yellow opaque
    colour[p["colour"] == "P"] = [1, 0.75, 0.8, 1] # Pink opaque
    cloud["colour"] = colour
    
    return cloud


def save(plotter, method, outdir):
    """ Save 3D interactive visualisation and above/side/front screenshots """
    plotter.export_html(os.path.join(outdir, f"{method}_interactive.html"))
    
    # Save screenshots from above, side, and front views
    plotter.view_vector((0, 0, 1)) # Above / looking down z-axis
    plotter.screenshot(os.path.join(outdir, f"{method}_above.png"))
    plotter.view_vector((1, 0, 0)) # Side / looking along x-axis
    plotter.screenshot(os.path.join(outdir, f"{method}_side.png"))
    plotter.view_vector((0, 1, 0)) # Front / looking along y-axis
    plotter.screenshot(os.path.join(outdir, f"{method}_front.png"))
    
    plotter.close()


def restrict_method(tagged, method):
    """ Change community method column name and drop other community method 
    results """
    print(tagged)
    keep_col = f"{method}_id"
    to_drop = [c for c in tagged.columns if c.endswith("_id") and c != keep_col]   
    trimmed = tagged.drop(columns = to_drop)
    trimmed = trimmed.rename(columns = {keep_col: "community_id"})
    return trimmed.persist()


def prepare_plotter(tagged: ddf.DataFrame, method):
    """ Stream data points into plotter """
    tagged = restrict_method(tagged, method)
    tagged = attach_colour_groups(tagged)
    print(tagged.head(25))
    
    # Take a sample of tagged_c - this will speed up the visualisation. 
    # Using frac = 0.05 -> ~800,000 points will be plotted for full drosophila
    # connectome (~16 million edges)
    tagged_sample = tagged.sample(frac = 0.05).persist()
    
    # Can't use map partitions here because creating an external effect
    # (adding points to plotter). Convert tagged dataframe into a list of dask
    # delayed objects instead. Each delayed object represents one partition of
    # the tagged connectome dataframe.
    plotter = pv.Plotter(off_screen=True)    
    for partition in tagged_sample.to_delayed():
        p = partition.compute()
        cloud = get_point_cloud(p)
        plotter.add_points(cloud, 
                           scalars = "colour", 
                           rgba = True,
                           render_points_as_spheres = True,
                           point_size = 3)
    return plotter
