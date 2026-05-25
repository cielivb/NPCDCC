""" Brain map visualisation script """

def drop_unassigned(connectome, id_col):
    """ Remove edges/rows where ID is NaN """
    return connectome[~connectome[id_col].isna()].persist()


def attach_colours(connectome, id_col):
    """ Attach column with colour group assignment to connectome.
    Edges are assigned to colour groups based on max neurotransmitter (NT)
    probability."""
    
    def assign_colour_group(tagged: pd.DataFrame) -> pd.Series:
        """ Return series containing colours 'g', 'B', 'Y', 'P'
        'g' = grey, 'B' = blue, 'Y' = yellow, 'P' = pink """
        # Concatenate NT columns then get the indexes of column-wise max values
        gaba, ach, other = tagged["gaba"], tagged["ach"], tagged["other"]
        probs = pd.concat([gaba, ach, other], axis=1)
        max_prob = probs.idxmax(axis=1) # Picks arbitrarily if tied
        
        # Assign colours
        colour = pd.Series("g", index=tagged.index) # Default to grey
        is_na = tagged[id_col].isna()
        colour[(~is_na) & (max_prob == "gaba")] = "B"
        colour[(~is_na) & (max_prob == "ach")] = "Y"
        colour[(~is_na) & (max_prob == "other")] = "P"
        return colour
    
    colour_series = connectome.map_partitions(
        assign_colour_group, meta=("colour", "object"))
    coloured = connectome.assign(colour=colour_series)
    return coloured.persist()
    

def get_plotter(connectome, id_col, plot_unassigned=True):
    """ Create a PyVista plotter for plotting EIOs for edges with IDs. 
    The plot shows edges with NaN IDs in either semi-transparent gray or not
    at all. Colours of classified edges is based on the highest
    neurotransmitter probability for that edge (blue = GABA, yellow =
    acetylcholien, pink = other).
    """
    data = connectome if plot_unassigned else drop_unassigned(connectome, id_col)
    sample = data.sample(frac = 0.05).persist() # Downsample for better render
    coloured = attach_colours(sample, id_col)
    
    # Can't use map_partitions because creating external effect (adding points 
    # to plotter). Convert tagged dataframe into list of delayed instead. Each 
    # delayed represents 1 partition of coloured dataframe.
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