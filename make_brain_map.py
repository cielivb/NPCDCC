""" PyVista Brain Map Creation """


def make_brain_map(tagged: ddf.DataFrame, coord_dir: str, outdir: str):
    """ Use PyVista to generate before and after brain map images. """
    global DATA_DIR
    # Load coord dataframe then left merge tagged dataframe with coord df.
    coord_raw_path = os.path.join(coord_dir, "flywire_synapses_783.parquet")
    coord_df = ddf.read_parquet(coord_raw_path)
    coord_df = coord_df.rename(columns = {"pre": "pre_pt_root_id",
                                          "post": "post_pt_root_id",
                                          "pre_pt_position_x": "pre_x",
                                          "pre_pt_position_y": "pre_y",
                                          "pre_pt_position_z": "pre_z",
                                          "post_pt_position_x": "post_x",
                                          "post_pt_position_y": "post_y",
                                          "post_pt_position_z": "post_z"})
    tagged_c = tagged.merge(coord_df, on = ["pre", "post"], how = "left")
    
    # "synapses were identified with two points, one in each neuron" (Zenodo). 
    # Take the mean of these two coordinates to use as true synapse coordinate.
    tagged_c["coord_x"] = tagged_c["pre_x"] + tagged_c["post_x"] / 2
    tagged_c["coord_y"] = tagged_c["pre_y"] + tagged_c["post_y"] / 2
    tagged_c["coord_z"] = tagged_c["pre_z"] + tagged_c["post_z"] / 2
    tagged_c = tagged_c.drop(columns = ["pre_x", "pre_y", "pre_z", 
                                        "post_x", "post_y", "post_z"])
    
    # Assign each edge to a colour group based on whether an edge is tagged, and 
    # what the dominant neurotransmitter type is at that edge.
    def assign_colour_group(tagged_c: pd.DataFrame) -> pd.Series:
        """ Return series containing colours 'g', 'B', 'Y', 'P' 
        'g' = grey, 'B' = blue, 'Y' = yellow, 'P' = pink """
        # Concatenate NT columns then get the indexes of column-wise max values
        gaba, ach, other = tagged_c["gaba"], tagged_c["ach"], tagged_c["other"]
        max_prob = pd.concat([gaba, ach, other], axis=1).idxmax(axis=1)
        
        # Assign colours
        colour = pd.Series("g") # Default to grey
        is_na = tagged_c["community_id"].isna()
        colour[(~is_na) & (max_prob == "gaba")] = "B"
        colour[(~is_na) & (max_prob == "ach")] = "Y"
        colour[(~is_na) & (max_prob == "other")] = "P"
        
        return colour
        
    tagged_c["colour"] = tagged_c.map_partitions(assign_colour_group, 
                                                 meta={"colour": "object"})
    
    # Take a sample of tagged_c - this will speed up the visualisation and still
    # get the general idea across. Using frac = 0.05 -> ~800,000 points will
    # be plotted.
    tagged_sample = tagged_c.sample(frac = 0.05)
    
        
    # Can't use map partitions here because creating an external effect
    # (adding points to plotter). Convert tagged dataframe into a list of dask
    # delayed objects instead. Each delayed object represents one partition of
    # the tagged connectome dataframe.
    plotter = pv.Plotter()    
    for partition in tagged_sample.to_delayed():
        p = partition.compute()
        
        # Extract numpy coords (pyvista requires numpy)
        coords = p[["coord_x", "coord_y", "coord_z"]].to_numpy()
        cloud = pv.PolyData(points)
        
        # Assign colours based on colour group assignment
        colours = np.empty(len(p), 4) # Empty RGBA array, as long as partition
        colours[p["colour" == "g"]] = [0.5, 0.5, 0.5, 0.2] # Grey semi-transparent
        colours[p["colour" == "B"]] = [0, 0, 1, 1] # Blue opaque
        colours[p["colour" == "Y"]] = [1, 1, 0, 1] # Yellow opaque
        colours[p["colour" == "P"]] = [1, 0.75, 0.8, 1] # Pink opaque
        cloud["colours"] = colours
        
        # Add points to the plotter
        plotter.add_points(cloud, scalars = "colours", rgba = True)
    
    # View the plot
    plotter.show()
        
    raise NotImplementedError