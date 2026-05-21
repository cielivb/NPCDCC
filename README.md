# NPCDCC - Neurotransmitter Probability Characterisation of Drosophila Connectome Communities


### Data source
to retrieve the raw data files, navigate to the data directory then run the commands in terminal:
aria2c -x 16 -s 16 -k 1M -o proofread_connections_783.feather https://zenodo.org/records/10676866/files/proofread_connections_783.feather
aria2c -x 16 -s 16 -k 1M -o flywire_synapses_783.feather https://zenodo.org/records/10676866/files/flywire_synapses_783.feather
Using aria2c is far less painful in times of time and complexity than curl or zenodo_get or manually downloading. The ~800 MB file only took around 5 mins, and the big one ~15-20 mins.
