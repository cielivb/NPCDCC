# NPCDCC - Neurotransmitter Probability Characterisation of Drosophila Connectome Communities


### Data source
to retrieve the raw data files, navigate to the data directory then run the commands in terminal:\n
aria2c -x 16 -s 16 -k 1M -o proofread_connections_783.feather https://zenodo.org/records/10676866/files/proofread_connections_783.feather \n
aria2c -x 16 -s 16 -k 1M -o flywire_synapses_783.feather https://zenodo.org/records/10676866/files/flywire_synapses_783.feather \n
Using aria2c is far less painful in times of time and complexity than curl or zenodo_get or manually downloading. The ~800 MB file only ~3 mins, and the big one ~13 mins. \n
To checksum:\n
 Get-FileHash proofread_connections_783.feather -Algorithm MD5 # Should be md5:f48f972d262323a102aed49af1396b8a\n 
 Get-FileHash flywire_synapses_783.feather -Algorithm MD5 # Should be md5:f8f1b97c9d4b0ea9b4c8b287f6b99091\n
 
