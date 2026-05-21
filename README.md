# NPCDCC - Neurotransmitter Probability Characterisation of Drosophila Connectome Communities

## Result Replication - Quick start
1. Download the raw data files into the repo's data directory (see Downloading required files section below)
2. In the repo root directory, execute `python preprocess.py` - this converts the raw data to parquet files and generates additional test files. This took about ~10 minutes on my local machine.
3. TODO

## Original data source
Full raw data metadata is available at the source repo: [https://zenodo.org/records/10676866](https://zenodo.org/records/10676866)

### Files used
1. proofread_connections_783.feather - the dataset ran through the community detection pipeline
2. flywire_synapses_783.feather - supplies coordinates used for generating brain maps (IN DEV)

### Downloading required files

Disclaimer: I have only tested this on Windows 11.\\

To retrieve the raw data files, navigate to the data directory in terminal, then run the following commands to download the raw data into the data directory. On my local machine, the ~800 MB file took about 3 minutes to download, and the big one took about 13 minutes.
```
aria2c -x 16 -s 16 -k 1M -o proofread_connections_783.feather https://zenodo.org/records/10676866/files/proofread_connections_783.feather
aria2c -x 16 -s 16 -k 1M -o flywire_synapses_783.feather https://zenodo.org/records/10676866/files/flywire_synapses_783.feather
```
To checksum the downloaded raw data:
```
# proofread_connections_783.feather MD5 checksum should be f48f972d262323a102aed49af1396b8a
Get-FileHash proofread_connections_783.feather -Algorithm MD5
# flywire_synapses_783.feather MD5 checksum should be f8f1b97c9d4b0ea9b4c8b287f6b99091
Get-FileHash flywire_synapses_783.feather -Algorithm MD5 
```
